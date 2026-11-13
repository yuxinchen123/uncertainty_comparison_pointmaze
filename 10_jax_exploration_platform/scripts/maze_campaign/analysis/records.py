"""Read a maze-campaign run folder's shards into one record per trained policy, grouped into configurations.

A run folder holds one shard per unit, `data/<unit id>.jsonl`, written by `scripts/run_training.py` with the
episode log on. A unit trains many policies at once: its `unit_start` record says which cell (settings) and which
seed each policy has, and every `episode_window` record carries one number per policy for each sum.

Which records of a shard count:

- a unit that was started again from zero appends a new `unit_start`; only the records after the last
  `unit_start` count;
- a unit that was resumed from a checkpoint appends a `unit_resume` and continues its windows, so the windows
  after the last `unit_start` are one unbroken sequence;
- a last line that is still being written (no line end yet) is left out.

How a unit's method is named, first rule that applies:

1. an entry of `methods` in the run folder's `report_settings.json` whose conditions the unit meets;
2. the `method` of the unit's specification file `specifications/<unit id>.json`, which the fleet's
   `build_units.py` writes (`cleanrl-ppo-rnd` reads "CleanRL PPO-RND"; other words keep their spelling with
   spaces for hyphens);
3. the unit's bonus (`rnd_next_state` reads "CleanRL PPO-RND", `none` reads "PPO without a bonus").

The run folder's optional `report_settings.json` (every key optional):

- `title`: the report's title;
- `other_run_folders`: run folders whose shards join this report (for example the baseline's);
- `baseline_method`: the method every other method is compared with (default "CleanRL PPO-RND");
- `methods`: a list of `{"name": words, "bonus": the unit's bonus (optional), "unit_id_pattern": a shell
  pattern on the unit id (optional), "reference": true for a row that never decides success (optional)}`;
- `reference_methods`: names (in words) of further methods that never decide success;
- `exclude_units`: shell patterns of unit ids that are left out of the report;
- `repeated_configurations`: "show" when units repeat a configuration on the same seeds on purpose. The report
  then uses, for each such seed, the unit that finished first (`split_repeats`) and lists the other runs in a
  section of their own. Without the key a repeat is an error, because an unplanned repeat is a mistake.
- `discovery_units`: shell patterns of the unit ids of the sweep made for the secondary metric (the mean capped
  first-goal step), for example `["development-for-discovery_*", "confirmation-for-discovery_*"]`. These units
  are read for the secondary metric only: their configurations enter its choice, its tables and its comparison,
  and stay out of every table, choice and figure of the primary metric, so that adding them never changes a
  configuration that was chosen for the whole-curve mean and already confirmed. Every other unit is read for both
  metrics. Where such a unit repeats a configuration and seed of another unit, the other unit is the one used.

Development or confirmation: a policy belongs to the confirmation when its seed is 30000 or above. A unit whose
specification names a phase starting with `development` or `confirmation` must agree with its seeds, or the
reader stops with an error.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import json
import math
from pathlib import Path

import numpy as np

ENVIRONMENT_NAMES = {('pointmaze', 'medium'): 'PointMaze Medium', ('pointmaze', 'large'): 'PointMaze Large',
                     ('antmaze', 'umaze'): 'AntMaze U-maze', ('antmaze', 'medium'): 'AntMaze Medium',
                     ('antmaze', 'large'): 'AntMaze Large'}
ENVIRONMENT_ORDER = tuple(ENVIRONMENT_NAMES.values())
REWARD_SETTING_NAMES = {'arrival_plus_one': 'reward 1 on arrival', 'step_cost': 'step cost'}
REWARD_SETTING_ORDER = tuple(REWARD_SETTING_NAMES)
# The return of an episode that never reaches the goal, in the reward the learner sees.
NEVER_REACHED_RETURN = {'arrival_plus_one': 0.0, 'step_cost': -1.0}
BASELINE_METHOD = 'CleanRL PPO-RND'
BONUS_METHOD_NAMES = {'rnd_next_state': BASELINE_METHOD, 'none': 'PPO without a bonus'}
# The words of a method in the fleet's unit specifications whose reading is not just spaces for hyphens.
SPECIFICATION_METHOD_NAMES = {'cleanrl-ppo-rnd': BASELINE_METHOD}
# Seeds below this are development seeds; seeds from it on are the fresh confirmation seeds (program.md).
CONFIRMATION_FIRST_SEED = 30000
# Trainer fields that do not name a configuration: which policies a process holds and how they are seeded, the
# lists its cells come from, the reward mapping and the rollout width (named separately), logging and speed
# switches, and the run's length (read from the unit's iterations). Every other field is a setting, so two
# units that differ in a field this list does not know are never pooled.
NOT_SETTINGS = frozenset({
    'n_copies', 'copies_per_group', 'copy_seed_offset', 'base_seed', 'sweep_seed_mode',
    'learning_rates', 'betas', 'parameter_settings', 'setting_names',
    'ext_reward_scale', 'ext_reward_offset', 'n_envs',
    'diagnostics', 'episode_log', 'track_coverage', 'flat_params', 'hoist_rollout', 'scan_unroll',
    'update_unroll', 'batch_stats_f32', 'gather_minibatches_per_epoch', 'total_iterations'})
SETTING_WORDS = {'int_coef': 'intrinsic weight', 'learning_rate': 'learning rate', 'ent_coef': 'entropy coefficient',
                 'clip_coef': 'clip coefficient', 'vf_coef': 'value coefficient', 'gamma_int': 'intrinsic discount',
                 'gamma_ext': 'extrinsic discount', 'gae_lambda': 'advantage decay', 'ext_coef': 'extrinsic weight',
                 'num_steps': 'rollout steps', 'update_epochs': 'update epochs', 'num_minibatches': 'minibatches',
                 'max_grad_norm': 'gradient clip', 'prime_iterations': 'random-action rollouts before training',
                 'lr_anneal_fraction_of_run': 'share of the run over which the learning rate falls to zero'}
# Settings named first, in this order, when a row is described; the others follow in alphabetical order.
SETTING_ORDER = ('int_coef', 'learning_rate', 'ent_coef', 'clip_coef', 'gamma_int')
WINDOW_SUMS = ('episodes', 'arrivals', 'return_sum', 'environment_return_sum', 'length_sum', 'steps_to_goal_sum')
REPORT_SETTING_KEYS = {'title', 'other_run_folders', 'baseline_method', 'methods', 'reference_methods',
                       'exclude_units', 'repeated_configurations', 'discovery_units'}


@dataclass(frozen=True, eq=False)
class Policy:
    """One trained policy: what it is, and its sums over the training episodes that end in each logging window."""
    unit_id: str
    environment: str               # in words, for example 'PointMaze Large'
    environment_identity: str      # the unit's whole environment configuration, to refuse pooling two environments
    reward_setting: str            # 'arrival_plus_one' or 'step_cost'
    method: str                    # in words
    environments_per_policy: int
    planned_steps: int             # environment steps per policy the unit trains for
    settings: tuple                # ((name, value), ...) in name order
    seed: int
    base_seed: int                 # the unit's base seed: two policies are the same seed only when it agrees too
    step_limit: int
    complete: bool                 # the unit wrote its completion record
    steps: np.ndarray              # [windows] environment steps of this policy at each window's end
    episodes: np.ndarray           # [windows] episodes that ended in the window
    arrivals: np.ndarray           # [windows] of those, the episodes that ended at the goal
    return_sum: np.ndarray         # [windows] sum of their returns in the reward the learner sees
    environment_return_sum: np.ndarray   # [windows] the same in the unmapped reward (1 on arrival)
    length_sum: np.ndarray         # [windows] sum of their lengths in steps
    steps_to_goal_sum: np.ndarray  # [windows] sum of the lengths of the episodes that ended at the goal
    coverage: np.ndarray           # [windows] share of open maze cells reached so far; None when not recorded
    first_goal_step: int           # training step of the first arrival; -1 when none is recorded so far

    @property
    def stage(self):
        """Which seeds this policy belongs to: 'development' or 'confirmation'."""
        return 'confirmation' if self.seed >= CONFIRMATION_FIRST_SEED else 'development'

    @property
    def seed_key(self):
        """What two policies must share to count as trained on the same seed."""
        return (self.base_seed, self.seed)

    @property
    def never_reached_return(self):
        """The return of an episode that never reaches the goal, in the reward the learner sees."""
        return NEVER_REACHED_RETURN[self.reward_setting]

    @property
    def configuration_key(self):
        """Everything but the seed: policies with equal keys are seeds of one configuration."""
        return (self.environment, self.reward_setting, self.method, self.environments_per_policy,
                self.planned_steps, self.settings)


@dataclass(frozen=True, eq=False)
class Configuration:
    """The policies that differ only by their seed, in seed order."""
    environment: str
    reward_setting: str
    method: str
    environments_per_policy: int
    planned_steps: int
    settings: tuple
    policies: tuple

    def setting(self, name):
        """One setting's value by name."""
        return dict(self.settings)[name]

    @property
    def key(self):
        """Everything its policies share: two groups with equal keys are the same configuration."""
        return (self.environment, self.reward_setting, self.method, self.environments_per_policy,
                self.planned_steps, self.settings)


@dataclass(frozen=True, eq=False)
class Run:
    """Everything read for one report: policies, one summary per unit, and the report settings."""
    policies: tuple
    units: tuple
    settings: dict

    @property
    def baseline_method(self):
        """The method every other method is compared with."""
        return self.settings.get('baseline_method', BASELINE_METHOD)

    @property
    def reference_methods(self):
        """Methods shown for the record that never decide success."""
        listed = {method['name'] for method in self.settings.get('methods', []) if method.get('reference')}
        return listed | set(self.settings.get('reference_methods', []))


def read_lines(path):
    """The complete JSON lines of a shard; a last line still being written (no line end yet) is left out."""
    # before: 'a\nb\nc' while the unit is writing line c; after: the records of lines a and b.
    lines = Path(path).read_text().split('\n')[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def latest_attempt(records):
    """The records from the last `unit_start` on; an empty list when the unit has not written one yet."""
    # before: [job, unit_start, window 1, job, unit_start, window 1, window 2]; after: the last four records.
    starts = [index for index, record in enumerate(records) if record.get('record') == 'unit_start']
    return records[starts[-1]:] if starts else []


def environment_name(start):
    """The unit's environment in words, from its family and map."""
    key = (start['environment'].split('_')[0], start['environment_config']['map_name'])
    if key not in ENVIRONMENT_NAMES:
        raise ValueError(f"Unit {start['unit_id']} trains in {start['environment']}, not one of the study's five environments.")
    return ENVIRONMENT_NAMES[key]


def reward_setting_of(scale, offset, step_limit):
    """Name the reward setting from the scale and offset the learner's reward has."""
    # before: (1.0, 0.0, 400) -> 'arrival_plus_one'; (0.0025, -0.0025, 400) -> 'step_cost'.
    if math.isclose(scale, 1.0) and offset == 0.0:
        return 'arrival_plus_one'
    if math.isclose(scale, 1.0 / step_limit) and math.isclose(offset, -1.0 / step_limit):
        return 'step_cost'
    raise ValueError(f'Reward scale {scale} and offset {offset} with a step limit of {step_limit} are neither '
                     'reward 1 on arrival nor the step cost.')


def method_name(start, methods, specified=None):
    """The unit's method in words: a listed method it meets, else its specification's method, else its bonus."""
    # before: bonus 'rnd_next_state', no listed method, specification method 'cleanrl-ppo-rnd' -> 'CleanRL PPO-RND';
    #         specification method 'even-start' -> 'even start'; no specification -> the bonus's own name.
    for method in methods:
        if (method.get('bonus', start['bonus']) == start['bonus']
                and fnmatch.fnmatchcase(start['unit_id'], method.get('unit_id_pattern', '*'))):
            return method['name']
    if specified is not None:
        return SPECIFICATION_METHOD_NAMES.get(specified, specified.replace('-', ' '))
    return BONUS_METHOD_NAMES.get(start['bonus'], start['bonus'].replace('_', ' '))


def specification_of(shard):
    """The unit's specification file beside the run folder's data, as a dictionary; empty without the file."""
    path = Path(shard).parent.parent / 'specifications' / f'{Path(shard).stem}.json'
    return json.loads(path.read_text()) if path.exists() else {}


def check_phase(unit_id, phase, seeds):
    """Refuse a unit whose phase name says development or confirmation while its seeds say the other."""
    # before: ('development-for-discovery', seeds 0 to 15) and ('confirmation', seeds 30000 on) pass, a phase
    #         that starts with neither word passes; ('confirmation-for-discovery', seed 12) is refused.
    named = next((stage for stage in ('development', 'confirmation') if (phase or '').startswith(stage)), None)
    other = [seed for seed in seeds if named is not None
             and ('confirmation' if seed >= CONFIRMATION_FIRST_SEED else 'development') != named]
    if other:
        raise ValueError(f'Unit {unit_id} has the phase {phase}, but its seed {other[0]} is a '
                         f"{'confirmation' if other[0] >= CONFIRMATION_FIRST_SEED else 'development'} seed "
                         f'(confirmation seeds start at {CONFIRMATION_FIRST_SEED}).')


def cell_values(start, cell):
    """One policy's trainer values by name: the unit's own values, replaced by its cell's."""
    # before: trainer_config {'learning_rate': 3e-4, 'int_coef': 1.0, 'n_copies': 48, ...},
    #         cell names ['learning_rate', 'int_coef'], cell [1e-4, 16.0]
    # after:  {'learning_rate': 1e-4, 'int_coef': 16.0, 'n_copies': 48, ...}
    return {**start['trainer_config'], **dict(zip(start['cell_setting_names'], cell))}


def settings_of(start, values):
    """The settings that name a configuration, in name order, from one policy's trainer values."""
    # before: {'learning_rate': 1e-4, 'int_coef': 16.0, 'n_copies': 48, 'betas': [0.0625, 1.0, 16.0], ...}
    # after:  (('int_coef', 16.0), ('learning_rate', 1e-4), ('lr_anneal_fraction_of_run', 1.0), ...)
    settings = {name: tuple(value) if isinstance(value, list) else value
                for name, value in values.items() if name not in NOT_SETTINGS}
    settings['lr_anneal_fraction_of_run'] = start['lr_anneal_iterations'] / start['iterations']
    return tuple(sorted(settings.items()))


def window_arrays(start, windows):
    """The windows' per-policy lists as arrays [windows, policies], after checking the windows are consecutive."""
    # before: one record per window, each field a list with one number per policy;
    # after:  {'episodes': array [windows, policies], ..., 'steps': array [windows]}.
    expected = 1
    for window in windows:
        if window['first_iteration'] != expected:
            raise ValueError(f"Unit {start['unit_id']}: a window starts at iteration {window['first_iteration']} "
                             f'where iteration {expected} was expected; its windows are not consecutive.')
        expected = window['last_iteration'] + 1
    missing = [name for name in WINDOW_SUMS if f'{name}_per_copy' not in windows[0]]
    if missing:
        raise ValueError(f"Unit {start['unit_id']} has no episode log ({missing[0]}_per_copy is missing); "
                         'train it with --episode-log.')
    arrays = {name: np.asarray([window[f'{name}_per_copy'] for window in windows], np.float64)
              for name in WINDOW_SUMS}
    arrays['steps'] = np.asarray([window['env_steps_per_copy'] for window in windows], np.int64)
    arrays['coverage'] = (np.asarray([window['coverage_per_copy'] for window in windows], np.float64)
                          if 'coverage_per_copy' in windows[0] else None)
    # The last window holds each policy's first arrival so far; a policy with none has -1 or null.
    arrays['first_goal_step'] = np.asarray([-1 if step is None else step
                                            for step in windows[-1]['first_goal_step_per_copy']], np.int64)
    return arrays


def read_unit(path, methods=()):
    """One shard as (a summary of the unit, its policies); a unit with no window yet has no policies."""
    return unit_from_records(path, read_lines(path), methods)


def unit_from_records(path, records, methods=()):
    """A shard's records, already read, as (a summary of the unit, its policies); for a caller that reads a shard once."""
    attempt = latest_attempt(records)
    summary = {'unit_id': Path(path).stem, 'state': 'no start record yet', 'environment': None, 'method': None,
               'environments_per_policy': None, 'reward_settings': [], 'policies': 0, 'planned_steps': None,
               'windows': 0, 'steps_reached': 0, 'continued_from_a_checkpoint': 0, 'respawned_episodes': None,
               'finished': None,
               'started_from_zero_again': max(0, sum(record.get('record') == 'unit_start' for record in records) - 1)}
    if not attempt:
        return summary, []
    start = attempt[0]
    windows = [record for record in attempt if record['record'] == 'episode_window']
    completion = [record for record in attempt if record['record'] == 'unit_complete']
    environment = start['environment_config']
    if environment.get('continuing_task', True) or environment.get('reward_shift', 0.0) != 0:
        raise ValueError(f"Unit {start['unit_id']}: the study's measures need episodes that end at the goal and "
                         'an environment reward of 1 on arrival and 0 elsewhere.')
    step_limit = environment['max_episode_steps']
    planned = start['iterations'] * start['rollout_steps'] * start['envs_per_copy']
    cells = [cell_values(start, cell) for cell in start['cell_settings']]
    rewards = [reward_setting_of(values['ext_reward_scale'], values['ext_reward_offset'], step_limit)
               for values in cells]
    specification = specification_of(path)
    check_phase(start['unit_id'], specification.get('phase'), [int(seed) for seed in start['copy_seed_index']])
    summary.update(
        unit_id=start['unit_id'], environment=environment_name(start),
        method=method_name(start, methods, specification.get('method')),
        environments_per_policy=start['envs_per_copy'], policies=start['copies'], planned_steps=planned,
        reward_settings=sorted(set(rewards), key=REWARD_SETTING_ORDER.index), windows=len(windows),
        state='complete' if completion else 'unfinished',
        continued_from_a_checkpoint=sum(record['record'] == 'unit_resume' for record in attempt),
        # When the unit finished, as its completion record says (the time the record was written when it names none).
        finished=(completion[0].get('attempt_finished') or completion[0].get('written_at')) if completion else None,
        respawned_episodes=completion[0].get('nan_respawned_episodes') if completion else None)
    if not windows:
        return summary, []
    arrays = window_arrays(start, windows)
    summary['steps_reached'] = int(arrays['steps'][-1])

    # One policy per copy: its cell's settings, its seed, and its column of every window array.
    policies = []
    for copy, (cell_index, seed) in enumerate(zip(start['copy_cell'], start['copy_seed_index'])):
        policies.append(Policy(
            unit_id=start['unit_id'], environment=summary['environment'],
            environment_identity=json.dumps(environment, sort_keys=True), reward_setting=rewards[cell_index],
            method=summary['method'], environments_per_policy=start['envs_per_copy'], planned_steps=planned,
            settings=settings_of(start, cells[cell_index]), seed=int(seed),
            base_seed=start['trainer_config']['base_seed'], step_limit=step_limit, complete=bool(completion),
            steps=arrays['steps'], **{name: arrays[name][:, copy] for name in WINDOW_SUMS},
            coverage=None if arrays['coverage'] is None else arrays['coverage'][:, copy],
            first_goal_step=int(arrays['first_goal_step'][copy])))
    return summary, policies


def read_report_settings(run_dir):
    """The run folder's optional report_settings.json; an absent file means every default."""
    path = Path(run_dir) / 'report_settings.json'
    settings = json.loads(path.read_text()) if path.exists() else {}
    unknown = set(settings) - REPORT_SETTING_KEYS
    if unknown:
        raise ValueError(f'{path} holds unknown keys {sorted(unknown)}; allowed: {sorted(REPORT_SETTING_KEYS)}.')
    if settings.get('repeated_configurations', 'show') != 'show':
        raise ValueError(f"{path}: repeated_configurations is {settings['repeated_configurations']!r}; the only "
                         'value is "show" (leave the key out to refuse repeated configurations).')
    return settings


def load_run(run_dir):
    """Read every shard of the run folder, and of the other folders its report settings name."""
    run_dir = Path(run_dir)
    settings = read_report_settings(run_dir)
    folders = [run_dir] + [Path(folder) for folder in settings.get('other_run_folders', [])]
    units, policies = [], []
    for folder in folders:
        shards = sorted((folder / 'data').glob('*.jsonl'))
        if not shards:
            raise ValueError(f'{folder} holds no shard under data/.')
        for shard in shards:
            if any(fnmatch.fnmatchcase(shard.stem, pattern) for pattern in settings.get('exclude_units', ())):
                continue
            summary, members = read_unit(shard, settings.get('methods', ()))
            units.append({**summary, 'folder': folder.name})
            policies += members

    # One environment name must mean one environment: pooling two would compare different tasks.
    identities = defaultdict(set)
    for policy in policies:
        identities[policy.environment].add(policy.environment_identity)
    mixed = sorted(name for name, found in identities.items() if len(found) > 1)
    if mixed:
        raise ValueError(f'The units hold different environment configurations under the name {mixed[0]}: '
                         f'{sorted(identities[mixed[0]])}.')
    return Run(policies=tuple(policies), units=tuple(units), settings=settings)


def finish_order(unit_id, finished):
    """A sort key that puts the unit that finished first in front: finished units by time, then the others by name."""
    # before: {'b': '2026-11-12T10:00:00-05:00', 'a': '2026-11-12T09:00:00-08:00', 'c': None}
    # after:  'b' (15:00 UTC) sorts before 'a' (17:00 UTC), and 'c', which has not finished, after both.
    stamp = finished[unit_id]
    moment = datetime.fromisoformat(stamp).astimezone(timezone.utc) if stamp else datetime.max.replace(tzinfo=timezone.utc)
    return (moment, unit_id)


def use_order(unit_id, finished, last=()):
    """A sort key that puts the unit a report uses in front: the first to finish, the units named in `last` after all others."""
    # before: units 'a' (finished 10:00), 'b' (finished 09:00) and 'c' (finished 08:00), last = {'c'}
    # after:  'b', 'a', 'c': a unit of `last` never takes the place of another unit, however early it finished.
    return (unit_id in last, *finish_order(unit_id, finished))


def split_repeats(policies, finished, last=()):
    """Policies split into those a report uses and the repeats: of a seed trained by several units, the first to finish is used."""
    # before: units A (finished 10:00) and B (finished 09:00) both hold seeds 0 and 1 of one configuration, and A
    #         also holds another configuration; after: (B's two policies and A's other policies, A's two repeats).
    # `finished` maps every unit id to its finishing time (None when it has not finished); a unit named in `last`
    # is used only for the seeds no other unit trained (`use_order`).
    groups = defaultdict(list)
    for policy in policies:
        groups[(policy.configuration_key, policy.seed_key)].append(policy)
    repeats = []
    for (_, (_, seed)), members in groups.items():
        units = [policy.unit_id for policy in members]
        if len(set(units)) != len(units):
            raise ValueError(f'Seed {seed} was trained more than once for one configuration inside the unit '
                             f'{next(unit for unit in units if units.count(unit) > 1)}.')
        repeats += sorted(members, key=lambda policy: use_order(policy.unit_id, finished, last))[1:]
    left_out = {id(policy) for policy in repeats}
    return [policy for policy in policies if id(policy) not in left_out], repeats


def unit_words(unit_id):
    """A unit's name in words: its parts separated by commas, the words of a part by spaces."""
    # before: 'development-at-the-budget_pointmaze-medium_seeds-0-to-15'
    # after:  'development at the budget, pointmaze medium, seeds 0 to 15'.
    return unit_id.replace('_', ', ').replace('-', ' ')


def configurations(policies):
    """Group policies that share everything but the seed; each group's policies are in seed order."""
    # before: 48 policies of one unit (3 cells x 16 seeds); after: 3 configurations of 16 policies each.
    groups = defaultdict(list)
    for policy in policies:
        groups[policy.configuration_key].append(policy)
    result = []
    for members in groups.values():
        members = sorted(members, key=lambda policy: policy.seed_key)
        seeds = [policy.seed_key for policy in members]
        if len(set(seeds)) != len(seeds):
            repeated = next(seed for seed in seeds if seeds.count(seed) > 1)
            units = sorted({policy.unit_id for policy in members if policy.seed_key == repeated})
            raise ValueError(f'Seed {repeated[1]} was trained more than once for one configuration, in units {units}.')
        first = members[0]
        result.append(Configuration(first.environment, first.reward_setting, first.method,
                                    first.environments_per_policy, first.planned_steps, first.settings,
                                    tuple(members)))
    return result


def varying_settings(groups):
    """The names of the settings whose values differ among configurations, the main ones first."""
    # before: three configurations that differ only in int_coef; after: ['int_coef'].
    # A setting that some configurations do not hold at all (a field added later) also counts as differing.
    names = sorted({name for group in groups for name, _ in group.settings})
    differing = [name for name in names if len({dict(group.settings).get(name, 'not set') for group in groups}) > 1]
    return sorted(differing, key=lambda name: (SETTING_ORDER.index(name) if name in SETTING_ORDER
                                               else len(SETTING_ORDER), name))


def describe_settings(configuration, names):
    """The named settings of a configuration in words, for a table row or a legend."""
    # before: names ['int_coef', 'learning_rate']; after: 'intrinsic weight 16, learning rate 0.0001'.
    values = dict(configuration.settings)
    parts = []
    for name in names:
        value = values.get(name, 'not set')
        text = f'{value:g}' if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
        parts.append(f"{SETTING_WORDS.get(name, name.replace('_', ' '))} {text}")
    return ', '.join(parts)
