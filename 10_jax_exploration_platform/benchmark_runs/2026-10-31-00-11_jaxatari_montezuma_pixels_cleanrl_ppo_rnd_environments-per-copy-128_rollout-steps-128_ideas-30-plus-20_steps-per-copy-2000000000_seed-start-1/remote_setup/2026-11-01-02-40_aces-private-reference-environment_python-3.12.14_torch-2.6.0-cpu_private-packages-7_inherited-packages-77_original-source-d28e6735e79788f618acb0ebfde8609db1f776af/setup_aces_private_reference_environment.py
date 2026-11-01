"""Create one private CPU-reference layer and prepare metadata without changing the base.

Only a fresh system-site venv and seven offline locked wheels may be written.
Original assets, source checkouts, failures and the base JAX environment remain
inputs. No numerical imports, submissions, releases or connection retries exist.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import shlex
import signal
import socket
import subprocess
import sys
import threading
import types


PLAN_SHA = '6aa2df602816394ea46f59393ef666428dbdd385c99e2f8b19600426fe9852cd'
BASE_SHA = '95e99f9c213b23c6f8f82c8e8123d25d2a71acf3f4166217a872b0d1126e410a'
ASSET_SHA = '8e8e784b66b16d06daa48f7bbc57d5fbdff50b950f1b7ad942713213ba63aa8d'
OLD_PYTHON = '/scratch/user/u.sl359075/envs/rnd_atari_pixels_jax/bin/python'
BASE = Path('/scratch/user/u.sl359075/rnd_atari_pixels')
COMMANDS = []
MAX_FILE = 16 * 1024 * 1024
BOOTSTRAP = '''import hashlib,json,sys
line=sys.stdin.buffer.readline(2097153)
if len(line)>2097152 or not line.endswith(b"\\n"): raise ValueError("bounded setup header required")
payload=json.loads(line)
if hashlib.sha256(payload["driver_source"].encode()).hexdigest()!=sys.argv[1]: raise ValueError("setup source SHA changed")
namespace={"__name__":"aces_private_environment_remote","__file__":"<reviewed private environment setup>"}
exec(compile(payload["driver_source"],"<reviewed private environment setup>","exec"),namespace)
try: namespace["remote_setup"](payload)
except Exception as error:
 record={"kind":"aces_private_environment_setup_failure","error_type":type(error).__name__,"error":str(error),"commands":namespace["COMMANDS"],"automatic_retry_permitted":False,"jobs_submitted":0}
 sys.stdout.buffer.write((json.dumps(record)+"\\n").encode());sys.stdout.buffer.flush()
 raise
'''


def digest(contents):
    """Identify exact reviewed bytes rather than a reserialized input."""
    return hashlib.sha256(contents).hexdigest()


def module_from_source(name, source, expected):
    """Load only a hash-pinned metadata utility without invoking its CLI main."""
    if digest(source.encode()) != expected:
        raise ValueError('a reviewed metadata utility changed')
    module = types.ModuleType(name)
    module.__file__ = '<reviewed metadata utility>'
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def checked_payload(payload):
    """Bind all source, wheel, source-admission and environment observations to the frozen setup plan."""
    if digest(payload['plan'].encode()) != PLAN_SHA:
        raise ValueError('the private environment setup plan changed')
    plan = json.loads(payload['plan'])
    for name in ('base_driver', 'asset_driver', 'metadata_helper', 'original_plan', 'asset_plan',
            'full_base_inventory', 'wheel_verification', 'complete_dependency_preflight'):
        if digest(payload[name].encode()) != plan[name]['sha256']:
            raise ValueError('a setup input changed: ' + name)
    base = module_from_source('private_setup_base', payload['base_driver'], BASE_SHA)
    assets = module_from_source('private_setup_assets', payload['asset_driver'], ASSET_SHA)
    metadata = module_from_source('private_setup_metadata', payload['metadata_helper'], plan['metadata_helper']['sha256'])
    original = base.checked_plan(payload['original_plan'].encode())
    asset_plan = json.loads(payload['asset_plan'])
    prefix, root = Path(plan['private_prefix']), Path(plan['setup_directory'])
    if (plan.get('kind') != 'aces_private_cpu_reference_environment_setup_plan'
            or prefix.parent != Path('/scratch/user/u.sl359075/envs') or root.parent != BASE / 'remote_setup'
            or plan['private_python'] != str(prefix / 'bin/python') or plan['old_python'] != OLD_PYTHON
            or plan['base_prefix'] != str(Path(OLD_PYTHON).parent.parent) or plan['asset_view'] != asset_plan['asset_view']
            or plan['asset_record'] != asset_plan['asset_record'] or len(plan['wheels']) != 7
            or json.loads(payload['complete_dependency_preflight'])['proposed_closure']['active_dependency_count'] != 128):
        raise ValueError('the fixed private environment, asset view or complete preflight binding differs')
    return plan, base, assets, metadata, original


def local_payload(path):
    """Verify exact local inputs and seven whole wheels before any SSH process can start."""
    path = Path(path)
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise ValueError('an exact absolute ordinary setup plan path is required')
    contents = path.read_bytes()
    if digest(contents) != PLAN_SHA:
        raise ValueError('the local setup plan changed')
    plan = json.loads(contents)
    payload = {'plan': contents.decode(), 'driver_source': Path(__file__).read_text()}
    for name in ('base_driver', 'asset_driver', 'metadata_helper', 'original_plan', 'asset_plan',
            'full_base_inventory', 'wheel_verification', 'complete_dependency_preflight'):
        row = plan[name]
        source = Path(row['path'])
        if not source.is_absolute() or source != source.resolve(strict=True) or not source.is_file():
            raise ValueError('a local setup input is not its exact ordinary path')
        payload[name] = source.read_text()
    plan, base, assets, metadata, original = checked_payload(payload)
    base.validate_local(plan['original_plan']['path'])
    wheels = [metadata.wheel_metadata(row) for row in plan['wheels']]
    baseline = json.loads(payload['full_base_inventory'])
    preflight = json.loads(payload['complete_dependency_preflight'])
    proof = metadata.proposed_closure(baseline['distributions'], wheels, preflight['target_environment'])
    if proof['active_dependency_count'] != 128:
        raise ValueError('the fixed actual proposed dependency closure changed')
    return payload, plan, base, assets, metadata, original


def installation_environment():
    """Disable inherited pip destinations/configuration without changing HOME or importing numerical libraries."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith('PIP_')}
    environment['PIP_CONFIG_FILE'] = os.devnull
    environment['PYTHONNOUSERSITE'] = '1'
    return environment


def effective_plan(base, plan, original):
    """Select only the new interpreter and fresh outputs while retaining all original numerical settings."""
    effective = json.loads(json.dumps(original))
    effective.update(remote_packet_directory=plan['native_packet_directory'], remote_data_unit=plan['native_data_unit'])
    parameters = effective['native_prepare_parameters']
    parameters.update({'python': plan['private_python'], 'output-dir': plan['native_packet_directory'], 'asset-record': plan['asset_record']})
    effective['native_prepare_command'] = [plan['private_python'], '-I', '-B',
        str(Path(original['native_helper_root']) / base.PREFIX / 'aces_slurm_site_validation.py'), 'prepare',
        *[item for key, value in parameters.items() for item in ('--' + key, value)]]
    return effective


def pip_command(plan, wheelhouse, requirements):
    """Install only seven hash-locked offline binary wheels into the exact private prefix without uninstalling base packages."""
    return [plan['private_python'], '-I', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install',
        '--no-index', '--no-deps', '--require-hashes', '--only-binary=:all:', '--ignore-installed', '--no-compile',
        '--no-cache-dir', '--prefix=' + plan['private_prefix'], '--find-links=' + str(wheelhouse), '-r', str(requirements)]


def execute(base, command, root, *, environment=None, timeout=240):
    """Capture each metadata/setup command and its raw output under this one fresh setup directory."""
    probe_codes = [] if CURRENT_METADATA_SOURCE is None else [CURRENT_METADATA_SOURCE + '\nprint(json.dumps(snapshot(' + repr(value) + '),sort_keys=True))\n' for value in (False, True)]
    is_probe = len(command) == 5 and command[0] in (OLD_PYTHON, CURRENT_PRIVATE_PYTHON) and command[1:4] == ['-I', '-B', '-c'] and command[4] in probe_codes
    is_venv = CURRENT_PLAN is not None and command == [OLD_PYTHON, '-I', '-B', '-m', 'venv', '--without-pip', '--system-site-packages', '--copies', CURRENT_PLAN['private_prefix']]
    is_install = CURRENT_PLAN is not None and command == pip_command(CURRENT_PLAN, root / 'wheels', root / 'requirements.lock')
    is_check = command == [CURRENT_PRIVATE_PYTHON, '-I', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'check']
    if not (is_probe or is_venv or is_install or is_check or command == CURRENT_PREPARE or command == CURRENT_VERIFICATION):
        raise ValueError('only the exact reviewed metadata probes/venv/offline-install/check/prepare commands may execute')
    result = subprocess.run(command, env=environment, capture_output=True, timeout=timeout)
    number = len(COMMANDS) + 1
    stdout, stderr = root / ('command-' + str(number).zfill(2) + '.stdout'), root / ('command-' + str(number).zfill(2) + '.stderr')
    for path, contents in ((stdout, result.stdout), (stderr, result.stderr)):
        if len(contents) > MAX_FILE:
            raise ValueError('a setup command exceeded its bounded captured output')
        with path.open('xb') as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
    COMMANDS.append({'argv': command, 'returncode': result.returncode, 'stdout': base.file_record(stdout), 'stderr': base.file_record(stderr)})
    if result.returncode:
        raise ValueError('a private environment metadata/setup command failed: ' + shlex.join(command[:6]))
    return result.stdout.decode()


CURRENT_PRIVATE_PYTHON = None
CURRENT_METADATA_SOURCE = None
CURRENT_PLAN = None
CURRENT_PREPARE = None
CURRENT_VERIFICATION = None


def probe(base, plan, payload, root, python, *, hash_shared=False):
    """Run the selected actual interpreter only for package/path/hash/marker metadata."""
    source = payload['metadata_helper'] + '\nprint(json.dumps(snapshot(' + repr(hash_shared) + '),sort_keys=True))\n'
    return json.loads(execute(base, [python, '-I', '-B', '-c', source], root, environment=installation_environment()))


def verification_command(plan, original):
    """Verify native preparation and actual software only in a fresh selected private interpreter."""
    helper = str(Path(original['native_helper_root']) / '10_jax_exploration_platform/scripts/atari_pixel_campaign/aces_slurm_site_validation.py')
    source = """import importlib.util,json,sys
from pathlib import Path
path=Path(HELPER)
spec=importlib.util.spec_from_file_location('selected_private_preparation_verification',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
directory,unit,plan=module.prepared(Path(DIRECTORY))
if str(Path(sys.executable).absolute())!=SELECTED or plan['python']!=SELECTED or not sys.dont_write_bytecode:
 raise ValueError('preparation verification requires the selected private Python')
software=module.checked_software(plan)
if software['installed_backend']['packages']['torch']!='2.6.0+cpu':
 raise ValueError('selected private Python lacks exact CPUreference Torch metadata')
loaded=[name for name in ('jax','jaxlib','numpy','torch','flax','optax') if name in sys.modules]
if loaded: raise ValueError('software metadata verification imported numerical packages')
print(json.dumps({'kind':'selected_private_python_preparation_verification','passed':True,'python':SELECTED,'software_at_prepare':module.receipt(directory/'software_at_prepare.json'),'numerical_imports':loaded}))
"""
    source = source.replace('HELPER', json.dumps(helper)).replace('DIRECTORY', json.dumps(plan['native_packet_directory'])).replace('SELECTED', json.dumps(plan['private_python']))
    return [plan['private_python'], '-I', '-B', '-c', source]


def verify_preparation(base, plan, original, root, environment):
    """Reject a verification performed by the old parent or one that lacks the new private software view."""
    result = json.loads(execute(base, verification_command(plan, original), root, environment=environment))
    if (result.get('kind') != 'selected_private_python_preparation_verification' or result.get('passed') is not True
            or result.get('python') != plan['private_python'] or result.get('numerical_imports') != []):
        raise ValueError('software verification did not use the selected private interpreter')
    return result


def check_prefix(plan, before, private, *, installed=False):
    """Prove system-site inheritance and exact seven private overrides before allowing another prepare."""
    prefix = Path(plan['private_prefix'])
    if (private['prefix'] != str(prefix) or private['base_prefix'] != plan['base_prefix']
            or private['purelib'] != str(prefix / 'lib/python3.12/site-packages') or private['python'] != plan['private_python']
            or private['version'] != '3.12.14' or private['numerical_imports'] or not private['dont_write_bytecode']):
        raise ValueError('the private interpreter does not have its exact expected prefix/inheritance')
    expected = set(PRIVATE_VERSIONS) if installed else set()
    actual = {row['name'] for row in private['distributions'] if Path(row['metadata_path']).is_relative_to(prefix)}
    if actual != expected:
        raise ValueError('private distributions differ from the exact seven allowed overrides')
    for name, row in before['effective'].items():
        if name not in expected and private['effective'].get(name) != row:
            raise ValueError('a shared base package provider/version changed under the private interpreter')
    if installed:
        for name, version in PRIVATE_VERSIONS.items():
            if private['effective'][name]['version'] != version or not Path(private['effective'][name]['metadata_path']).is_relative_to(prefix):
                raise ValueError('a private CPU/reference package has another version/provider')


PRIVATE_VERSIONS = {'torch': '2.6.0+cpu', 'filelock': '4.2.4', 'networkx': '3.7', 'jinja2': '3.1.6',
    'sympy': '1.13.1', 'mpmath': '1.3.0', 'markupsafe': '3.0.3'}


def timeout_handler(signum, frame):
    """Stop this own setup process after its bounded metadata-only deadline."""
    raise TimeoutError('private reference environment setup exceeded 1200 seconds')


def remote_setup(payload):
    """Create only a fresh private venv, run full metadata checks, then prepare a fresh native packet."""
    global CURRENT_PRIVATE_PYTHON, CURRENT_METADATA_SOURCE, CURRENT_PLAN, CURRENT_PREPARE, CURRENT_VERIFICATION
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1200)
    try:
        if (os.getuid() != 56626 or pwd.getpwuid(os.getuid()).pw_name != 'u.sl359075'
                or socket.gethostname().split('.')[0] != 'alogin1' or str(Path(sys.executable).absolute()) != OLD_PYTHON or not sys.dont_write_bytecode):
            raise ValueError('run setup through actual ACES old owner/interpreter with -I -B')
        plan, base, assets, metadata, original = checked_payload(payload)
        CURRENT_PRIVATE_PYTHON = plan['private_python']
        CURRENT_METADATA_SOURCE = payload['metadata_helper']
        CURRENT_PLAN = plan
        CURRENT_PREPARE = effective_plan(base, plan, original)['native_prepare_command']
        CURRENT_VERIFICATION = verification_command(plan, original)
        assets.REMOTE_COMMANDS = COMMANDS
        root, prefix = Path(plan['setup_directory']), Path(plan['private_prefix'])
        if any(Path(path).exists() or Path(path).is_symlink() for path in
                (root, prefix, plan['native_packet_directory'], plan['native_data_unit'])):
            raise ValueError('the environment/setup/packet/data paths must all be fresh')
        base.ordinary(root.parent, directory=True)
        base.ordinary(prefix.parent, directory=True)
        helper = base.helper_module(original)
        before_roots = {'old': base.old_roots(original, COMMANDS), 'cloned': assets.cloned_roots(base, original)}
        before_inputs = assets.input_records(base, original)
        before_mapping = base.mapping_check(helper, original)
        asset_record = helper.read_exact(plan['asset_record'])
        before_assets = assets.asset_roster(base, plan['asset_view'])
        helper.load('train_optimized').verify_assets(asset_record, {'XDG_DATA_HOME': plan['asset_view']})
        if len(before_assets) != 4531 or any((Path(plan['asset_view']) / name).stat().st_nlink != 1 for name in before_assets):
            raise ValueError('reuse only the exact existing ordinary physical4531-file view')
        root.mkdir()
        wheelhouse = root / 'wheels'
        wheelhouse.mkdir()
        rows = []
        for row in plan['wheels']:
            target = wheelhouse / Path(row['path']).name
            base.receive_file(sys.stdin.buffer, target, row)
            rows.append({**row, 'path': str(target)})
        if sys.stdin.buffer.read(1):
            raise ValueError('the setup stream contains bytes beyond the seven fixed wheels')
        wheels = [metadata.wheel_metadata(row) for row in rows]
        before = probe(base, plan, payload, root, OLD_PYTHON, hash_shared=True)
        observed = json.loads(payload['full_base_inventory'])
        expected_base = {metadata.normalized(row['name']): {**row, 'name': metadata.normalized(row['name'])} for row in observed['distributions']}
        if before['effective'] != expected_base or before['prefix'] != plan['base_prefix'] or before['base_prefix'] != plan['base_prefix']:
            raise ValueError('actual full base providers differ from the saved read-only inventory')
        proof = metadata.proposed_closure(list(before['effective'].values()), wheels, before['environment'])
        if proof['active_dependency_count'] != 128:
            raise ValueError('the actual complete dependency closure differs from the reviewed128 requirements')
        base.write_record(root / 'base_before.json', before)
        execute(base, [OLD_PYTHON, '-I', '-B', '-m', 'venv', '--without-pip', '--system-site-packages', '--copies', str(prefix)], root, environment=installation_environment())
        private_before = probe(base, plan, payload, root, plan['private_python'])
        check_prefix(plan, before, private_before)
        if any(Path(private_before['purelib']).iterdir()):
            raise ValueError('the fresh no-pip private site-packages must be empty before installation')
        supported = set(private_before['supported_tags'])
        if any(not supported.intersection(wheel['tags']) for wheel in wheels):
            raise ValueError('an exact CPU/reference wheel is not compatible with actual target tags')
        requirements = root / 'requirements.lock'
        lines = [metadata.normalized(row['project']) + '==' + row['version'] + ' --hash=sha256:' + row['sha256'] for row in rows]
        with requirements.open('x') as output:
            output.write('\n'.join(lines) + '\n')
            output.flush()
            os.fsync(output.fileno())
        execute(base, pip_command(plan, wheelhouse, requirements), root, environment=installation_environment(), timeout=300)
        private = probe(base, plan, payload, root, plan['private_python'])
        check_prefix(plan, before, private, installed=True)
        closure = metadata.dependency_check(list(private['effective'].values()), private['environment'])
        if closure['active_dependency_count'] != 128:
            raise ValueError('installed complete dependency closure changed')
        execute(base, [plan['private_python'], '-I', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'check'], root, environment=installation_environment())
        installed_files = [metadata.installed_wheel_files(row, private['purelib']) for row in rows]
        base.write_record(root / 'private_environment.json', private)
        base.write_record(root / 'full_dependency_preflight.json', closure)
        base.write_record(root / 'installed_locked_wheel_files.json', installed_files)
        effective = effective_plan(base, plan, original)
        environment = base.native_environment(effective)
        environment = {key: value for key, value in environment.items() if not key.startswith('PIP_')}
        environment['PIP_CONFIG_FILE'] = os.devnull
        environment['XDG_DATA_HOME'] = plan['asset_view']
        execute(base, effective['native_prepare_command'], root, environment=environment)
        os.environ.clear()
        os.environ.update(environment)
        verification = verify_preparation(base, plan, original, root, environment)
        base.write_record(root / 'selected_private_python_verification.json', verification)
        after = probe(base, plan, payload, root, OLD_PYTHON, hash_shared=True)
        after_roots = {'old': base.old_roots(original, COMMANDS), 'cloned': assets.cloned_roots(base, original)}
        if (before != after or before_roots != after_roots or before_inputs != assets.input_records(base, original)
                or before_mapping != base.mapping_check(helper, original) or before_assets != assets.asset_roster(base, plan['asset_view'])):
            raise ValueError('the preserved base files/assets/inputs/source roots changed during setup')
        if any(name in sys.modules for name in base.NUMERICAL_MODULES):
            raise ValueError('private environment setup imported a numerical module')
        base.write_record(root / 'base_after.json', after)
        base.write_record(root / 'preserved_inputs_and_roots.json', {'inputs': before_inputs, 'roots': before_roots,
            'mapping': before_mapping, 'asset_view': asset_record, 'all_unchanged': True})
        records = base.prepared_files(effective) + [('setup/' + name, root / name) for name in
            ('base_before.json', 'base_after.json', 'private_environment.json', 'full_dependency_preflight.json',
             'installed_locked_wheel_files.json', 'preserved_inputs_and_roots.json', 'selected_private_python_verification.json', 'requirements.lock')]
        records += [('setup/' + Path(row[key]['path']).name, Path(row[key]['path'])) for row in COMMANDS if isinstance(row.get('stdout'), dict) for key in ('stdout', 'stderr')]
        files = [{'relative_path': relative, **base.file_record(path)} for relative, path in records]
        if any(row['bytes'] > MAX_FILE for row in files):
            raise ValueError('a returned setup artifact exceeds its explicit size bound')
        receipt = {'schema_version': 1, 'kind': 'aces_actual_private_reference_environment_preparation', 'passed': True,
            'plan_sha256': PLAN_SHA, 'driver_sha256': digest(payload['driver_source'].encode()),
            'private_prefix': plan['private_prefix'], 'private_python': plan['private_python'], 'base_prefix': plan['base_prefix'],
            'base_package_files_unchanged': True, 'private_packages': PRIVATE_VERSIONS, 'active_dependency_count': 128,
            'commands': COMMANDS, 'prepared_files': list(files), 'numerical_imports': [], 'jobs_submitted': 0,
            'workers_permitted': False, 'case_credit': 0, 'idea_credit': 0, 'admission_credit': 0}
        final = root / 'native_private_reference_preparation_receipt.json'
        base.write_record(final, receipt)
        files.append({'relative_path': 'setup/native_private_reference_preparation_receipt.json', **base.file_record(final)})
        sys.stdout.buffer.write((json.dumps({'receipt': receipt, 'files': files}) + '\n').encode())
        for row in files:
            base.send_file(sys.stdout.buffer, row['path'], row)
        sys.stdout.buffer.flush()
    finally:
        signal.alarm(0)


def accept_response(base, path, directory, plan, original, source_hash):
    """Keep returned packet/setup files beneath exact known local names and verify this setup's receipt."""
    fixed = {relative: str(remote) for relative, remote in base.prepared_files(effective_plan(base, plan, original))}
    names = {'base_before.json', 'base_after.json', 'private_environment.json', 'full_dependency_preflight.json',
        'installed_locked_wheel_files.json', 'preserved_inputs_and_roots.json', 'selected_private_python_verification.json', 'requirements.lock', 'native_private_reference_preparation_receipt.json'}
    with base.ordinary(path).open('rb') as stream:
        line = stream.readline(base.MAX_HEADER + 1)
        if len(line) > base.MAX_HEADER or not line.endswith(b'\n'):
            raise ValueError('the returned setup header is absent or oversized')
        header = json.loads(line)
        receipt = header['receipt']
        if (set(header) != {'receipt', 'files'} or receipt.get('kind') != 'aces_actual_private_reference_environment_preparation'
                or receipt.get('passed') is not True or receipt.get('plan_sha256') != PLAN_SHA
                or receipt.get('driver_sha256') != source_hash or receipt.get('private_python') != plan['private_python']
                or receipt.get('private_packages') != PRIVATE_VERSIONS or receipt.get('base_package_files_unchanged') is not True
                or receipt.get('active_dependency_count') != 128 or receipt.get('numerical_imports') != []
                or receipt.get('workers_permitted') is not False or any(type(receipt.get(key)) is not int or receipt[key] != 0
                    for key in ('jobs_submitted', 'case_credit', 'idea_credit', 'admission_credit'))):
            raise ValueError('the returned setup receipt changed identity or zero-job scope')
        for command in receipt['commands']:
            if isinstance(command.get('stdout'), dict):
                for key in ('stdout', 'stderr'):
                    name = Path(command[key]['path']).name
                    if not name.startswith('command-') or name.rsplit('.', 1)[-1] not in ('stdout', 'stderr'):
                        raise ValueError('a returned raw command path is not a setup artifact')
                    names.add(name)
        fixed.update({'setup/' + name: str(Path(plan['setup_directory']) / name) for name in names})
        rows = header['files']
        if len(rows) != len(fixed) or {row['relative_path'] for row in rows} != set(fixed):
            raise ValueError('returned setup artifact roster is incomplete or duplicated')
        for row in rows:
            if (set(row) != {'relative_path', 'path', 'sha256', 'bytes'} or fixed[row['relative_path']] != row['path']
                    or type(row['bytes']) is not int or not 0 <= row['bytes'] <= MAX_FILE):
                raise ValueError('a returned setup artifact changed its exact path/size')
            base.receive_file(stream, directory / 'received' / row['relative_path'], row)
        if stream.read(1):
            raise ValueError('setup returned unregistered trailing bytes')
    if json.loads((directory / 'received/setup/native_private_reference_preparation_receipt.json').read_bytes()) != receipt:
        raise ValueError('the returned setup receipt does not match its raw saved file')
    return receipt


def send_inputs(stream, payload, plan, base, errors):
    """Stream only the bounded metadata header and seven fixed wheel files with incremental hashes."""
    try:
        header = (json.dumps(payload) + '\n').encode()
        if len(header) > base.MAX_HEADER:
            raise ValueError('the setup metadata header exceeds its explicit bound')
        stream.write(header)
        for row in plan['wheels']:
            base.send_file(stream, row['path'], row)
        stream.flush()
    except Exception as error:
        errors.append(type(error).__name__ + ': ' + str(error))
    finally:
        stream.close()


def local_execute(payload, plan, base, original, directory):
    """Use one already-open-master SSH session without checking the master, authenticating or retrying."""
    if socket.gethostname().split('.')[0] != 'portal05':
        raise ValueError('run private reference setup only from portal05')
    directory = Path(directory).absolute()
    base.ordinary(directory.parent, directory=True)
    directory.mkdir()
    source_hash = digest(payload['driver_source'].encode())
    command = [*base.SSH_OPTIONS, 'exec ' + shlex.join([OLD_PYTHON, '-I', '-B', '-c', BOOTSTRAP, source_hash])]
    base.write_record(directory / 'invocation.json', {'plan_sha256': PLAN_SHA, 'driver_sha256': source_hash,
        'bootstrap_sha256': digest(BOOTSTRAP.encode()), 'ssh_options': base.SSH_OPTIONS, 'master_separately_checked': False,
        'timeout_seconds': 1260, 'automatic_retry_permitted': False, 'jobs_submitted': 0})
    (directory / 'executed_driver.py').write_text(payload['driver_source'])
    errors = []
    with (directory / 'stdout.raw').open('xb') as output, (directory / 'stderr.raw').open('xb') as error:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=error)
        writer = threading.Thread(target=send_inputs, args=(process.stdin, payload, plan, base, errors), daemon=True)
        writer.start()
        try:
            process.wait(timeout=1260)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            errors.append('the fixed setup transport deadline expired; no retry permitted')
        writer.join(timeout=10)
    if writer.is_alive():
        errors.append('the setup input writer did not finish')
    base.write_record(directory / 'transport_result.json', {'returncode': process.returncode, 'errors': errors,
        'stdout': base.file_record(directory / 'stdout.raw'), 'stderr': base.file_record(directory / 'stderr.raw'), 'automatic_retry_permitted': False})
    if process.returncode or errors:
        raise ValueError('private reference setup did not complete; raw evidence retained')
    return accept_response(base, directory / 'stdout.raw', directory, plan, original, source_hash)


def main():
    """Validate all fixed inputs by default; root may separately execute the reviewed setup once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--receipt-directory', type=Path)
    arguments = parser.parse_args()
    payload, plan, base, assets, metadata, original = local_payload(arguments.plan)
    if arguments.execute:
        if arguments.receipt_directory is None:
            raise ValueError('execution requires a fresh absolute metadata receipt directory')
        result = local_execute(payload, plan, base, original, arguments.receipt_directory)
        print(json.dumps({'passed': result['passed'], 'private_python': plan['private_python'], 'jobs_submitted': 0}))
    else:
        print(json.dumps({'plan_sha256': PLAN_SHA, 'private_python': plan['private_python'], 'wheel_count': 7,
            'complete_active_requirements': 128, 'native_prepare_command': effective_plan(base, plan, original)['native_prepare_command'],
            'network_calls': 0, 'pip_invocations': 0}, indent=2))


if __name__ == '__main__':
    main()
