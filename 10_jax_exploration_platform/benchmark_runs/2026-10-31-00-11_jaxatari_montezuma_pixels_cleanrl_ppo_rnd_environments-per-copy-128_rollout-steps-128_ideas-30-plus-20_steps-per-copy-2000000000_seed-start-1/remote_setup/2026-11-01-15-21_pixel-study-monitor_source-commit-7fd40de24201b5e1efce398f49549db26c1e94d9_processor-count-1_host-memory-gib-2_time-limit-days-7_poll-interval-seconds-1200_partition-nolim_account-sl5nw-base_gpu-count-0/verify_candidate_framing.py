"""Check the frozen monitor's exact print/report framing without running its tick."""
import ast
import contextlib
import io
import json
from pathlib import Path

P=Path(__file__).resolve().parent
manifest=json.loads((P/'source_snapshot.json').read_text())
module=ast.parse((Path(manifest['source_root'])/'10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py').read_text())
tick=next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='tick')
text_assignment=next(node for node in tick.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='text' for target in node.targets))
report_assignment=next(node for node in tick.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='report' for target in node.targets))
printing=next(node for node in tick.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='print')
namespace={'rows':['TRAINING','WAITING','HELD','FINISHED','PLANNED'],'now':'synthetic time only'}
stdout=io.StringIO()
with contextlib.redirect_stdout(stdout):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[text_assignment,printing,report_assignment],type_ignores=[])),'<exact-frozen-monitor-statements>','exec'),namespace)
raw=stdout.getvalue().encode();report=namespace['report'].encode()
assert raw.endswith(b'\n\n')
assert report==b'# Pixel Montezuma study status\n\nsynthetic time only\n\n```text\n'+raw[:-1]+b'```\n'
assert raw not in report
candidate=(P/'candidate_full_tick.py').read_text();ast.parse(candidate)
assert "table = raw_stdout[:-1]" in candidate
assert "report = own_writes['latest_report.md'][0]" in candidate
assert "status_raw = own_writes['latest_status.json'][0]" in candidate
assert 'original_write_text(path, data, *args, **kwargs)' in candidate
print(json.dumps({'passed':True,'actual_frozen_tick_framing_statements':True,'extra_print_newline_removed_exactly_once':True,'candidate_own_write_attribution':True,'synthetic_only':True,'full_tick_executed':False}))
