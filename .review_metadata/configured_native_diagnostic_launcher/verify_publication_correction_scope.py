import ast,hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent
if os.environ.get('SLURM_JOB_ID')!='6821437' or not os.environ.get('SLURM_STEP_ID','').isdigit():raise ValueError('owned numeric CPU step required')
old=ast.parse((D/'independent_preserved_local_only_publication_rejection/launch_diagnostic_once.py').read_bytes());new=ast.parse((D/'launch_diagnostic_once.py').read_bytes())
oldf={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)};newf={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
if set(newf)-set(oldf)!={'require_published_commit'} or set(oldf)-set(newf):raise ValueError('unexpected function scope')
checked=[]
for name in oldf:
 n=newf[name]
 if name=='check_publication':
  n.body=[v for v in n.body if not (isinstance(v,ast.Expr) and isinstance(v.value,ast.Call) and isinstance(v.value.func,ast.Name) and v.value.func.id=='require_published_commit')]
 if ast.dump(oldf[name],include_attributes=False)!=ast.dump(n,include_attributes=False):raise ValueError('existing function changed: '+name)
 checked.append(name)
f=D/'author_source_freeze_3.json';frozen=json.loads(f.read_bytes())
for name,entry in frozen['files'].items():
 if hashlib.sha256((D/name).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('frozen byte changed')
value={'kind':'configured_native_launcher_narrow_publication_scope','passed':True,'all_existing_functions_preserved_except_one_publication_predicate_call':True,'functions_verified':checked,'source_freeze':{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()},'slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_step_id':os.environ['SLURM_STEP_ID'],'actual_scheduler_actions':0}
p=D/('publication_scope_6821437_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('x') as s:json.dump(value,s,indent=2,sort_keys=True);s.write('\n');s.flush();os.fsync(s.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'step':os.environ['SLURM_STEP_ID'],'functions_verified':len(checked)},sort_keys=True))
