"""Compile the committed archive as text without importing any prepared module."""
from pathlib import Path
import json,sys
packet=Path(__file__).resolve().parent
manifest=json.loads((packet/'source_snapshot.json').read_bytes())
source=Path(manifest['source_root'])
files=[source/name for name in manifest['files']if name.endswith('.py')]
for path in files:compile(path.read_bytes(),str(path),'exec')
assert not any(name.split('.')[0]in {'jax','jaxlib','numpy','torch','jaxtari','flax','optax'}for name in sys.modules)
print(json.dumps({'source_commit':manifest['source_commit'],'python':sys.version,'python_executable':sys.executable,'python_files_compiled':len(files),'prepared_modules_imported':False,'numerical_imports':False}))
