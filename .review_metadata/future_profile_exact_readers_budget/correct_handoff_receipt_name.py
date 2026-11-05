"""Preserve an author-only receipt-name typo, joining the existing actual result without rerunning it."""
import hashlib
import json
import os
from pathlib import Path
M=Path(__file__).resolve().parent
original=M/'author_handoff.json';raw=original.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='42d4d4af4ea5aecd3bacfce9450f324d99afdcabb949bd5e082762c4adca86fe'
value=json.loads(raw);actual=M/'adapter_checks_6821437_32.json';receipt=json.loads(actual.read_bytes())
assert receipt['passed'] is True and receipt['tests']==4 and receipt['errors']==0 and receipt['failures']==0
for result in value['focused_results']:
    if result.get('receipt')=='adapter_checks_6821437_34':result['receipt']='adapter_checks_6821437_32'
value['preserved_author_metadata_correction']={'original_handoff':{'path':str(original),'sha256':hashlib.sha256(raw).hexdigest()},
    'reason':'One manually typed result name said34; the actual four-method receipt is32. Original preserved, no tests repeated.',
    'actual_receipt':{'path':str(actual),'sha256':hashlib.sha256(actual.read_bytes()).hexdigest()}}
target=M/'author_handoff_corrected.json'
with target.open('x') as handle:
    json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(str(target));print(hashlib.sha256(target.read_bytes()).hexdigest())
