"""Reconcile distinct clean sequential control sessions without changing raw ledgers."""
from pathlib import Path
import json,zipfile


def reconciled_prior_hours(input_root,protocol_sha256,minimum_hours):
    sessions={}
    def add(s):
        if s.get('protocol_sha256')!=protocol_sha256 or not s.get('closed_cleanly'):return
        key=s['session_id']
        row=(float(s['accounting_start_unix']),float(s['finished_unix']),float(s['prior_accounted_seconds']))
        if key in sessions and sessions[key]!=row:raise ValueError('Conflicting copies of a session ledger')
        sessions[key]=row
    for p in Path(input_root).rglob('control_session_state.json'):add(json.loads(p.read_text()))
    for p in Path(input_root).rglob('*.zip'):
        if not p.name.lower().startswith('balora_deterministic_lora_control_results'):continue
        with zipfile.ZipFile(p) as z:
            names=[n for n in z.namelist() if n.endswith('/control_session_state.json') or n=='control_session_state.json']
            if len(names)!=1:raise ValueError('Control ZIP must have one session ledger')
            add(json.loads(z.read(names[0])))
    cumulative=0.;finish=0.
    for start,end,prior in sorted(sessions.values()):
        if start<finish or end<start:raise ValueError('Overlapping control sessions require accounting review')
        cumulative=max(cumulative,prior)+end-start;finish=end
    result=max(float(minimum_hours),cumulative/3600.)
    print('Reconciled prior hours:',result,'| distinct clean supplied sessions:',len(sessions),flush=True)
    return result
