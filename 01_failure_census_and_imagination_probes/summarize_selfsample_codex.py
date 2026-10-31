"""All-grid descriptive summary; missing cells are not silently replaced or dropped."""
import hashlib,json
from pathlib import Path

def summarize(grid, directory):
    cells=[]
    for spec in grid['cells']:
        path=directory/(spec['name']+'.json')
        row={k:spec[k] for k in ('ordinal','env_seed','point','stream_repeat')}
        if not path.exists(): row.update(status='not_run',success=None)
        else:
            raw=path.read_bytes(); d=json.loads(raw)
            expected={'env_seed':spec['env_seed'],'branch_step':spec['branch_step'],'root_seed':spec['root_seed'],
                      'stream_repeat':spec['stream_repeat'],'source_sha256':spec['source_sha256']}
            errors=[k for k,v in expected.items() if d.get(k)!=v]
            actions=d.get('executed_actions')
            if not isinstance(actions,list) or d.get('executed_actions_sha256')!=hashlib.sha256(json.dumps(actions).encode()).hexdigest():errors.append('actions_hash')
            if type(d.get('success')) is not bool:errors.append('success_type')
            row.update(status='invalid' if errors else 'complete',success=d.get('success') if not errors else None,
                       errors=errors,sha256=hashlib.sha256(raw).hexdigest())
        cells.append(row)
    groups=[]
    for seed in sorted({x['env_seed'] for x in cells}):
        for point in ('pre','post'):
            subset=[x for x in cells if x['env_seed']==seed and x['point']==point]
            if len(subset)!=8 or {x['stream_repeat'] for x in subset}!=set(range(1,9)):
                raise ValueError('grid not exactly eight distinct streams')
            success=sum(x['success'] is True for x in subset); missing=sum(x['status']!='complete' for x in subset)
            groups.append({'env_seed':seed,'point':point,'planned':8,'success_count':success,'missing_or_invalid':missing,
              'success_fraction_all8':success/8 if not missing else None,
              'any_success_at8':bool(success) if success or not missing else None,
              'success_fraction_bounds':[success/8,(success+missing)/8]})
    return {'groups':groups,'cells':cells,'scope':'descriptive development scenes; stream repetitions are not independent starts'}
