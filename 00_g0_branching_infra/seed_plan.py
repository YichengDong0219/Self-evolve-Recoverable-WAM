"""Stable explicit seeds. Does not mutate any RNG or select a candidate."""
import hashlib
import json


def seed_for(root_seed, start_id, stream, decision, repeat=0, candidate=None):
    if type(root_seed) is not int or root_seed < 0 or not isinstance(start_id, str) or not start_id:
        raise ValueError('invalid root seed/start id')
    if stream not in ('candidate', 'continuation', 'selection'):
        raise ValueError('unknown stream')
    if any(type(x) is not int or x < 0 for x in (decision, repeat)):
        raise ValueError('invalid decision/repeat')
    if stream in ('candidate', 'selection') and repeat != 0:
        raise ValueError('fixed candidate pool/selection requires repeat=0')
    if stream == 'candidate':
        if type(candidate) is not int or candidate < 0:
            raise ValueError('candidate stream requires candidate id')
    elif candidate is not None:
        raise ValueError('continuation/selection must not depend on candidate id')
    key = ['fact-probe-seeds/1', root_seed, start_id, stream, decision, repeat, candidate]
    return int.from_bytes(hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()[:8], 'big') % (2**63)
