"""Causal timestamp audit for Claude's frozen developmental rules (no threshold tuning).
Onset is retrospective; detection is first time enough observations are available.
"""
import math


def detect(rows):
    if not rows: raise ValueError('empty labels')
    if [r['step'] for r in rows] != list(range(rows[0]['step'],rows[0]['step']+len(rows))):
        raise ValueError('dense consecutive labels required')
    z0=rows[0]['p'][2]; last_high=None; still=0
    for i,r in enumerate(rows):
        for key in ('p','tcp_l','tcp_r'):
            if len(r[key])!=3 or any(not math.isfinite(v) for v in r[key]): raise ValueError('invalid geometry')
        if r['p'][2] > z0+.10: last_high=i
        hits=[]
        if i>=10:
            prev=rows[i-10]
            sep=lambda x:min(math.dist(x['p'],x['tcp_l']),math.dist(x['p'],x['tcp_r']))
            if prev['p'][2]-r['p'][2]>=.03 and sep(r)-sep(prev)>=.03:
                hits.append(('a_decoupled_fall',i-10))
        if last_high is not None and r['p'][2]<z0+.02:
            hits.append(('b_back_to_table',last_high))
        if i:
            still=still+1 if math.dist(r['p'],rows[i-1]['p'])<1e-4 else 0
            if still>=96 and last_high is None and r['step']>=150:
                hits.append(('c_stall_no_lift',i-96))
        if hits:
            rule,j=hits[0] # freeze same-time priority a,b,c, first detection wins
            onset=rows[j]['step']; detected=r['step']
            return {'rule':rule,'t_onset_retrospective':onset,'t_detect':detected,
                    'pre':((onset-1)//48)*48,'post':((detected//48)+1)*48}
    return None
