"""Classify hanging_mug failures by task phase from 48-step boundary mug poses + gripper commands."""
import json, glob, collections, sys

def grip_events(A):
    ev = []
    for g, n in ((6, "L"), (13, "R")):
        st = A[0][g] > 0.5
        for k in range(1, len(A)):
            v = A[k][g]; ns = True if v > 0.5 else (False if v < 0.3 else st)
            if ns != st: ev.append((k, n + ("o" if ns else "c"))); st = ns
    return sorted(ev)

def phase(ro):
    A = ro["executed_actions"]; S = ro["states"]
    zs = sorted((int(k), v["actors"]["0:039_mug"]["p"][2]) for k, v in S.items() if k != "final" and "actors" in v)
    if "final" in S and "actors" in S["final"]: zs.append((S["final"]["action_step"], S["final"]["actors"]["0:039_mug"]["p"][2]))
    z0 = zs[0][1]; ev = grip_events(A)
    first = lambda e: next((k for k, x in ev if x == e), None)
    rc, ro_ = first("Rc"), first("Ro")
    if rc is None: return "left_phase", ev
    after = [z for t, z in zs if t >= rc + 48]
    if not after or max(after) < z0 + 0.05: return "right_never_lifted", ev
    # right lifted: when did it fall
    peak_t = max((z, t) for t, z in zs if t >= rc)[1]
    fall_t = next((t for t, z in zs if t > peak_t and z < z0 + 0.05), None)
    if fall_t is None: return "lifted_no_fall", ev
    if ro_ is not None and fall_t >= ro_: return "fell_after_right_release", ev
    return "fell_while_right_holding", ev

def summarize(name, files, filt=lambda j: not j.get("success")):
    c = collections.Counter(); out = {}
    for f in files:
        j = json.load(open(f))
        if j.get("success") not in (True, False) or not filt(j): continue
        p, ev = phase(j); c[p] += 1; out[j["env_seed"]] = p
    print(name, sum(c.values()), dict(c.most_common()))
    return out
