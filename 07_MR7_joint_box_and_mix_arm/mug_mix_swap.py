"""Swap flagged recovery seeds of a Mix list for audited spares (Claude, 2026-10-30, user request).

usage: python mug_mix_swap.py <old list dir> <spare audit.json> <out dir> <seed,seed,...>
The listed recovery seeds become expert-demo seeds; replacements come from the spare pool (valid seeds not drawn):
the spares are ordered by random.Random(20261030).shuffle(sorted spares) and the first ones whose audit row
(audit_mix_lists.py) has no flag and a stall after the first motion <= 15 frames are taken. Seeds stay the 100 E-arm
seeds; row formats unchanged. The old list dir is not modified.
"""
import json
import os
import random
import shutil
import sys

DRAW_SEED, MAX_STALL = 20261030, 15


def main(old, spare_audit, out, drop):
    drop = [int(s) for s in drop.split(",")]
    m = json.load(open(f"{old}/mix.json"))
    assert set(drop) <= set(m["recovery_seeds"])
    spares = sorted(set(m["valid"]) - set(m["recovery_seeds"]))
    rows = {r["seed"]: r for r in json.load(open(spare_audit))["rows"]}
    assert set(rows) == set(spares), "the audit must cover exactly the spare pool"
    order = list(spares)
    random.Random(DRAW_SEED).shuffle(order)
    ok = [s for s in order if not rows[s]["flags"] and rows[s]["stall_after_first_motion"] <= MAX_STALL]
    assert len(ok) >= len(drop), f"only {len(ok)} eligible spares"
    add = ok[:len(drop)]
    rec = sorted(set(m["recovery_seeds"]) - set(drop) | set(add))
    exp = [s for s in m["e_arm_seeds"] if s not in rec]
    os.makedirs(out, exist_ok=False)
    E = {json.loads(x)["env_seed"]: x for x in open("/workspace/fact_local/sft_lists_final_g7_1029/E_expert.jsonl")}
    with open(f"{out}/R_recovery.jsonl", "w") as fr, open(f"{out}/R_prefix.jsonl", "w") as fp:
        for s in rec:
            x = m["pool"][str(s)]
            fr.write(json.dumps({"npz": x["npz"], "instruction": x["instruction"], "env_seed": s,
                                 "action_masked_until": x["action_masked_until"]}) + "\n")
            fp.write(json.dumps({"npz": x["npz"], "instruction": x["instruction"], "env_seed": s,
                                 "end": x["end"], "failure_active_from": x["failure_active_from"]}) + "\n")
    with open(f"{out}/E_expert.jsonl", "w") as fe:
        fe.write("".join(E[s] for s in exp))
    shutil.copy(spare_audit, f"{out}/spare_audit.json")
    json.dump({**m, "recovery_seeds": rec, "expert_seeds": exp,
               "swap": {"from": old, "dropped": drop, "added": add, "spare_order": order, "eligible": ok,
                        "rule": f"random.Random({DRAW_SEED}).shuffle(sorted spares); first with no audit flag and "
                                f"stall <= {MAX_STALL}"}}, open(f"{out}/mix.json", "w"), indent=1)
    print("dropped", drop, "added", add, "| spare order", order, "| eligible", ok)


if __name__ == "__main__":
    main(*sys.argv[1:])
