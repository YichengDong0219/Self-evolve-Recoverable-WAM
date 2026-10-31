"""Build a categorised, clearly named results tree from fact_g0_claude/runs (Claude, 2026-10-27).

Hard links only (no copies, no moves): runs/ stays the source of truth that scripts and JSONs reference.
Usage: build_results_tree.py <runs_dir> <out_dir>
"""
import json
import os
import shutil
import sys
from pathlib import Path

RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
if OUT.exists():
    raise SystemExit(f"refusing to overwrite {OUT}")
index = []


def link(src, dst):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def outcome(d):
    return f"{'SUCCESS' if d['success'] else 'FAIL'}_{d['steps']}steps"


def episode(prefix_dir, d, stem_src, stem_dst, extras):
    """extras: list of (source suffix relative to runs, destination suffix)."""
    for src_rel, suffix in extras:
        link(RUNS / src_rel.format(n=stem_src), prefix_dir / f"{stem_dst}{suffix}")


# ---- 00 G0 branch validation -------------------------------------------------------------------
g0 = OUT / "00_G0_branch_validation"
names = {"mug_R1": "hanging_mug_seed100000_rollout_A_unseeded-instruction",
         "mug_R1b": "hanging_mug_seed100000_rollout_B_unseeded-instruction",
         "mug_R2": "hanging_mug_seed100000_replay_A_of-R1", "mug_R2b": "hanging_mug_seed100000_replay_B_of-R1",
         "mug_B96": "hanging_mug_seed100000_branch-at-step096_of-R1", "mug_B192": "hanging_mug_seed100000_branch-at-step192_of-R1",
         "mug_B288": "hanging_mug_seed100000_branch-at-step288_of-R1",
         "mug_S1": "hanging_mug_seed100000_rollout_A", "mug_S1b": "hanging_mug_seed100000_rollout_B",
         "mug_S2": "hanging_mug_seed100000_replay_of-rollout-A",
         "mug_SB96": "hanging_mug_seed100000_branch-at-step096", "mug_SB192": "hanging_mug_seed100000_branch-at-step192",
         "mug_SB288": "hanging_mug_seed100000_branch-at-step288",
         "cup_S1": "place_empty_cup_seed100000_rollout_A", "cup_S1b": "place_empty_cup_seed100000_rollout_B",
         "cup_S2": "place_empty_cup_seed100000_replay_of-rollout-A",
         "cup_SB48": "place_empty_cup_seed100000_branch-at-step048", "cup_SB96": "place_empty_cup_seed100000_branch-at-step096",
         "cup_SB144": "place_empty_cup_seed100000_branch-at-step144"}
for src, dst in names.items():
    sub = "1_before-instruction-fix_FAILED-reproduction" if src.startswith("mug_R") or src.startswith("mug_B") \
        else "2_after-instruction-fix_PASSED"
    for ext in (".json", ".log"):
        link(RUNS / f"{src}{ext}", g0 / sub / f"{dst}{ext}")
for v, dst in (("A", "A_prefix-cache-ON_action-only"), ("B_cache_off", "B_prefix-cache-OFF_action-only"),
               ("C_cache_off_images", "C_prefix-cache-OFF_return-images")):
    link(RUNS / f"model_check_{v}.json", g0 / "3_model_numeric_checks" / f"model_check_{dst}.json")
link(RUNS / "mug_request.npz", g0 / "3_model_numeric_checks" / "observation_used_for_model_checks.npz")

# ---- 01 P1 census (deployment config) ---------------------------------------------------------
p1 = OUT / "01_P1_failure-census_demo_clean_deploy-config"
for f in sorted(RUNS.glob("P1_*.json")):
    d = json.loads(f.read_text())
    n = f.stem
    task = d["task"]
    task_dir = "adjust_bottle_sanity-check" if task == "adjust_bottle" else task
    cat = "success" if d["success"] else "fail"
    stem = f"P1_{task}_seed{d['env_seed']}_{outcome(d)}"
    dst = p1 / task_dir / cat
    episode(dst, d, n, stem, [("{n}.json", ".trajectory.json"), ("videos/{n}.mp4", ".video.mp4"),
                              ("labels/{n}.label.json", ".per-step-labels.json")])
    index.append(("P1", task, d["env_seed"], d["success"], d["steps"]))
link(RUNS / "p1_summary.jsonl", p1 / "P1_per-episode_summary.jsonl")

# ---- 02 CMP: FACT on the OpenWAM G0 scenes ------------------------------------------------------
cmp_ = OUT / "02_CMP_FACT-on-OpenWAM-scenes_demo_randomized_deploy-config"
openwam = {}
jobs = RUNS / "cmp_jobs.txt"
if jobs.exists():
    for line in jobs.read_text().splitlines():
        j = json.loads(line)
        openwam[(j["task"], j["seed"])] = j["openwam_success"]
for f in sorted(RUNS.glob("CMP_*.json")):
    d = json.loads(f.read_text())
    ow = openwam.get((d["task"], d["env_seed"]))
    ow_s = "unknown" if ow is None else ("SUCCESS" if ow else "FAIL")
    stem = f"CMP_{d['task']}_seed{d['env_seed']}_FACT-{outcome(d)}_OpenWAM-{ow_s}"
    episode(cmp_ / d["task"], d, f.stem, stem, [("{n}.json", ".trajectory.json"), ("videos/{n}.mp4", ".video.mp4")])
    index.append(("CMP", d["task"], d["env_seed"], d["success"], d["steps"]))

# ---- 03 IMG: imagination recorded live ---------------------------------------------------------
img = OUT / "03_IMG_real-vs-imagined_demo_clean_cache-off-return-images"
for f in sorted(RUNS.glob("IMG_*.json")):
    d = json.loads(f.read_text())
    cat = "success" if d["success"] else "fail"
    stem = f"IMG_{d['task']}_seed{d['env_seed']}_{outcome(d)}"
    episode(img / d["task"] / cat, d, f.stem, stem,
            [("sbs/{n}.sbs.mp4", ".real-vs-imagined.mp4"), ("{n}.json", ".trajectory.json"),
             ("{n}.imagined.npz", ".imagined-keyframes.npz")])
    index.append(("IMG", d["task"], d["env_seed"], d["success"], d["steps"]))

# ---- 04 reconstructed imagination for a P1 episode ----------------------------------------------
rec = OUT / "04_reconstructed-imagination_for-P1-episodes"
for f in sorted((RUNS / "imagine").glob("*.real_vs_imagined.mp4")):
    n = f.name.replace(".real_vs_imagined.mp4", "")
    d = json.loads((RUNS / f"{n}.json").read_text())
    stem = f"P1_{d['task']}_seed{d['env_seed']}_{outcome(d)}_reconstructed"
    for s_src, s_dst in ((".real_vs_imagined.mp4", ".real-top_vs_imagined-bottom.mp4"),
                         (".real_vs_imagined.json", ".action-diff-per-decision.json"),
                         (".real_vs_imagined.imagined.npz", ".imagined-keyframes.npz")):
        link(RUNS / "imagine" / f"{n}{s_src}", rec / f"{stem}{s_dst}")

# ---- 05 scenes whose outcome flips between P1 (cache on) and IMG (cache off) ---------------------
flip = OUT / "05_outcome-flips_P1-vs-IMG_same-scene"
p1map = {(t, s): (ok, st) for b, t, s, ok, st in index if b == "P1"}
for b, t, s, ok, st in index:
    if b == "IMG" and (t, s) in p1map and p1map[(t, s)][0] != ok:
        p_ok, p_st = p1map[(t, s)]
        d_dir = flip / f"{t}_seed{s}_P1-{'SUCCESS' if p_ok else 'FAIL'}_IMG-{'SUCCESS' if ok else 'FAIL'}"
        p_cat = "success" if p_ok else "fail"
        i_cat = "success" if ok else "fail"
        p_stem = f"P1_{t}_seed{s}_{'SUCCESS' if p_ok else 'FAIL'}_{p_st}steps"
        i_stem = f"IMG_{t}_seed{s}_{'SUCCESS' if ok else 'FAIL'}_{st}steps"
        link(p1 / t / p_cat / f"{p_stem}.video.mp4", d_dir / f"{p_stem}.video.mp4")
        link(img / t / i_cat / f"{i_stem}.real-vs-imagined.mp4", d_dir / f"{i_stem}.real-vs-imagined.mp4")

# ---- 99 run records -----------------------------------------------------------------------------
meta = OUT / "99_run-records"
for name in ("ledger.jsonl", "code_versions.txt", "P1_progress.txt", "CMP_progress.txt", "IMG_progress.txt",
             "cmp_jobs.txt"):
    link(RUNS / name, meta / name)

# ---- README ------------------------------------------------------------------------------------
def tally(batch, task):
    rows = [x for x in index if x[0] == batch and x[1] == task]
    return f"{sum(r[3] for r in rows)}/{len(rows)}"


lines = ["# FACT 实验结果（按批次分类，2026-10-27）", "",
         "文件名格式：`<批次>_<任务>_seed<场景seed>_<SUCCESS|FAIL>_<步数>steps.<内容>.<扩展名>`。",
         "全部为硬链接，原始数据仍在 `runs/`（脚本与轨迹记录引用原路径）。", "",
         "| 目录 | 内容 | 配置 |", "|---|---|---|",
         "| 00_G0_branch_validation | 分支验收：同 seed 复现、无模型重放、从中途分支；模型侧数值对照 | demo_clean |",
         "| 01_P1_failure-census | 失败普查，每任务 20 个 episode（adjust_bottle 4 个健全性） | demo_clean，部署配置（prefix cache 开、只返回动作） |",
         "| 02_CMP_FACT-on-OpenWAM-scenes | FACT 在 OpenWAM G0 的 10 个场景上（同 seed、同指令原文），文件名含两者结局 | demo_randomized，部署配置 |",
         "| 03_IMG_real-vs-imagined | 采样时同步记录想象；视频左=实际三视角，右=同一次决策的想象 | demo_clean，prefix cache 关 + 返回视频 |",
         "| 04_reconstructed-imagination | 为 P1 episode 事后重建想象（上=实际，下=想象），附每次决策动作差 | 同 P1 |",
         "| 05_outcome-flips | P1 与 IMG 同一场景结局不同的 5 个场景，两段视频放在一起 | — |",
         "| 99_run-records | GPU 账本、代码版本、进度日志 | — |", "", "## 成功率", "",
         "| 批次 | 任务 | 成功/总数 |", "|---|---|---|"]
for b in ("P1", "CMP", "IMG"):
    for t in sorted({x[1] for x in index if x[0] == b}):
        lines.append(f"| {b} | {t} | {tally(b, t)} |")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print(f"built {OUT}: {sum(1 for _ in OUT.rglob('*') if _.is_file())} files")
