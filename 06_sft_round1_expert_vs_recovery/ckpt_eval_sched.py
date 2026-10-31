"""Evaluate every saved checkpoint as soon as it exists, on whatever 4090 is idle (Claude, 2026-10-29, user request).

Runs on the local box (reaches both servers). Single instance (flock). Every 3 min:
  1 list complete checkpoint copies on Amax (~/fact_sft_amax/ckpts/<tag>_step<N>, written by snap_ckpts.sh)
  2 for each one not yet evaluated, pick a machine with idle 4090s (no compute process, < 1 GB used):
      old 4090 cluster GPUs 0-3 first (never needed by the training pipeline);
      Amax GPU 8 only after the pipeline logged "eval mugE_nv finished"; Amax GPUs 4-8 after "all done"
    (the Amax pipeline needs GPUs 4-7 for training / R eval and GPU 8 for the E-final eval)
  3 a checkpoint is evaluated entirely on ONE machine (its seeds split over that machine's idle GPUs), with the
    same protocol as base / E / R: eval_task.sh (old) or eval_task_amax.sh (Amax) = FACT server with prefix cache,
    N=1, --skip_future_state_value, 48-step replanning, task hanging_mug demo_clean, one rollout per seed
  4 seeds: SEEDS100 = the 100 seeds whose expert pre-check passed on both machines in the base eval (the pre-check
    is flaky and model-independent); a seed that still comes out INVALID is re-run (up to 2 extra passes)
  5 results: <machine eval root>/<tag>_step<N>/ (json per seed); copied to fact_local/eval_ckpts/; the checkpoint
    copy on the old cluster is deleted after its eval
State: fact_local/eval_ckpts/state.json, log: fact_local/eval_ckpts/sched.log. Never uses the RTX PRO 6000s.
"""
import fcntl
import json
import os
import subprocess
import time

L = "/workspace/fact_local"
OUT = f"{L}/eval_ckpts"
A, C = "user@AMAX_HOST", "user@CLUSTER_HOST"
SEEDS100 = f"{OUT}/seeds100.txt"
REF = 10387499812  # size of diffusion_pytorch_model.bin
M = {
    "old": dict(host=C, gpus=[0, 1, 2, 3], ckroot="/workspace_cluster/fact_sft/ckpts_eval",
                evroot="/workspace_cluster/fact_sft/eval_mug", script="/workspace_cluster/fact_g0_claude/code/eval_task.sh",
                seeddir="/workspace_cluster/fact_sft/eval_mug"),
    "amax": dict(host=A, gpus=[4, 5, 6, 7, 8], ckroot="/workspace/fact_sft_amax/ckpts",
                 evroot="/workspace/fact_local/eval_mug", script="/workspace/fact_local/code/eval_task_amax.sh",
                 seeddir="/workspace/fact_local/eval_mug"),
}


def sh(host, cmd, timeout=600):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", host, cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def launch(host, cmd):
    """Start a detached remote job and return at once. Plain ssh waited for the backgrounded job's session and hit
    the 600 s timeout (10-30): the job had started but the state update was lost, so the same checkpoint was
    launched again on the next idle GPU. ssh -n -f returns as soon as the remote command is forked."""
    # no capture: the backgrounded ssh keeps captured pipes open and subprocess would wait for the remote job anyway
    subprocess.run(["ssh", "-n", "-f", "-o", "ConnectTimeout=20", host, cmd], stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


def log(msg):
    with open(f"{OUT}/sched.log", "a") as f:
        f.write(time.strftime("%F %T ") + msg + "\n")


def idle_gpus(mk, allowed):
    q = sh(M[mk]["host"], "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; echo ==; "
                          "nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader; echo ==; "
                          "nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader")
    mem_s, apps_s, bus_s = (q.split("==") + ["", "", ""])[:3]
    mem = {int(a): int(b) for a, b in (l.split(",") for l in mem_s.split("\n") if l.strip())}
    bus = {b.strip(): int(i) for i, b in (l.split(",") for l in bus_s.split("\n") if l.strip())}
    busy = {bus[b.strip()] for b in apps_s.split("\n") if b.strip() and b.strip() in bus}
    return [g for g in allowed if g in mem and mem[g] < 1024 and g not in busy]


def amax_allowed():
    # 10-30 user: no step-3194 evals on Amax; its 4090s join the pool as soon as they are idle (the idle check
    # keeps training and other users' cards out)
    # 10-30 14:08: the MR7 smoke moved to the cluster (user), Amax 6/7 back in the pool
    return [4, 5, 6, 7, 8]


def main():
    os.makedirs(OUT, exist_ok=True)
    lock = open(f"{OUT}/.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    st_path = f"{OUT}/state.json"
    st = json.load(open(st_path)) if os.path.exists(st_path) else {}
    seeds = open(SEEDS100).read().split()
    for mk in M:  # seed list on both machines
        sh(M[mk]["host"], f"mkdir -p {M[mk]['seeddir']} && printf '%s\\n' {' '.join(seeds)} > {M[mk]['seeddir']}/seeds100.txt")
    log(f"start; {len(seeds)} seeds")
    while True:
        try:
            step(st, st_path, seeds)
        except Exception as e:  # ssh hiccup etc.: log and retry next round
            log(f"round error: {type(e).__name__}: {str(e)[:200]}")
        time.sleep(180)


GPU_ERR = "ErrorDeviceLost|CUDA error: out of memory|No CUDA GPUs are available|CUDA is unavailable|GPU requires reset"


def guard():
    """10-30 safety: on any GPU-error sign on a machine (a card needing reset, or an error line in a running eval's
    logs), stop our evals there and pause that machine (file <OUT>/PAUSED_<machine>; delete it to resume)."""
    for mk, m in M.items():
        if os.path.exists(f"{OUT}/PAUSED_{mk}"):
            continue
        hit = sh(m["host"], f"(nvidia-smi -q | grep -c 'requires reset'; "
                            f"find {m['evroot']} -name '*.log' -mmin -10 2>/dev/null | xargs -r grep -lE '{GPU_ERR}' 2>/dev/null | head -3) | tr '\n' ' '")
        parts = hit.split()
        if parts and (parts[0] != "0" or len(parts) > 1):
            sh(m["host"], "pkill -u $USER -f '[e]val_task'; pkill -u $USER -f '[g]0_driver.py rollout'; "
                          "pkill -u $USER -f '[i]nference_server_probe'")
            open(f"{OUT}/PAUSED_{mk}", "w").write(time.strftime("%F %T ") + hit + "\n")
            log(f"GUARD: GPU error sign on {mk} ({hit.strip()[:200]}); our evals there stopped, machine paused")


def step(st, st_path, seeds):
    guard()
    paused = {mk for mk in M if os.path.exists(f"{OUT}/PAUSED_{mk}")}
    # 10-30 fix: a GPU launched on in this round still looks idle (its server has not allocated memory yet); 11:37 the
    # INVALID re-run of R1000 and the R2000 helper both took cluster GPU 1 and the second server died on the port
    claimed = set()

    def free(mk, allowed):
        return [g for g in idle_gpus(mk, allowed) if (mk, g) not in claimed]
    if True:
        ck = sorted(sh(A, "cd ~/fact_sft_amax/ckpts 2>/dev/null && ls -d mug*_step* 2>/dev/null | grep -v tmp").split())
        for name in ck:
            s = st.setdefault(name, {"status": "new"})
            if s["status"] == "new" and not s.get("on_old"):
                # 10-30: copy straight Amax -> cluster (Amax key authorized on the cluster by the user), ahead of any
                # free GPU so the eval can start the moment a card frees; at most 3 staged copies on the cluster
                staged = [k for k, v in st.items() if v.get("on_old") and v["status"] != "done"]
                if len(staged) < 3:
                    tp = f"{M['old']['ckroot']}/{name}"
                    sh(A, f"ssh -o BatchMode=yes user@CLUSTER_HOST 'mkdir -p {tp}' && "
                          f"rsync -a ~/fact_sft_amax/ckpts/{name}/ user@CLUSTER_HOST:{tp}/", timeout=3600)
                    size = sh(C, f"stat -c %s {tp}/diffusion_pytorch_model.bin")
                    if size == str(REF):
                        s["on_old"] = True
                        log(f"{name}: copied to the cluster")
                    else:
                        log(f"{name}: copy to the cluster has wrong size {size}; retry later")
            if s["status"] == "new":
                for mk, allowed in (("old", M["old"]["gpus"]), ("amax", amax_allowed())):
                    gs = free(mk, allowed) if allowed and mk not in paused else []
                    if not gs:
                        continue
                    m = M[mk]
                    tp = f"{m['ckroot']}/{name}"
                    if mk == "old" and not s.get("on_old"):
                        continue  # not copied yet
                    # split the seeds over this machine's idle GPUs; one FACT server + simulator per GPU
                    shards = {g: seeds[i::len(gs)] for i, g in enumerate(gs)}
                    for g, ss in shards.items():
                        sf = f"{m['seeddir']}/seeds100_{name}_gpu{g}.txt"
                        launch(m["host"], f"printf '%s\\n' {' '.join(ss)} > {sf} && cd {m['evroot']} && setsid nohup bash {m['script']} "
                                      f"{g} {name} {tp} {sf} hanging_mug {m['evroot']} > {m['evroot']}/eval_{name}_gpu{g}.log 2>&1 < /dev/null &")
                    claimed.update((mk, g) for g in gs)
                    s.update(status="running", machine=mk, gpus=gs, started=time.strftime("%F %T"), passes=1)
                    log(f"{name}: eval started on {mk} GPUs {gs}")
                    break
            elif s["status"] == "running" and s["machine"] not in paused:
                m = M[s["machine"]]
                running = sh(m["host"], f"pgrep -u $USER -f '[e]val_task.*{name} ' | wc -l")
                if running != "0":
                    # 10-30: a GPU on the same machine that frees up while this eval is still running, with no new
                    # checkpoint waiting for it, joins this eval on the remaining seeds in reverse order (the
                    # workers meet in the middle; eval_task skips seeds that already have a result)
                    waiting = [k for k, v in st.items() if v["status"] == "new" and (v.get("on_old") or s["machine"] == "amax")]
                    allowed = M["old"]["gpus"] if s["machine"] == "old" else amax_allowed()
                    gs = [g for g in free(s["machine"], allowed) if g not in s["gpus"]] if allowed and not waiting else []
                    if gs:
                        res = sh(m["host"], f"cd {m['evroot']}/{name} && ls hanging_mug_seed*.json 2>/dev/null")
                        done = {x[len('hanging_mug_seed'):-5] for x in res.split()}
                        rest = [x for x in reversed(seeds) if x not in done]
                        if len(rest) > 10:
                            g = gs[0]
                            sf = f"{m['seeddir']}/seeds100_{name}_help_gpu{g}.txt"
                            launch(m["host"], f"printf '%s\\n' {' '.join(rest)} > {sf} && cd {m['evroot']} && setsid nohup bash {m['script']} "
                                              f"{g} {name} {s.get('tp', m['ckroot'] + '/' + name)} {sf} hanging_mug {m['evroot']} "
                                              f"> {m['evroot']}/eval_{name}_help_gpu{g}.log 2>&1 < /dev/null &")
                            claimed.add((s["machine"], g))
                            s["gpus"] = s["gpus"] + [g]
                            log(f"{name}: GPU {g} joined ({len(rest)} seeds left, reverse order)")
                    continue
                res = sh(m["host"], f"cd {m['evroot']}/{name} && ls hanging_mug_seed*.json 2>/dev/null")
                done = {x[len('hanging_mug_seed'):-5] for x in res.split()}
                missing = [x for x in seeds if x not in done]
                if missing and s["passes"] < 3:
                    gs = free(s["machine"], s["gpus"])
                    if not gs:
                        continue
                    sf = f"{m['seeddir']}/seeds100_{name}_retry{s['passes']}.txt"
                    g = gs[0]
                    claimed.add((s["machine"], g))
                    launch(m["host"], f"printf '%s\\n' {' '.join(missing)} > {sf} && cd {m['evroot']} && setsid nohup bash {m['script']} "
                                  f"{g} {name} {s.get('tp', m['ckroot'] + '/' + name)} {sf} hanging_mug {m['evroot']} "
                                  f"> {m['evroot']}/eval_{name}_retry{s['passes']}.log 2>&1 < /dev/null &")
                    s["passes"] += 1
                    log(f"{name}: re-running {len(missing)} INVALID seeds on {s['machine']} GPU {g}")
                    continue
                os.makedirs(f"{OUT}/{name}", exist_ok=True)
                subprocess.run(["rsync", "-a", "--include=*.json", "--include=progress_*", "--exclude=*",
                                f"{m['host']}:{m['evroot']}/{name}/", f"{OUT}/{name}/"])
                succ = sum(json.load(open(f"{OUT}/{name}/hanging_mug_seed{x}.json"))["success"] for x in seeds if x in done)
                s.update(status="done", finished=time.strftime("%F %T"), valid=len(done & set(seeds)), success=succ,
                         still_invalid=missing)
                log(f"{name}: done on {s['machine']}: success {succ} / valid {s['valid']} (still invalid: {missing})")
                if s["machine"] == "old":
                    sh(C, f"rm -rf {M['old']['ckroot']}/{name}")
            json.dump(st, open(st_path, "w"), indent=1)
        json.dump(st, open(st_path, "w"), indent=1)


if __name__ == "__main__":
    main()
