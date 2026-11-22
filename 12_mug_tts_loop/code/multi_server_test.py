"""Test of the multi-connection FACT server (Claude, 2026-11-22; user: "一个服务是否可以对接多个环境，最大化利用显卡资源").
System python3 on Amax. Runs on fully idle cards only (A and B in parallel); never touches other users' processes.
 A. server on a 4090 (--a-gpu, fact env), drivers on the same card:
    A1 one driver, 170001 decision 0 candidate 0, K = 1   -> action hash must equal the gate row
    A2 three concurrent drivers, candidates 1, 2, 3       -> each hash must equal its gate row
    A3 throughput: n = 1, 2, 4 concurrent drivers x 2 pairs each (decisions 1-2 of 170001; hashes vs gate rows too)
 B. server on --b-server-gpu (--b-env fact = 4090, fact_bw = PRO 6000), drivers on the 4090 --b-driver-gpu:
    B1 one driver, candidate 0 -> must equal the gate row (4090 server) / the single-connection PRO 6000 smoke hash
    B2 candidates 1, 2, 3 three concurrent, then the same three one at a time -> hashes must agree (and vs gate)
    B3 throughput: n = 1, 2, 4, 6 concurrent drivers x 2 pairs each
Per batch: wall time, pairs per minute, server-card GPU utilization (sampled every second), server CPU cores used
(utime+stime over wall), peak memory of the server card and of the driver card.
15:16 the first attempt aborted (PRO 6000 smi 0 busy with another user's training) -> B now defaults to a 4090
server (smi 6) with drivers on smi 7; the PRO 6000 variant runs later with --parts B --b-env fact_bw ... when free.
Output: r0/multi_test/<tag>/summary.json, summary.txt, MULTI_TEST_DONE, multi_test.log.
usage: setsid nohup python3 code/multi_server_test.py [--wait] [--parts A,B] [--tag t] > /dev/null 2>&1 < /dev/null &
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

M = Path("/workspace/fact_local/mug_loop_1121"); RD = M / "r0"
CODE = M / "code"; ENV = "/workspace/fact_local/env.sh"; SDIR = Path("/workspace/fact_local/runs_server")
EP = RD / "hanging_mug_seed170001.json"
FAST = "--enable_prefix_cache --skip_future_state_value"
SMOKE_BW_C0 = "7b715051e7a27459"
T = RD / "multi_test"; LOG = T / "multi_test.log"


def log(m):
    with LOG.open("a") as f:
        f.write(time.strftime("%F %T ") + m + "\n")


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def mem(g):
    u = sh(f"nvidia-smi -i {g} --query-gpu=memory.used --format=csv,noheader,nounits")
    return int(u) if u.isdigit() else None


def idle(g):
    return sh(f"nvidia-smi -i {g} --query-compute-apps=pid --format=csv,noheader") == "" and (mem(g) or 0) < 1024


def cpu_s(pid):
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except OSError:
        return None


def gate_rows():
    out = {}
    for r in map(json.loads, (RD / "hanging_mug_seed170001.cont.jsonl").read_text().splitlines()):
        if r.get("start_ok"):
            out[(r["decision"], r["cand"])] = r
    return out


def start(v, g, port, env):
    sh(f"bash {CODE}/start_server_multi.sh {v} {g} {port} {env} {FAST}")
    d = SDIR / f"server_{v}"
    for _ in range(120):
        if "ready and listening" in (d / "server.log").read_text(errors="ignore"):
            return d, int((d / "pid").read_text().strip())
        time.sleep(5)
    raise RuntimeError(f"server {v} not ready")


def stop(d):
    pid = (d / "pid").read_text().strip()
    sh(f"kill -- -$(ps -o pgid= -p {pid} | tr -d ' ') 2>/dev/null; sleep 5; kill -9 {pid} 2>/dev/null; true")


def driver(od, pairs, port, dg):
    od.mkdir(parents=True, exist_ok=True)
    cmd = (f"FACT_ARCH=8.9 source {ENV} {dg} > /dev/null; export PYTHONPATH=$PYTHONPATH:{CODE}; "
           f"timeout 1800 python {CODE}/continue_record.py --episode {EP} --pairs {','.join(f'{d}:{i}' for d, i in pairs)} "
           f"--port {port} --unit 1 --out-dir {od} --img-k 0")
    with (od / "driver.log").open("w") as lf:
        subprocess.run(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT)
    f = od / "seed170001.unit1.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    return {(r["decision"], r["cand"]): r for r in rows if r.get("start_ok") and "success" in r}


def batch(out, name, jobs, port, dg, gpu, spid):
    """jobs: list of pair lists, one driver each, all concurrent."""
    util, mx, stop_ev = [], {gpu: 0, dg: 0}, threading.Event()

    def sample():
        while not stop_ev.is_set():
            u = sh(f"nvidia-smi -i {gpu} --query-gpu=utilization.gpu --format=csv,noheader,nounits")
            if u.isdigit():
                util.append(int(u))
            for g in mx:
                mx[g] = max(mx[g], mem(g) or 0)
            time.sleep(1)
    th = threading.Thread(target=sample); th.start()
    res, threads = {}, []
    c0, t0 = cpu_s(spid), time.time()
    for j, pairs in enumerate(jobs):
        def run(j=j, pairs=pairs):
            res.update(driver(out / f"{name}_d{j}", pairs, port, dg))
        threads.append(threading.Thread(target=run)); threads[-1].start()
    for t in threads:
        t.join()
    wall, c1 = time.time() - t0, cpu_s(spid)
    stop_ev.set(); th.join()
    n = len(res)
    o = dict(drivers=len(jobs), pairs=n, wall_s=round(wall, 1), pairs_per_min=round(60 * n / wall, 2) if wall else None,
             gpu_util_mean=round(sum(util) / len(util), 1) if util else None,
             server_cpu_cores=round((c1 - c0) / wall, 2) if c0 is not None and c1 is not None else None,
             peak_mem_mb={str(g): v for g, v in mx.items()})
    log(f"{name}: {o}")
    return res, o


def vs_gate(r, gs):
    return dict(equal=sum(r[k]["cont_actions_sha"] == gs[k]["cont_actions_sha"] for k in r if k in gs),
                compared=sum(k in gs for k in r),
                differ=[[*k, gs[k]["host"]] for k in sorted(r) if k in gs and r[k]["cont_actions_sha"] != gs[k]["cont_actions_sha"]])


def part_a(summary, gs, out, g):
    port = 8500
    d, spid = start(f"multitest_a_gpu{g}", g, port, "fact")
    r, o = batch(out, "A1", [[(0, 0)]], port, g, g, spid)
    summary["A1"] = dict(o, hash=r[(0, 0)]["cont_actions_sha"], gate=gs[(0, 0)]["cont_actions_sha"],
                         equal=r[(0, 0)]["cont_actions_sha"] == gs[(0, 0)]["cont_actions_sha"])
    r, o = batch(out, "A2", [[(0, 1)], [(0, 2)], [(0, 3)]], port, g, g, spid)
    summary["A2"] = dict(o, vs_gate=vs_gate(r, gs))
    pool = [(dd, i) for dd in (1, 2) for i in range(8)]
    summary["A3"] = {}
    for n in (1, 2, 4):
        r, o = batch(out, f"A3_n{n}", [pool[2 * j:2 * j + 2] for j in range(n)], port, g, g, spid)
        summary["A3"][n] = dict(o, vs_gate=vs_gate(r, gs))
    stop(d)


def part_b(summary, gs, out, g, dg, env):
    port = 8510
    d, spid = start(f"multitest_b_gpu{g}", g, port, env)
    ref = SMOKE_BW_C0 if env == "fact_bw" else gs[(0, 0)]["cont_actions_sha"]
    r, o = batch(out, "B1", [[(0, 0)]], port, dg, g, spid)
    summary["B1"] = dict(o, server_env=env, hash=r[(0, 0)]["cont_actions_sha"], reference=ref,
                         equal=r[(0, 0)]["cont_actions_sha"] == ref)
    rc, oc = batch(out, "B2_concurrent", [[(0, 1)], [(0, 2)], [(0, 3)]], port, dg, g, spid)
    rs = {}
    for k in ((0, 1), (0, 2), (0, 3)):
        rr, _ = batch(out, f"B2_seq_c{k[1]}", [[k]], port, dg, g, spid)
        rs.update(rr)
    summary["B2"] = dict(oc, concurrent_equals_sequential=[rc[k]["cont_actions_sha"] == rs[k]["cont_actions_sha"]
                                                           for k in sorted(rc) if k in rs],
                         vs_gate=vs_gate(rc, gs) if env == "fact" else None)
    pool = [(dd, i) for dd in (1, 2) for i in range(8)]
    summary["B3"] = {}
    for n in (1, 2, 4, 6):
        r, o = batch(out, f"B3_n{n}", [pool[2 * j:2 * j + 2] for j in range(n)], port, dg, g, spid)
        summary["B3"][n] = dict(o, vs_gate=vs_gate(r, gs) if env == "fact" else None)
    stop(d)


def main():
    global T, LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="wait for r0/unit_measure/UNIT_ALL_DONE first")
    ap.add_argument("--parts", default="A,B"); ap.add_argument("--tag", default="4090")
    ap.add_argument("--a-gpu", type=int, default=4)
    ap.add_argument("--b-server-gpu", type=int, default=6); ap.add_argument("--b-driver-gpu", type=int, default=7)
    ap.add_argument("--b-env", default="fact", choices=["fact", "fact_bw"])
    a = ap.parse_args()
    T = RD / "multi_test" / a.tag; T.mkdir(parents=True, exist_ok=True); LOG = T / "multi_test.log"
    parts = a.parts.split(",")
    log(f"start {vars(a)}")
    if a.wait:
        while not (RD / "unit_measure" / "UNIT_ALL_DONE").exists():
            time.sleep(60)
        time.sleep(60)
    cards = ([a.a_gpu] if "A" in parts else []) + ([a.b_server_gpu, a.b_driver_gpu] if "B" in parts else [])
    for g in cards:
        if not idle(g):
            log(f"gpu {g} not idle; abort"); return
    gs = gate_rows()
    summary = {}

    def safe(f, *x):
        try:
            f(*x)
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f"{f.__name__} failed: {e!r}\n{traceback.format_exc()}")
    th = []
    if "A" in parts:
        th.append(threading.Thread(target=safe, args=(part_a, summary, gs, T, a.a_gpu)))
    if "B" in parts:
        th.append(threading.Thread(target=safe, args=(part_b, summary, gs, T, a.b_server_gpu, a.b_driver_gpu, a.b_env)))
    for t in th:
        t.start(); time.sleep(30)
    for t in th:
        t.join()
    (T / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (T / "summary.txt").write_text("\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n")
    log("done")
    (T / "MULTI_TEST_DONE").write_text(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    main()
