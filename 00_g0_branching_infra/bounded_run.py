"""Single-host foreground runner. One explicit GPU; persistent conservative ledger.
Child must not daemonize/setsid. This runner never signals a pre-existing process.
Hard deadline includes a 20s termination allowance. Crashes keep full reservation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def gpu_empty(uuid):
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
    selected = None
    for line in r.stdout.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)!=3 or not parts[1].isdigit() or not parts[2].isdigit(): raise ValueError('unknown GPU table')
        if parts[0]==uuid: selected=parts
    if selected is None: raise ValueError('unknown GPU UUID')
    if int(selected[1])>64 or int(selected[2])!=0: return False
    r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
    for line in r.stdout.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)!=2 or not parts[0].startswith('GPU-') or not parts[1].isdigit(): raise ValueError('unknown process table')
        if parts[0]==uuid: return False
    return True


def save(path,data):
    t=path.with_suffix('.tmp'); t.write_text(json.dumps(data,indent=2)+'\n'); os.replace(t,path)


def run(root, uuid, seconds, command):
    if not 60<=seconds<=2700 or not command: raise ValueError('invalid bounded invocation')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=root/'ledger.json'
        state=json.loads(ledger.read_text()) if ledger.exists() else {'used_gpu_seconds':0,'active':None,'runs':[]}
        if state['active'] or state['used_gpu_seconds']+seconds>8*3600: raise ValueError('unresolved run or G0 budget exceeded')
        if not gpu_empty(uuid): raise ValueError('GPU occupied; no wait or launch')
        out=root/('run_%04d'%len(state['runs'])); out.mkdir(exist_ok=False)
        start=time.monotonic(); deadline=start+seconds
        state['active']={'gpu':uuid,'reserved':seconds,'start_unix':time.time(),'output':str(out)}
        save(ledger,state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1')
        proc=None; rc=None
        try:
            with (out/'stdout').open('wb') as stdout,(out/'stderr').open('wb') as stderr:
                proc=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                save(out/'launch.json',{'pid':proc.pid,'command':command,'gpu':uuid,'seconds':seconds})
                try: rc=proc.wait(timeout=max(0,deadline-time.monotonic()-20))
                except subprocess.TimeoutExpired: pass
                # Also reap surviving same-group workers after normal parent exit.
                try: os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: proc.wait(timeout=min(5,max(0,deadline-time.monotonic()-15)))
                except subprocess.TimeoutExpired: pass
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait(timeout=max(.01,deadline-time.monotonic()-10))
            clean=gpu_empty(uuid)
            elapsed=time.monotonic()-start
            row={'rc':rc,'wall_seconds':elapsed,'gpu':uuid,'cleanup_gpu_empty':clean,'within_deadline':elapsed<=seconds,'output':str(out)}
            save(out/'result.json',row)
            state['runs'].append(row)
            if clean and elapsed<=seconds:
                state['used_gpu_seconds']+=elapsed; state['active']=None
            # unresolved or overrun retains full reservation, cannot start a new run.
            save(ledger,state)
            return 0 if clean and elapsed<=seconds and rc==0 else 1
        except BaseException:
            if proc is not None:
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--seconds',type=int,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    raise SystemExit(run(a.root,a.gpu_uuid,a.seconds,command))
