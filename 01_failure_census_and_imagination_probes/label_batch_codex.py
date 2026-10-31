"""First bounded IMG validation batch, no model; one attempt per rollout.
Executed inside bounded_run, same process group; failures abort, never silently drop.
"""
import argparse,json,os,subprocess,time
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True);a=p.parse_args()
    sources=[a.runs/f'IMG_hanging_mug_demo_clean_{i:02d}.json' for i in range(20)]
    if not all(x.is_file() for x in sources): raise SystemExit('fixed IMG sources incomplete; do not substitute')
    a.out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,FACT_PROBE_DIR=str(a.code),PYTHONDONTWRITEBYTECODE='1',
      PYTHONPATH='/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:'+str(a.code),
      CUDA_HOME='/usr/local/cuda-12.4',TORCH_CUDA_ARCH_LIST='8.9',
      TORCH_EXTENSIONS_DIR='/workspace_cluster/.cache/openwam-robotwin-torch-extensions')
    env['PATH']='/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:'+env['PATH']
    with (a.out/'journal.jsonl').open('x') as journal:
        for source in sources:
            target=a.out/(source.stem+'.label.json'); start=time.monotonic()
            with (a.out/(source.stem+'.stdout')).open('xb') as stdout,(a.out/(source.stem+'.stderr')).open('xb') as stderr:
                rc=subprocess.call(['/workspace_cluster/miniconda3/envs/robotwin/bin/python',str(a.code/'p1_label.py'),'--rollout',str(source),'--out',str(target)],env=env,cwd='/workspace_cluster/RoboTwin-FACT',stdout=stdout,stderr=stderr)
            journal.write(json.dumps({'source':str(source),'rc':rc,'wall_s':time.monotonic()-start})+'\n');journal.flush()
            if rc: raise SystemExit(rc)
            d=json.loads(target.read_text())
            if not d['replay_outcome_matches'] or not d.get('final_state_matches',False): raise SystemExit('replay mismatch; batch stops')
