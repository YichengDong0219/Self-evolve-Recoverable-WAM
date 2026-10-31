"""Fixed all-20 descriptive cross-configuration audit. No threshold fitting."""
import argparse,json,hashlib
from pathlib import Path
from onset_timing_codex import detect
from mug_failure_onset import onset

def analyze(root):
    entries=[]; counts={'failure_detected':0,'failure_not_detected':0,'success_flagged':0,'success_not_flagged':0,'missing_or_invalid':0}
    for i in range(20):
        path=root/f'IMG_hanging_mug_demo_clean_{i:02d}.label.json'
        row={'episode_index':i,'path':str(path)}
        if not path.exists():
            row['status']='missing';counts['missing_or_invalid']+=1
        else:
            raw=path.read_bytes();d=json.loads(raw)
            row.update(sha256=hashlib.sha256(raw).hexdigest(),env_seed=d['env_seed'],success=d['success'])
            if not d['replay_outcome_matches'] or not d.get('final_state_matches',False):
                row['status']='invalid_replay';counts['missing_or_invalid']+=1
            else:
                event=detect(d['rows']);row.update(status='valid',timing=event,original=onset(d['rows']),expert_check_attempts=d.get('expert_check_attempts'))
                key=('success_flagged' if event else 'success_not_flagged') if d['success'] else ('failure_detected' if event else 'failure_not_detected')
                counts[key]+=1
        entries.append(row)
    failure_n=counts['failure_detected']+counts['failure_not_detected'];success_n=counts['success_flagged']+counts['success_not_flagged']
    return {'scope':'same/overlapping scenes across config; not independent holdout','planned_n':20,'counts':counts,
      'failure_detection_rate':counts['failure_detected']/failure_n if failure_n else None,
      'success_false_alarm_rate':counts['success_flagged']/success_n if success_n else None,
      'complete':counts['missing_or_invalid']==0,'entries':entries}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); result=analyze(a.labels)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
