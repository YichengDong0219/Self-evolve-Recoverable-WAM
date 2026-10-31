"""Replay recorded *executed* qpos commands, never rerun the model to recover a queue.
Capture callbacks must export task-specific state. No snapshot validity claim here.
"""
import hashlib
import json
import math


def action_digest(actions):
    for row in actions:
        if not isinstance(row, list) or len(row) != 14 or any(type(x) not in (float,int) or not math.isfinite(x) for x in row):
            raise ValueError('expected finite 14-dimensional executed qpos commands')
    return hashlib.sha256(json.dumps(actions,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def replay(task, actions, expected_sha, capture, emit, stop_before=None):
    if action_digest(actions) != expected_sha:
        raise ValueError('executed action manifest mismatch')
    if stop_before is None: stop_before=len(actions)
    if type(stop_before) is not int or not 0 <= stop_before <= len(actions):
        raise ValueError('invalid branch boundary')
    if int(task.take_action_cnt) != 0 or task.eval_success:
        raise ValueError('replay requires a fresh independently initialized scene')
    emit({'kind':'state','index':0,'state':capture(task)})
    for i, action in enumerate(actions[:stop_before]):
        if task.eval_success or task.take_action_cnt >= task.step_lim:
            raise RuntimeError('original prefix cannot be replayed: early termination')
        before=task.take_action_cnt
        # Exact client commands are reused, no model calls/rounding/queue recomputation.
        task.take_action(action, action_type='qpos')
        if task.take_action_cnt != before+1:
            raise RuntimeError('command was not executed exactly once')
        emit({'kind':'action','index':i,'qpos':action})
        emit({'kind':'state','index':i+1,'state':capture(task)})
    return {'executed':stop_before,'actions_sha256':expected_sha,'state':capture(task),
            'branch_validated':False}


def capture_physics(task):
    """Raw SAPIEN state inventory, separate from unit-aware acceptance schema.
    Missing task/control/RNG coverage remains explicit; never fills with {}.
    """
    def arr(x): return x.tolist()
    actors={}
    for i, a in enumerate(task.scene.get_all_actors()):
        pose=a.get_pose()
        actors[str(i)+':'+a.get_name()]={'p':arr(pose.p),'q_wxyz':arr(pose.q),
          'velocity':arr(a.get_velocity()),'angular_velocity':arr(a.get_angular_velocity())}
    joints={}
    for i, a in enumerate(task.scene.get_all_articulations()):
        joints[str(i)+':'+a.get_name()]={'qpos':arr(a.get_qpos()),'qvel':arr(a.get_qvel()),
          'drive_target':arr(a.get_drive_target()),'drive_velocity_target':arr(a.get_drive_velocity_target())}
    return {'actors':actors,'articulations':joints,'action_step':int(task.take_action_cnt),
      'eval_success':bool(task.eval_success),'missing':['task_stage','success_counter','rng','gripper_target','client_queue','controller_state'],
      'acceptance_ready':False}
