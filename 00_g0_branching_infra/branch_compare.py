"""CPU comparison of adapter-exported states; not a snapshot implementation.
Each metric group maps stable object/joint names to flat numeric lists.
Missing/empty groups fail closed. Quaternion order is wxyz, angle in radians.
Caller must compare every frozen timepoint, including the terminal state.
"""
import math
import json

LIMITS = {'position_m': .001, 'qpos_rad': .0001,
          'rotation_wxyz': .001, 'linear_velocity_m_s': .001,
          'angular_velocity_rad_s': .001, 'qvel_rad_s': .001,
          'drive_target': .0001, 'gripper_target': .0001}
EXACT = ('task_variables', 'success_counter', 'action_step', 'controller_state',
         'interpolation_state', 'rng_state', 'remaining_action_queue', 'official_success')


def vector(x):
    if not isinstance(x, list) or not x or any(type(v) not in (int, float) or not math.isfinite(v) for v in x):
        raise ValueError('nonfinite, empty or nonnumeric vector')
    return x


def error(a, b, metric):
    a, b = vector(a), vector(b)
    if len(a) != len(b):
        raise ValueError('shape mismatch')
    if metric == 'rotation_wxyz':
        if len(a) != 4:
            raise ValueError('quaternion requires 4 values')
        na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
        if min(na, nb) < 1e-12:
            raise ValueError('zero quaternion')
        return 2 * math.acos(min(1., abs(sum(x*y for x, y in zip(a, b)) / na / nb)))
    if metric in ('position_m', 'linear_velocity_m_s', 'angular_velocity_rad_s'):
        if len(a) != 3:
            raise ValueError('xyz requires 3 values')
        return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
    return max(abs(x-y) for x, y in zip(a, b))


def compare(a, b):
    issues, maxima = [], {}
    allowed = set(LIMITS) | set(EXACT) | {'png_sha256', 'contact_set'}
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {'pass': False, 'issues': ['state must be a dict']}
    for side, state in [('a', a), ('b', b)]:
        unknown = set(state) - allowed
        if unknown:
            issues.append(side + ': unknown keys ' + str(sorted(unknown)))
    for metric, limit in LIMITS.items():
        aa, bb = a.get(metric), b.get(metric)
        if not isinstance(aa, dict) or not isinstance(bb, dict) or not aa or set(aa) != set(bb):
            issues.append(metric + ': missing/empty/key mismatch'); continue
        errors = []
        for name in aa:
            try:
                e = error(aa[name], bb[name], metric)
                errors.append(e)
                if e >= limit:
                    issues.append(metric + '/' + name + ': tolerance exceeded')
            except (ValueError, TypeError, OverflowError) as exc:
                issues.append(metric + '/' + name + ': ' + str(exc))
        maxima[metric] = max(errors) if errors else None
    for key in EXACT:
        for side, state in [('a', a), ('b', b)]:
            item = state.get(key)
            valid = isinstance(item, dict) and set(item) == {'captured', 'value'} and item['captured'] is True
            if valid:
                value = item['value']
                if key in ('success_counter', 'action_step'):
                    valid = type(value) is int and value >= 0
                elif key == 'official_success':
                    valid = type(value) is bool
                elif key == 'remaining_action_queue':
                    valid = type(value) is list  # captured empty queue is valid, absent queue is not
                else:
                    valid = type(value) is dict and bool(value)
                try:
                    json.dumps(value, allow_nan=False)
                    valid = valid and json.loads(json.dumps(value)) == value
                except (ValueError, TypeError):
                    valid = False
            if not valid:
                issues.append(side + '/' + key + ': invalid capture schema')
        if a.get(key) != b.get(key):
            issues.append(key + ': unequal')
    # These are diagnostic only, not evidence that state restoration is correct.
    diagnostics = {k: a.get(k) == b.get(k) for k in ('png_sha256', 'contact_set')}
    return {'pass': not issues, 'issues': issues, 'max_errors': maxima,
            'diagnostic_equal': diagnostics, 'scope': 'one exported state pair only'}
