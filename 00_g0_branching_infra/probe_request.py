"""Strict opt-in probe wire protocol; used by patched server, no torch import."""
def request_seeds(observation, count):
    seeds = observation.get('probe_seeds')
    if type(count) is not int or count < 1 or count > 8:
        raise ValueError('probe count must be 1..8')
    if not isinstance(seeds, list) or len(seeds) != count:
        raise ValueError('one explicit probe seed per candidate is required')
    if any(type(s) is not int or not 0 <= s < 2**63 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct uint63 integers')
    return seeds[:]


def attach_probe_request(observation, seeds):
    result = dict(observation)
    result['best_of_N'] = len(seeds)
    result['probe_seeds'] = list(seeds)
    request_seeds(result, len(seeds))
    return result
