"""Subclass factory for the exported ModelClient; no upstream edits or imports.
Caller binds a unique start before each episode. sink must persist request/response.
"""
from seed_plan import seed_for
from probe_request import attach_probe_request

class RecordingTransport:
    def __init__(self, transport, sink):
        self.transport, self.sink = transport, sink
    def inference(self, request):
        self.sink('request', request)
        response = self.transport.inference(request)
        self.sink('response', response)
        if response.get('probe_seeds') != request['probe_seeds']:
            raise ValueError('server seed receipt mismatch')
        return response
    def __getattr__(self, name):
        return getattr(self.transport, name)


def probe_client_class(base):
    class ProbeClient(base):
        def bind_probe(self, start_id, root_seed, sink, repeat=0, count=1):
            if type(count) is not int or not 1 <= count <= 8:
                raise ValueError('count must be 1..8')
            if count > 1 and repeat != 0:
                raise ValueError('candidate pool is generated once, not per repetition')
            seed_for(root_seed, start_id, 'continuation', 0, repeat=repeat)
            if isinstance(self.client, RecordingTransport):
                self.client = self.client.transport
            self.client = RecordingTransport(self.client, sink)
            self._probe_binding = (start_id, root_seed, repeat, count)
            self._probe_decision = 0
        def reset(self):
            super().reset()
            self._probe_binding = None
        def _build_request(self, example):
            if not getattr(self, '_probe_binding', None):
                raise ValueError('bind_probe required after reset')
            start, root, repeat, count = self._probe_binding
            d = self._probe_decision
            seeds = ([seed_for(root, start, 'candidate', d, candidate=i) for i in range(count)]
                     if count > 1 else [seed_for(root, start, 'continuation', d, repeat=repeat)])
            request = attach_probe_request(super()._build_request(example), seeds)
            request['probe_decision'] = d
            self._probe_decision += 1
            return request
    return ProbeClient
