"""Multi-connection FACT probe server (Claude, 2026-11-22; user: "一个服务是否可以对接多个环境，最大化利用显卡资源").
Same policy, arguments and output as fact_local/code/inference_server_probe.py (its main() is reused unchanged); only
the TCP front end differs: every accepted connection is served in its own thread and the inference handler is
serialized by one lock, so each request is computed exactly as by the single-connection server. Requests are
independent: the pipeline's prefix KV cache is built and dropped inside one call, the server keeps no episode
state, the sampling seeds come with each request, and inference() is wrapped in torch.no_grad() itself.
CPU threads are capped (FACT_SERVER_THREADS, default 4): with the default (one OpenMP thread per core = 192 on
Amax) 13 servers used ~160 cores on 11-22 14:05 mostly spinning.
usage: python inference_server_multi.py <inference_server_probe.py arguments>
"""
import os
import socket
import threading

N = os.environ.get("FACT_SERVER_THREADS", "4")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, N)

import torch  # noqa: E402

torch.set_num_threads(int(N))
torch.set_num_interop_threads(max(1, int(N) // 2))

import inference_server_probe as P  # noqa: E402
from world_action_model.sockets import RobotInferenceServer, _set_tcp_socket_options  # noqa: E402


class MultiConnectionServer(RobotInferenceServer):
    def __init__(self, model, host="127.0.0.1", port=5555):
        super().__init__(model, host, port)
        self.server_socket.listen(64)
        lock = threading.Lock()
        inner = self._endpoints["inference"].handler

        def locked(data, _inner=inner):
            with lock:
                return _inner(data)
        self.register_endpoint("inference", locked)

    def run(self):
        print(f"Server is ready and listening on tcp://{self.host}:{self.port} (transport=tcp, multi-connection, "
              f"cpu threads {torch.get_num_threads()})", flush=True)
        while self.running:
            try:
                conn, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            _set_tcp_socket_options(conn)
            threading.Thread(target=self._serve_connection, args=(conn,), daemon=True).start()
        self.server_socket.close()


P.RobotInferenceServer = MultiConnectionServer      # P.main() builds the server through this module-level name

if __name__ == "__main__":
    P.main()
