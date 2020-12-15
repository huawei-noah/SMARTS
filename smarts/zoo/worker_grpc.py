from concurrent import futures
import argparse
import logging
import sys
import pathlib
import subprocess
import grpc

from smarts.zoo import agent_pb2
from smarts.zoo import agent_pb2_grpc
from smarts.core.utils.networking import find_free_port

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"Worker")

def spawn_networked_agent(port, auth_key):
    cmd = [
        sys.executable,  # path to the current python binary
        str((pathlib.Path(__file__).parent / "run_agent.py").absolute().resolve()),
        "--port",
        str(port),
        "--auth_key",
        auth_key,
    ]

    agent_proc = subprocess.Popen(cmd)
    return agent_proc


def spawn_local_agent(socket_file, auth_key):
    cmd = [
        sys.executable,  # path to the current python binary
        str((pathlib.Path(__file__).parent / "run_agent.py").absolute().resolve()),
        "--port",
        str(port),
        "--auth_key",
        auth_key,
    ]

    agent_proc = subprocess.Popen(cmd)
    return agent_proc

class AgentServicer(agent_pb2_grpc.AgentServicer):
    """Provides methods that implement functionality of agent server."""

    def __init__(self):
        self.agent_procs = []

    def __del__(self):
        # cleanup
        log.debug("Cleaning up zoo worker")
        for proc in self.agent_procs:
            proc.kill()
            proc.wait()
    
    def SpawnWorkers(self, request, context):
        port = find_free_port()

        if request.name == "allocate_networked_agent":
            proc = spawn_networked_agent(port, auth_key)
            if proc:
                agent_procs.append(proc)
            return agent_pb2.Connection(
                agent_pb2.Status(result="success"), 
                port=port)
        else if request.name == "allocate_local_agent":
            proc = spawn_local_agent(auth_key)
            if proc:
                agent_procs.append(proc)
            return agent_pb2.Connection(
                agent_pb2.Status(result="success"), 
                port=port)    
        else:
            return agent_pb2.Connection(
                agent_pb2.Status(result="error"), 
                port=0)
    
def serve(port, auth_key):
    log.debug(f"Starting Zoo Worker on port {port}")
    assert isinstance(
        auth_key, (str, type(None))
    ), f"Received auth_key of type {type(auth_key)}, but need auth_key of type <class 'string'> or <class 'NoneType'>."
    auth_key = auth_key if auth_key else ""
    auth_key_conn = str.encode(auth_key) if auth_key else None

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    agent_pb2_grpc.add_AgentServicer_to_server(
        AgentServicer(), server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    
    def stop_server(unused_signum, unused_frame):
        server.stop(0)
        log.debug("GRPC server stopped.")

    # Catch keyboard interrupt
    signal.signal(signal.SIGINT, stop_server)
    
    server.wait_for_termination()
    log.debug("Serving exited.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Listens for requests to allocate agents and executes them on-demand"
    )
    parser.add_argument(
        "--port", type=int, default=7432, help="Port to listen on",
    )
    parser.add_argument(
        "--auth_key",
        type=str,
        default=None,
        help="Authentication key for connection to run agent",
    )
    args = parser.parse_args()
    auth_key = args.auth_key if args.auth_key else ""
    serve(args.port, auth_key)

