# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import argparse
import grpc
import logging
import os
import pathlib
import signal
import subprocess
import sys
from concurrent import futures

from smarts.zoo import master_pb2_grpc
from smarts.zoo import master_servicer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"master.py - pid({os.getpid()})")


def serve(port):
    ip = "[::]"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    master_servicer_object = master_servicer.MasterServicer()
    master_pb2_grpc.add_MasterServicer_to_server(master_servicer_object, server)
    server.add_insecure_port(f"{ip}:{port}")
    server.start()
    log.debug(f"Master - ip({ip}), port({port}), pid({os.getpid()}): Started serving.")

    def stop_server(unused_signum, unused_frame):
        master_servicer_object.destroy()
        server.stop(0)
        log.debug(
            f"Master - ip({ip}), port({port}), pid({os.getpid()}): Received interrupt signal."
        )

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    # Wait to receive server termination signal
    server.wait_for_termination()
    log.debug(f"Master - ip({ip}), port({port}), pid({os.getpid()}): Server exited")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Listen for requests to allocate agents and execute them on-demand."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7432,
        help="Port to listen for remote client connections.",
    )

    args = parser.parse_args()
    serve(args.port)
