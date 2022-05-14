import os
import os.path as osp
from concurrent import futures
import grpc
import argparse
import sys
BUILD_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "build/service/")
sys.path.insert(0, BUILD_DIR)
import video_pb2_grpc
import video_pb2


class VideoServicer(video_pb2_grpc.VideoProcessorServicer):

    def __init__(self):
        pass

    def Compute(self, request, context):
        n = request.algorithm
        value = self._process(n)

        response = video_pb2.VideoResponse()
        response.value = value

        return response

    # n is a string
    def _process(self, n):
        if n == 1:
            return 1
        elif n == 2:
            return 2
        elif n == 3:
            return 3
        else:
            print("Something wrong at grpc server")
            return 0  # Failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8080, type=int)
    args = vars(parser.parse_args())

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = VideoServicer()
    video_pb2_grpc.add_VideoProcessorServicer_to_server(servicer, server)

    try:
        server.add_insecure_port(f"{args['ip']}:{args['port']}")
        server.start()
        print(f"Run gRPC Server at {args['ip']}:{args['port']}")
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
