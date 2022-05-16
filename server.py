import os
import os.path as osp
from concurrent import futures
import grpc
import argparse
import sys
from turtle import pos
import cv2
import argparse
import multiprocessing as multiprocess
import mediapipe as mp
import numpy as np

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
            q2.put(1)
            return 1
        elif n == 2:
            q2.put(2)
            return 2
        elif n == 3:
            q2.put(3)
            return 3
        else:
            q2.put(0)
            return 0  


def gstreamer_camera(queue):
    # Use the provided pipeline to construct the video capture in opencv
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)1920, height=(int)1080, "
        "format=(string)NV12, framerate=(fraction)30/1 ! "
        "queue ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, "
        "width=(int)1920, height=(int)1080, "
        "format=(string)BGRx, framerate=(fraction)30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink"
    )
    # Complete the function body
    cap = cv2.VideoCapture(pipeline,  cv2.CAP_GSTREAMER)
    # cnt = 0
    try:
        while True:
            _, frame = cap.read()  # 一直讀 frame 出來，numpy 格式，RGB 3 channel
            # if not ret:
            #     break
            # if cnt % 5 == 0:
            #     queue.put(frame)
            #     cnt = 0
            # cnt += 1
            queue.put(frame)
            # print(queue.qsize())
            # print(time.strftime('%X'), frame.shape)
    except KeyboardInterrupt as e:
        cap.release()
    # pass


def gstreamer_rtmpstream(queue):
    # Use the provided pipeline to construct the video writer in opencv
    pipeline = (
        "appsrc ! "
        "video/x-raw, format=(string)BGR ! "
        "queue ! "
        "videoconvert ! "
        "video/x-raw, format=RGBA ! "
        "nvvidconv ! "
        "nvv4l2h264enc bitrate=8000000 ! "
        "h264parse ! "
        "flvmux ! "
        'rtmpsink location="rtmp://localhost/rtmp/live live=1"'
    )

    writer = cv2.VideoWriter(pipeline, 0, 25.0, (1920, 1080))

    algorithm = 0
    while True:
        frame = queue.get()
        if not q2.empty():
            algorithm = q2.get()
        if algorithm == 1:
            frame = hand_tracking(frame)
        elif algorithm == 2:
            frame = object_detection(frame)
        elif algorithm == 3:
            frame = post_estimation(frame)
        writer.write(frame)
    # print()
    # out.write(frame)

    # pass

def hand_tracking(image):
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils


    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        return image


def object_detection(image):
    mp_object_detection = mp.solutions.object_detection
    mp_drawing = mp.solutions.drawing_utils

    # For static images:

    with mp_object_detection.ObjectDetection(
        min_detection_confidence=0.1) as object_detection:

        results = object_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        return image


def post_estimation(image):
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    #cap = cv2.VideoCapture(0)

    while True:
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # for id, lm in enumerate(results.pose_landmarks.landmark):
        #     h, w,c = image.shape
        #     cx, cy = int(lm.x*w), int(lm.y*h)
        #     cv2.circle(image, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return image






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8080, type=int)
    args = vars(parser.parse_args())

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = VideoServicer()
    video_pb2_grpc.add_VideoProcessorServicer_to_server(servicer, server)


    try:
        q = multiprocess.Queue(maxsize=300)
        q2 = multiprocess.Queue(maxsize=10)

        p1 = multiprocess.Process(target=gstreamer_camera, args=(q, ))
        p2 = multiprocess.Process(target=gstreamer_rtmpstream, args=(q,))

        p1.start()
        p2.start()

        server.add_insecure_port(f"{args['ip']}:{args['port']}")
        server.start()
        print(f"Run gRPC Server at {args['ip']}:{args['port']}")
        server.wait_for_termination()
        p1.join()
        p2.join()

    except KeyboardInterrupt:
        pass


