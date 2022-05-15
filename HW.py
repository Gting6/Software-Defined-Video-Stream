from turtle import pos
import cv2
import argparse
import multiprocessing as multiprocess
import mediapipe as mp
import numpy as np



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
    cnt = 0
    try:
        while True:
            _, frame = cap.read()  # 一直讀 frame 出來，numpy 格式，RGB 3 channel
            # if not ret:
            #     break
            if cnt % 5 == 0:
                queue.put(frame)
                cnt = 0
            cnt += 1
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

    # cnt = 0

    while True:
        frame = queue.get()
        # cnt += 1
        # if cnt % 5 == 0:
            # frame = cv2.flip(frame, 0)
        # frame = hand_tracking(frame)
        # frame = object_detection(frame)
        frame = post_estimation(frame)
        # im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5) as pose:

        # image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        return image




# Complelte the code
if __name__ == '__main__':
    # image = cv2.imread('shao.jpg')
    # result = post_estimation(image)
    # cv2.imwrite('mp-shao.jpg', result)

    queue = multiprocess.Queue(maxsize=300)

    p1 = multiprocess.Process(target=gstreamer_camera, args=(queue,))
    p2 = multiprocess.Process(target=gstreamer_rtmpstream, args=(queue,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
