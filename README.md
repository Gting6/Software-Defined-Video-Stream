# Software-Defined-Video-Stream

## NTUEE NMLAB Homework: Mixture of gstreamer, nginx, protobuf, gmpc.
Client use gmpc protocol to communicate with server.
Server use gstreamer as data producer, nginx as streaming server. 
![image](https://user-images.githubusercontent.com/46078333/168885909-c56673d1-ffc7-4664-8d2d-4a63161ab1c7.png)

## Prerequsite
`pip3 install -r requirements.txt`

## Usage
1. Run the server: `python3 server.py`
2. Run the client: `python3 client.py --ip localhost --order 1` 
  - order 0 = Just streaming, 1 = hand tracking, 2 = object detection, 3 = post estimation.
3. To see the streaming video, Type `ffplay -fflags nobuffer rtmp://192.168.55.1/rtmp/live` at host.

## References
1. Part of the code is modified from TA [johnnylord](https://gist.github.com/johnnylord)
2. Hand tracking, object detection, and post estimation is modified from [MediaPipe documentation](https://mediapipe.readthedocs.io/en/latest/).
