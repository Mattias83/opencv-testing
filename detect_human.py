import cv2
import os
from cvzone.PoseModule import PoseDetector


detector = PoseDetector()
camera_url = os.getenv("RTSP_URL", "")
vcap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
count = 0
while 1:
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        break
    else:
        frame = detector.findPose(frame)
        lmList, bboxInfo = detector.findPosition(frame)
        if lmList:
            print("human detected!")
