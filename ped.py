import cv2
import os
from cvzone.PoseModule import PoseDetector

# pedestrian_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
camera_url = os.getenv("RTSP_URL", "")
detector = PoseDetector()
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
        cv2.imwrite("frame%d.jpg" % count, frame)
        count += 1
