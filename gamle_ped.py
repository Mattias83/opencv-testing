import cv2
import os

pedestrian_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
camera_url = os.getenv("RTSP_URL", "")
vcap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
count = 0
while 1:
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        break
    else:
        pedestrians = pedestrian_cascade.detectMultiScale(frame, 1.1, 1)
        for x, y, w, h in pedestrians:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, "Thief!", (x + 6, y - 6), font, 0.5, (0, 255, 0), 1
            )
        cv2.imwrite("frame%d.jpg" % count, frame)
        count += 1
