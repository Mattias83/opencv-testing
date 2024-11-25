import cv2
import os

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
camera_url = os.getenv("RTSP_URL", "")
cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)

count = 0
while 1:
    ret, frame = cap.read()
    if ret == False:
        print("frame is empty")
        break
    else:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects, weights = hog.detectMultiScale(
            img_gray, winStride=(2, 2), padding=(10, 10), scale=1.02
        )

        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.13:
                print("too small")
                continue
            elif weights[i] < 0.3 and weights[i] > 0.13:
                print("high confidence")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if weights[i] < 0.7 and weights[i] > 0.3:
                print("moderate coonfidence")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 122, 255), 2)
            if weights[i] > 0.7:
                print("low confidence")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                "High confidence",
                (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Moderate confidence",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 122, 255),
                2,
            )
            cv2.putText(
                frame,
                "Low confidence",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.imwrite("frame%d.jpg" % count, frame)
            count += 1
            cv2.waitKey(0)
