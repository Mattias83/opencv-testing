import cv2
import cvzone
import os

from cvzone.PoseModule import PoseDetector


detector = PoseDetector()
snapshot_url = os.getenv("CAMERA_SNAPSHOT_URL", "")
img = cvzone.downloadImageFromUrl(url=snapshot_url)


img = detector.findPose(img)
lmList, bboxInfo = detector.findPosition(img)
cv2.imwrite("snapshot.jpg", img)
