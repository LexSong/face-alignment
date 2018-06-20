import cv2
import sys

from face_detection.api import S3FD
from face_detection.bbox import nms
from face_alignment.api import FaceAlignment
from face_alignment.api import LandmarksType


s3fd = S3FD("models/s3fd_convert.pth")
face_alignment = FaceAlignment(LandmarksType._3D)

score_threshold = 0.5
nms_threshold = 0.3

image = cv2.imread(sys.argv[1])

boxes = s3fd.detect(image, score_threshold)
boxes = boxes[nms(boxes, nms_threshold)]

figure = image.copy()

for box in boxes:
    box = box[0:4]
    landmarks = face_alignment.get_landmarks(image, box)

    for pt in landmarks:
        pt = tuple(pt[0:2].astype(int))
        cv2.circle(figure, pt, 3, (0, 255, 0))

cv2.imshow('', figure)
cv2.waitKey(0)
