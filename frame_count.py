import os
import cv2
import glob

for file in glob.glob("*.mp4"):
    cap= cv2.VideoCapture(file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total Frame Count of', file, length )