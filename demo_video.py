import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import time
import imutils

from src import model
from src import util
from src.body import Body
from src.hand import Hand


body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

folder = '/Users/dishangzhe/Desktop/VirtualTryOn/pytorch-openpose'
video_path = folder + '/video/input.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    start = time.time()
    
    success, frame = cap.read()
    if not success:
        break

    frame = imutils.resize(frame, width=256)

    candidate, subset = body_estimation(frame)
    canvas = copy.deepcopy(frame)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    print(time.time() - start)

    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()
