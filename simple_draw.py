import cv2 
from multiprocessing import Process, Queue
import mediapipe as mp 
import os
import time 
import numpy as np
import pdb

frame_shape = (1080, 1920)
canvas_shape = (1080, 1920)
# Initialize webcam video capture
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[1])  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[0])  # Set height
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Set brightness
print(cap.get(cv2.CAP_PROP_FPS))


# Initialize mediapipe Hands object for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

pasttime = 0

xp, yp = 0, 0
# Create a blank canvas to draw on
canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), np.uint8)


def capture_frames(q):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            q.put(frame)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(img)
    lanmark = []

    if results.multi_hand_landmarks:
        for hn in results.multi_hand_landmarks:
            for id, lm in enumerate(hn.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lanmark.append([id, cx, cy])
            mpdraw.draw_landmarks(frame, hn, mpHands.HAND_CONNECTIONS)
    
    if len(lanmark) != 0:
        x1, y1 = lanmark[8][1], lanmark[8][2]
        cv2.line(frame, (xp, yp), (x1, y1), (0,0,255), 20, cv2.FILLED)
        cv2.line(canvas, (xp, yp), (x1, y1), (0,0,255), 20, cv2.FILLED)
        xp, yp = x1, y1

    # Prepare the canvas for blending with the frame
    # imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    # imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # # Use bitwise operations to blend the frame with the canvas
    # frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    # Calculate and display the frames per second (FPS) on the frame
    ctime = time.time()
    fps = 1 / (ctime - pasttime)
    pasttime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (490, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    # Show the webcam frame and the canvas
    cv2.imshow('cam', frame)
    # cv2.imshow('canvas', canvas)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), np.uint8)