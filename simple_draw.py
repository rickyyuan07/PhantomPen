import cv2 
from multiprocessing import Process, Queue
import mediapipe as mp 
import os
import time 
import numpy as np
import pdb

def capture_frames(q):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            q.put(frame)
            
def catmull_rom_spline(P0, P1, P2, P3, num_points=20):
    """Compute Catmull-Rom spline between P1 and P2."""
    curve = []
    for t in np.linspace(0, 1, num_points):
        t2, t3 = t * t, t * t * t
        f1 = -0.5*t3 + t2 - 0.5*t
        f2 = 1.5*t3 - 2.5*t2 + 1
        f3 = -1.5*t3 + 2*t2 + 0.5*t
        f4 = 0.5*t3 - 0.5*t2
        x = int(P0[0] * f1 + P1[0] * f2 + P2[0] * f3 + P3[0] * f4)
        y = int(P0[1] * f1 + P1[1] * f2 + P2[1] * f3 + P3[1] * f4)
        curve.append((x, y))
    return curve

def smooth_draw(points, canvas, color=(0, 255, 0)):
    """Draw smooth curves using Catmull-Rom splines."""
    if len(points) < 4:
        return
    for i in range(1, len(points) - 2):
        smooth_points = catmull_rom_spline(points[i-1], points[i], points[i+1], points[i+2])
        for j in range(1, len(smooth_points)):
            cv2.line(canvas, smooth_points[j-1], smooth_points[j], color, 2, cv2.LINE_AA)

if __name__ == "__main__":
    frame_shape = (480, 640)
    canvas_shape = (480, 640)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[1])  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[0])  # Set height
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Set brightness
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(cap.get(cv2.CAP_PROP_FPS))

    # Initialize mediapipe Hands object for hand tracking
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpdraw = mp.solutions.drawing_utils

    pasttime = 0

    pts = []
    # Create a blank canvas to draw on
    canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), np.uint8)


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
            pts.append((x1, y1))
            smooth_draw(pts[-4:], canvas)

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
            pts = []