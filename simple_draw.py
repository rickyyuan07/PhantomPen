import cv2
import mediapipe as mp
import numpy as np
import time
from multiprocessing import Process, Queue
from collections import deque

class PhantomPen:
    FRAME_WIDTH, FRAME_HEIGHT = 640, 480
    CANVAS_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)

    def __init__(self):
        """Initialize camera, Mediapipe, and canvas"""
        self.frame_queue = Queue()
        self.capture_process = Process(target=self.capture_frames, args=(self.frame_queue,))
        self.capture_process.start()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        self.canvas = np.zeros(self.CANVAS_SHAPE, np.uint8)
        self.points = deque(maxlen=1024)  # Stores drawing points
        self.prev_x, self.prev_y = 0, 0
        self.past_time = time.time()

    def capture_frames(self, queue):
        """Captures frames asynchronously using multiprocessing"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Set brightness
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set buffer size to 1 to reduce latency

        while True:
            ret, frame = cap.read()
            if ret:
                queue.put(frame)

    def catmull_rom_spline(self, P0, P1, P2, P3, num_points=20):
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

    def smooth_draw(self):
        """Draw smooth curves using Catmull-Rom splines."""
        if len(self.points) < 4:
            return
        for i in range(1, len(self.points) - 2):
            smooth_points = self.catmull_rom_spline(self.points[i-1], self.points[i], self.points[i+1], self.points[i+2])
            for j in range(1, len(smooth_points)):
                cv2.line(self.canvas, smooth_points[j-1], smooth_points[j], (0, 255, 0), 2, cv2.LINE_AA)

    def process_frame(self, frame):
        """Process frame to detect hand landmarks and draw on canvas"""
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((id, cx, cy))
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame, landmarks

    def draw_on_frame(self, frame):
        """Draw overlay of the canvas on the frame"""
        frame[:] = cv2.addWeighted(frame, 1.0, self.canvas, 0.5, 0)

    def reset_canvas(self):
        """Clear the drawing canvas"""
        self.canvas = np.zeros(self.CANVAS_SHAPE, np.uint8)
        self.points.clear()

    def run(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                frame, landmarks = self.process_frame(frame)

                if landmarks:
                    x1, y1 = landmarks[8][1], landmarks[8][2]  # Index finger tip
                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = x1, y1
                    self.points.append((x1, y1))
                    self.smooth_draw()

                self.draw_on_frame(frame)

                # Display FPS
                curr_time = time.time()
                fps = int(1 / (curr_time - self.past_time))
                self.past_time = curr_time
                cv2.putText(frame, f'FPS: {fps}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

                cv2.imshow("PhantomPen", frame)

                # Key press handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):  # Clear canvas
                    self.reset_canvas()
                elif key == ord('q'):  # Quit
                    break

        cv2.destroyAllWindows()
        self.capture_process.terminate()

if __name__ == "__main__":
    app = PhantomPen()
    app.run()