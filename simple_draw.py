import cv2
import mediapipe as mp
import numpy as np
import time
import pdb

class PhantomPen:
    FRAME_WIDTH, FRAME_HEIGHT = 640, 480
    CANVAS_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)

    def __init__(self):
        """Initialize camera, Mediapipe, and canvas"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Set brightness
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        self.canvas = np.zeros(self.CANVAS_SHAPE, np.uint8)
        self.points = [[]]  # Stores drawing points
        self.prev_x, self.prev_y = 0, 0
        self.past_time = time.time()
        self.signiture_idx = 0

    def catmull_rom_spline(self, P0, P1, P2, P3, num_points=20):
        """Compute Catmull-Rom spline between P1 and P2."""
        t = np.linspace(0, 1, num_points)
        t2 = t * t
        t3 = t2 * t
        f1 = -0.5 * t3 + t2 - 0.5 * t
        f2 = 1.5 * t3 - 2.5 * t2 + 1
        f3 = -1.5 * t3 + 2 * t2 + 0.5 * t
        f4 = 0.5 * t3 - 0.5 * t2
        x = (P0[0] * f1 + P1[0] * f2 + P2[0] * f3 + P3[0] * f4).astype(int)
        y = (P0[1] * f1 + P1[1] * f2 + P2[1] * f3 + P3[1] * f4).astype(int)
        return list(zip(x, y))

    def smooth_draw(self):
        """Draw smooth curves using Catmull-Rom splines."""
        if len(self.points[-1]) < 4:
            return
        
        smooth_points = []
        for i in range(1, len(self.points[-1]) - 2):
            smooth_points = self.catmull_rom_spline(self.points[-1][i-1], self.points[-1][i], self.points[-1][i+1], self.points[-1][i+2])


        if len(smooth_points) > 1:
            cv2.polylines(self.canvas, [np.array(smooth_points, np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)

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

    def reset_canvas(self):
        """Clear the drawing canvas"""
        self.canvas = np.zeros(self.CANVAS_SHAPE, np.uint8)
        self.points = [[]]

    def save_signiture(self):
        try:
            any_x = np.flatnonzero(np.any(self.canvas, axis=(1,2)))
            any_y = np.flatnonzero(np.any(self.canvas, axis=(0,2)))
            min_x, max_x = any_x[0], any_x[-1]
            min_y, max_y = any_y[0], any_y[-1]
            signiture = self.canvas[min_x:max_x, min_y:max_y, :]
            cv2.imshow(f"Signiture {self.signiture_idx}", signiture)
            np.save(f"signiture_{self.signiture_idx}.npy", signiture)

            self.reset_canvas()
            self.signiture_idx += 1
        except:
            pass

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, landmarks = self.process_frame(frame)

            if landmarks:
                if np.hypot(landmarks[4][1] - landmarks[8][1], landmarks[4][2] - landmarks[8][2]) < 25:
                    # Using index finger tip (landmark id 8)
                    x1, y1 = landmarks[8][1], landmarks[8][2]
                    self.points[-1].append((x1, y1))
                    self.smooth_draw()
                else:
                    self.points.append([])

            frame = cv2.bitwise_or(frame, self.canvas)

            # Display FPS
            curr_time = time.time()
            fps = int(1 / (curr_time - self.past_time))
            self.past_time = curr_time
            cv2.putText(frame, f'FPS: {fps}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            cv2.imshow("PhantomPen", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Clear canvas
                self.reset_canvas()
            elif key == ord('s'):
                self.save_signiture()
            elif key == ord('q'):  # Quit
                break

        cv2.destroyAllWindows()
        self.cap.release()

if __name__ == "__main__":
    app = PhantomPen()
    app.run()