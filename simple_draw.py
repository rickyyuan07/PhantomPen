import os
import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import pdb

class PhantomPen:
    FRAME_WIDTH, FRAME_HEIGHT = 640, 480
    CANVAS_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)

    def __init__(self, args):
        self.args = args
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
        self.past_time = time.time()  # For FPS calculation

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
        # Calculate Catmull-Rom spline only for the last 4 points
        smooth_points = self.catmull_rom_spline(*self.points[-1][-4:])

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

    def save_signature(self,):
        try:
            any_x = np.flatnonzero(np.any(self.canvas, axis=(1,2)))
            any_y = np.flatnonzero(np.any(self.canvas, axis=(0,2)))
            min_x, max_x = any_x[0], any_x[-1]
            min_y, max_y = any_y[0], any_y[-1]
            signature = self.canvas[min_x:max_x, min_y:max_y, :]
            cv2.imshow(f"{self.args.name} {self.args.signature_idx}", signature)
            signature_dir = os.path.join(self.args.signature_dir, self.args.name, f"{self.args.signature_idx}.npy")
            np.save(signature_dir, signature)

            # Convert the image to RGBA format
            image_rgba = cv2.cvtColor(signature, cv2.COLOR_BGR2RGBA)

            # Step 3: Identify black background pixels (BGR = (0, 0, 0))
            black_mask = (image_rgba[:, :, 0] == 0) & (image_rgba[:, :, 1] == 0) & (image_rgba[:, :, 2] == 0)

            # Step 4: Set black background pixels to transparent
            image_rgba[black_mask] = [0, 0, 0, 0]

            # Step 1: Identify green pixels (exact match: BGR = (0, 255, 0))
            green_mask = (signature[:, :, 0] == 0) & (signature[:, :, 1] == 255) & (signature[:, :, 2] == 0)

            # Step 2: Replace green pixels with black (fully opaque)
            image_rgba[green_mask] = [0, 0, 0, 255]

            # Save the resulting image as a PNG with transparency
            signature_dir2 = os.path.join(self.args.signature_dir, self.args.name, f"{self.args.signature_idx}.png")

            cv2.imwrite(signature_dir2, image_rgba)

            self.reset_canvas()
            self.args.signature_idx += 1
        except:
            pass

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, landmarks = self.process_frame(frame)

            if landmarks:
                finger, thumb = landmarks[8], landmarks[4]
                # if the user is pinching the thumb and index finger
                if np.hypot(thumb[1] - finger[1], thumb[2] - finger[2]) < 20:
                    # Using the mean distance between the thumb and index finger
                    x1, y1 = (thumb[1] + finger[1]) // 2, (thumb[2] + finger[2]) // 2
                    self.points[-1].append((x1, y1))
                    self.smooth_draw()
                elif self.points[-1]:
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
                self.save_signature()
            elif key == ord('q'):  # Quit
                break

        cv2.destroyAllWindows()
        self.cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple draw & signature collection app")
    parser.add_argument("-n", "--name", type=str, default="user", help="the name of the user")
    parser.add_argument("--signature_dir", type=str, default="signatures", help="Directory to store signatures")
    args = parser.parse_args()
    
    user_dir = os.path.join(args.signature_dir, args.name)
    os.makedirs(user_dir, exist_ok=True)  # Create directory if it doesn't exist

    files = os.listdir(user_dir)
    args.signature_idx = 0
    if files:
        args.signature_idx = int(os.path.splitext(max(files))[0]) + 1
    print(args.signature_idx)

    app = PhantomPen(args)
    app.run()
