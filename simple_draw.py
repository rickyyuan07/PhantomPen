import os
import cv2
import mediapipe as mp
import numpy as np
import time
import argparse

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

        self.style = args.style
        # Define colors for different styles
        colors = {
            "glow": (0, 255, 0),
            "neon_blue": (255, 0, 255),
            "fire": (0, 165, 255)
        }
        self.color = colors.get(self.style, (0, 255, 0))

        self.canvas = np.zeros(self.CANVAS_SHAPE, np.uint8)  # Canvas to write on (with effects)
        self.signature = np.zeros(self.CANVAS_SHAPE, np.uint8)  # Canvas to store signatures (pure black and white)
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
        """Draw smooth curves using Catmull-Rom splines with different stroke styles."""
        if len(self.points[-1]) < 4:
            return
        
        # Calculate Catmull-Rom spline only for the last 4 points
        smooth_points = self.catmull_rom_spline(*self.points[-1][-4:])

        temp_canvas = np.zeros_like(self.canvas)  # Create a temporary canvas

        cv2.polylines(temp_canvas, [np.array(smooth_points, np.int32)], isClosed=False, color=self.color, thickness=2)

        # Apply different glow effects
        if self.style in ["glow", "neon_blue", "fire"]:
            blur_amount = 15 if self.style == "glow" else 25  # More blur for neon
            glow = cv2.GaussianBlur(temp_canvas, (blur_amount, blur_amount), blur_amount)

            # Increase intensity for different effects
            glow_intensity = 1.3 if self.style == "glow" else 1.7
            self.canvas = cv2.addWeighted(self.canvas, 1.0, glow, glow_intensity, 0)

        
        # Draw the original strokes again for sharper effect
        cv2.polylines(self.canvas, [np.array(smooth_points, np.int32)], isClosed=False, color=self.color, thickness=2)

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
            ## DEBUG, store self.canvas
            # cv2.imwrite(f"signatures/{self.args.name}_{self.args.signature_idx}.png", self.canvas)
            ## DEBUG

            # Identify non-black pixel regions
            any_x = np.flatnonzero(np.any(self.canvas, axis=(1, 2)))
            any_y = np.flatnonzero(np.any(self.canvas, axis=(0, 2)))

            # Ensure there is a valid drawing before processing
            if any_x.size == 0 or any_y.size == 0:
                print("No signature found to save.")
                return

            # Crop to bounding box of the drawing
            min_x, max_x = any_x[0], any_x[-1] + 1
            min_y, max_y = any_y[0], any_y[-1] + 1
            signature = self.canvas[min_x:max_x, min_y:max_y, :]

            # Convert green pixels (drawing color) to white
            mask = np.all(signature == self.color, axis=-1)  # Identify green pixels
            signature[mask] = [255, 255, 255]  # Change green to white

            # Display the signature preview
            cv2.imshow(f"Signature - {self.args.name} {self.args.signature_idx}", signature)

            # Create the directory if it doesn't exist
            user_signature_dir = os.path.join(self.args.signature_dir, self.args.name)
            os.makedirs(user_signature_dir, exist_ok=True)

            # Save as .npy file
            npy_path = os.path.join(user_signature_dir, f"{self.args.signature_idx}.npy")
            np.save(npy_path, signature)

            # Convert to RGBA for transparency
            image_rgba = cv2.cvtColor(signature, cv2.COLOR_BGR2RGBA)
            image_rgba[mask] = [255, 255, 255, 255]  # White for the signature
            image_rgba[~mask] = [0, 0, 0, 0]  # Transparent background

            # Save as PNG with transparency
            png_path = os.path.join(user_signature_dir, f"{self.args.signature_idx}.png")
            cv2.imwrite(png_path, image_rgba)

            # Reset the canvas and increment index
            self.reset_canvas()
            self.args.signature_idx += 1

            print(f"Signature saved at {png_path}")

        except Exception as e:
            print(f"Error saving signature: {e}")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, landmarks = self.process_frame(frame)

            if landmarks:
                finger, thumb = landmarks[8], landmarks[4]
                # if the user is pinching the thumb and index finger
                if np.hypot(thumb[1] - finger[1], thumb[2] - finger[2]) < 18:
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
    parser.add_argument("-s", "--signature_dir", type=str, default="signatures", help="Directory to store signatures")
    parser.add_argument("-st", "--style", type=str, choices=["glow", "neon_blue", "fire"], default="glow", help="Drawing style")
    args = parser.parse_args()
    
    user_dir = os.path.join(args.signature_dir, args.name)
    os.makedirs(user_dir, exist_ok=True)  # Create directory if it doesn't exist

    files = [f for f in os.listdir(user_dir) if f.endswith(".npy")]
    args.signature_idx = 0
    if files:
        args.signature_idx = max([int(os.path.splitext(f)[0]) for f in files]) + 1
    print(args.signature_idx)

    app = PhantomPen(args)
    app.run()
