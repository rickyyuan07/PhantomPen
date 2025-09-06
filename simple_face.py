import cv2
import face_recognition
import numpy as np
import json
import os
import subprocess
import tempfile
import time
import configparser

known_face_encodings = []
known_face_names = []

def load_face_data():
    global known_face_encodings, known_face_names
    try:
        if os.path.exists('face_data.json'):
            with open('face_data.json', 'r') as f:
                data = json.load(f)
                known_face_names = data['names']
                known_face_encodings = [np.array(enc) for enc in data['encodings']]
    except Exception as e:
        print(f"Error loading data: {e}")

def save_face_data():
    try:
        data = {
            'names': known_face_names,
            'encodings': [enc.tolist() for enc in known_face_encodings]
        }
        with open('face_data.json', 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving data: {e}")

def delete_face(name):
    try:
        global known_face_encodings, known_face_names
        if name in known_face_names:
            index = known_face_names.index(name)
            del known_face_names[index]
            del known_face_encodings[index]
            save_face_data()

            image_path = os.path.join('face_images_data', f"{name}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)

            print(f"Successfully deleted face data for {name}")
        else:
            print(f"No face data found for {name}")
    except Exception as e:
        print(f"Error deleting face data: {e}")

def detected_face(target_name: str, cap, timeout: int = 10) -> bool:
    """
        持續從攝影機讀取影像，直到 timeout 秒內是否辨識到指定名稱的人臉。
        :param target_name: 欲偵測的已知人名（必須已存在於資料庫中）
        :param timeout: 最多偵測幾秒
        :return: True 表示成功辨識到目標人臉；False 表示未偵測到
    """
    load_face_data()
    if target_name not in known_face_names:
        print(f"'{target_name}' not in known face database.")
        return False

    start_time = time.time()
    print("Starting face verification...")

    print("Instructions:")
    print("  - Press 'q': Quit the program.")

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            if True in matches:
                matched_index = matches.index(True)
                name = known_face_names[matched_index]

                if name == target_name:
                    # 標記並顯示
                    # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    # cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    # cv2.imshow("Face Verification", frame)
                    print(f"[✅] Matched: {name}")
                    return True

            # 標記未知人臉
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Face Verification", frame)

        # 使用者按下 'q' 離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[❌] Face verification failed.")
    return False

if __name__ == "__main__":
    load_face_data()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Connect to the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open the camera")
        exit(1)

    add_face_mode = False
    new_face_name = None
    recognized_names = set()

    print("Instructions:")
    print("  - Press 'n': Enter 'Add Face' mode. The next detected face will be added to the database.")
    print("  - Press 'd': Delete face data. You will be prompted to enter a name.")
    print("  - Press 'q': Quit the program.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read camera frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # If in "Add Face" mode
        if add_face_mode:
            if face_encodings:
                face_encoding = face_encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(new_face_name)
                save_face_data()

                if not os.path.exists('face_images_data'):
                    os.makedirs('face_images_data')
                face_image_path = os.path.join('face_images_data', f"{new_face_name}.jpg")
                cv2.imwrite(face_image_path, frame)

                print(f"Successfully added face data for {new_face_name}")
            else:
                print("No face detected, please try again")

            add_face_mode = False
            new_face_name = None

        else:
            # General mode: Attempt recognition
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    recognized_names.add(name)

                # Draw rectangle around the face
                marking_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), marking_color, 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, marking_color, 2)

        cv2.imshow("Face Recognition - Press q to quit", frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            new_face_name = input("Enter name to add: ")
            if new_face_name.strip():
                add_face_mode = True
            else:
                print("Name cannot be empty. Add face cancelled.")
        elif key == ord('d'):
            del_name = input("Enter name to delete: ")
            delete_face(del_name)

    print("End of rolling attendance")
    cap.release()
    cv2.destroyAllWindows()