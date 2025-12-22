import cv2
import mediapipe as mp
import time
import os
import datetime # Used for formatting timestamp

# ===== Configuration =====
# Input the gesture names you want to detect (must match folder names)
TARGET_GESTURES = ["gesture_1", "gesture_2", "1", "2"] 

# Interval for logging data to terminal (in seconds)
LOG_INTERVAL = 5 

# Manually define hand connections (Skeleton lines)
# This avoids "mp.solutions" attribute errors
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),     # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),# Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
])

# ===== Load Model (Buffer method for path safety) =====
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'gesture_recognizer.task')

print(f"Loading model from: {model_path}")

try:
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    print("Model loaded successfully into memory.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# ===== Initialize Gesture Recognizer =====
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_buffer=model_buffer),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

recognizer = GestureRecognizer.create_from_options(options)

# ===== Custom Drawing Function =====
def draw_landmarks_manually(image, landmarks):
    height, width, _ = image.shape
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for lm in landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        points.append((px, py))
    
    # Draw connections (White lines)
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(image, points[start_idx], points[end_idx], (255, 255, 255), 2)
        
    # Draw joints (Red dots)
    for point in points:
        cv2.circle(image, point, 4, (0, 0, 255), -1)

# ===== Main Loop =====
cap = cv2.VideoCapture(0)
print(f"\nCamera starting... \n   - Skeleton: Always visible\n   - Target Gestures {TARGET_GESTURES}: Green Box\n   - Data Logging: Every {LOG_INTERVAL} seconds")

# Timer initialization
last_log_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    height, width, _ = frame.shape

    # Run recognition
    frame_timestamp_ms = int(time.time() * 1000)
    result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)

    # Variables for current frame status
    display_text = "None"
    status_color = (0, 0, 255) # Red (Default)
    current_log_gesture = "None"
    current_log_score = 0.0

    # If hands are detected
    if result.hand_landmarks:
        for idx in range(len(result.hand_landmarks)):
            hand_landmarks = result.hand_landmarks[idx]
            
            # Draw skeleton
            draw_landmarks_manually(frame, hand_landmarks)

            # Check gesture
            if result.gestures and len(result.gestures) > idx:
                gesture_category = result.gestures[idx][0].category_name
                score = result.gestures[idx][0].score
                
                # Update log variables (take the first detected hand as primary)
                if idx == 0:
                    current_log_gesture = gesture_category
                    current_log_score = score

                # Logic for Target Gestures (Green Box)
                if gesture_category in TARGET_GESTURES and score > 0.5:
                    display_text = f"{gesture_category} ({int(score*100)}%)"
                    status_color = (0, 255, 0) # Green

                    # Calculate Bounding Box
                    x_list = [lm.x for lm in hand_landmarks]
                    y_list = [lm.y for lm in hand_landmarks]
                    
                    x_min = int(min(x_list) * width) - 20
                    y_min = int(min(y_list) * height) - 20
                    x_max = int(max(x_list) * width) + 20
                    y_max = int(max(y_list) * height)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    cv2.putText(frame, display_text, (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Display Status on top-left
    cv2.putText(frame, f"Status: {display_text.split(' ')[0]}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
    
    cv2.imshow('Gesture Recognition System', frame)

    # =====Data Logging Logic (Every 5 Seconds)=====
    if time.time() - last_log_time >= LOG_INTERVAL:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to terminal
        print(f"[{timestamp}] Gesture: {current_log_gesture} | Score: {current_log_score:.4f}")
        
        # Reset timer
        last_log_time = time.time()

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
recognizer.close()