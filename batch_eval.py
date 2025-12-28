import os
import zipfile
import cv2
import pandas as pd
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== 1. Configuration & Environment Setup =====
ZIP_FILE = '/content/testing_dataset.zip'   # Your uploaded zip file name
EXTRACT_TO = '/content/final_data'          # Destination folder for extraction
MODEL_PATH = '/content/gesture_recognizer.task' # Your model file name

# Label Mapping Table (Folder Name : Model Output Label)
# This ensures "1" is correctly matched with "gesture_1" for accuracy calculation
label_map = {
    "1": "gesture_1",
    "2": "gesture_2",
    "none": "none"
}

# ===== 2. Extraction Logic =====
if os.path.exists(ZIP_FILE):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_TO)
    time.sleep(1) # Wait for system file index to update
    print("Extraction Complete!")
else:
    print(f"Error: Cannot find {ZIP_FILE}. Please ensure the file is uploaded.")

# Automatically locate the actual data folder (handles nested zip structures)
def get_actual_path(base):
    for root, dirs, files in os.walk(base):
        if any(d in ['1', '2', 'none'] for d in dirs):
            return root
    return base

DATA_PATH = get_actual_path(EXTRACT_TO)
print(f"Data Path Locked: {DATA_PATH}")

# ===== 3. Initialize Gesture Recognizer =====
if not os.path.exists(MODEL_PATH):
    print(f"Error: Cannot find model file at {MODEL_PATH}")
else:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # ===== 4. Batch Inference =====
    all_results = []
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')

    print("\nStarting Batch Recognition...")
    for label in label_map.keys():
        folder_path = os.path.join(DATA_PATH, label)
        if not os.path.exists(folder_path):
            print(f"Skipping folder [{label}]: Path does not exist")
            continue
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
        print(f"Processing: Category [{label}], {len(files)} images found")

        for img_name in files:
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue

            # Image conversion and recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            prediction = recognizer.recognize(mp_image)

            # Extract prediction label
            pred_label = "none"
            score = 0.0
            if prediction.gestures:
                top = prediction.gestures[0][0]
                # If result is not empty/none, record the category name
                if top.category_name and top.category_name.lower() not in ["none", "unknown", ""]:
                    pred_label = top.category_name
                score = top.score

            all_results.append({
                "Actual_Category": label,
                "Model_Prediction": pred_label,
                "Confidence": score
            })

    # ===== 5. Statistics & Report Output =====
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Accuracy validation using the label_map
        def check_correct(row):
            expected = label_map.get(str(row['Actual_Category']))
            return "Correct" if expected == str(row['Model_Prediction']) else "Wrong"
        
        df['Result'] = df.apply(check_correct, axis=1)

        print("\n" + "="*50)
        print("Confusion Matrix (Actual vs Predicted)")
        print("="*50)
        # Display the confusion matrix
        summary = df.groupby(['Actual_Category', 'Model_Prediction']).size().unstack(fill_value=0)
        print(summary)
        
        print("-" * 50)
        correct_count = (df['Result'] == "Correct").sum()
        total_count = len(df)
        accuracy = (correct_count / total_count) * 100
        
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"Summary: {correct_count} Correct / {total_count} Total")
        print("="*50)
        
        # Optional: Export detailed results to CSV
        df.to_csv('/content/evaluation_report.csv', index=False, encoding='utf-8-sig')
        print("Detailed report saved to: /content/evaluation_report.csv")
    else:
        print("Execution Failed: No recognition results found.")

    recognizer.close()
