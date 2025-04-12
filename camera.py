import cv2
import time
import numpy as np
from inference_sdk import InferenceHTTPClient
import traceback # Import traceback for more detailed error info

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="gpm0fC7gaKhUgWSdHlfQ"  # Replace with your actual API key
)

MODEL_ID = "parking-lot-fmzu9/1" # Make sure this model exists and is deployed
SLOT_COUNT = 8 # Adjust if your model detects more/less than 8 slots total
OUTPUT_FILE = "static/slot_status.txt"

# Connect to Iriun webcam (usually index 1, change if needed)
# Try index 0 or 2 if 1 doesn't work
cap = cv2.VideoCapture(1) # Or 0, or 2 etc.
if not cap.isOpened():
    print("Error: Could not access Iriun webcam. Tried index 1.")
    # Optional: Try other indices
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Error: Could not access webcam at index 0 either.")
    #     exit()
    # print("✨ Connected to webcam at index 0.")
    exit()


print("✨ Iriun Webcam connected successfully.")

def write_status_to_file(occupied_slots):
    """ Write slot status to text file for JS to fetch """
    # Ensure occupied_slots doesn't exceed SLOT_COUNT
    occupied_slots = min(occupied_slots, SLOT_COUNT)
    status_line = ",".join(["1" if i < occupied_slots else "0" for i in range(SLOT_COUNT)])
    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write(status_line)
    except IOError as e:
        print(f"🚨 Error writing to status file {OUTPUT_FILE}: {e}")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame 💔")
        time.sleep(1) # Wait a bit before retrying
        continue # Try to grab the next frame

    # Resize for better inference performance and consistency
    # Using a standard size like 640x640 might be beneficial if your model was trained on that
    resized = cv2.resize(frame, (640, 480)) # Keep 640x480 or change to 640x640 if needed

    # --- No need to convert color or encode manually ---
    # rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) # Not needed if passing BGR array
    # is_success, img_encoded = cv2.imencode('.jpg', rgb_frame) # Not needed

    try:
        # Inference: Pass the resized frame (NumPy array) directly
        # The SDK client usually handles BGR/RGB conversion and encoding if needed by the API
        print("🚀 Sending frame for inference...")
        results = CLIENT.infer(resized, model_id=MODEL_ID) # *** CHANGE IS HERE ***

        # Parse detection result
        # Check if results is a dict and has 'predictions'
        if isinstance(results, dict) and "predictions" in results:
            detections = results.get("predictions", [])
            occupied_slots = len(detections) # Assuming each detection is one occupied slot
            print(f"🟡 Detected {occupied_slots} occupied slot(s).")
            write_status_to_file(occupied_slots)
        else:
             # Log the unexpected result structure
             print(f"🚨 Unexpected inference result format: {results}")
             # Optionally write a default state (e.g., all empty) or skip writing
             # write_status_to_file(0)


    except Exception as e:
        print(f"🚨 Error running inference: {e}")
        # Print detailed traceback for debugging
        print("--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---")
        # Optional: Add a small delay after an error before retrying
        time.sleep(2)


    # Display live video (Optional but helpful for debugging)
    # Draw bounding boxes if you want to visualize detections
    if isinstance(results, dict) and "predictions" in results:
         for detection in results["predictions"]:
              x = int(detection['x'])
              y = int(detection['y'])
              width = int(detection['width'])
              height = int(detection['height'])
              confidence = detection['confidence']
              class_name = detection['class']

              # Calculate top-left and bottom-right corners
              x1 = int(x - width / 2)
              y1 = int(y - height / 2)
              x2 = int(x + width / 2)
              y2 = int(y + height / 2)

              # Draw rectangle and label
              cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
              label = f"{class_name}: {confidence:.2f}"
              cv2.putText(resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    cv2.imshow("Live Feed - AI Parking System", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting live feed...")
        break

    # Adjust sleep time as needed - balances responsiveness and API usage
    time.sleep(2)  # Poll every 2 seconds

cap.release()
cv2.destroyAllWindows()
print("✅ Script finished.")