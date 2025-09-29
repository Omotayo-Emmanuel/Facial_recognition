import cv2, os, time

# ---------- SETTINGS ----------
PERSON_NAME = "Bello"   # change for each member
# Get path to "Documents" directory in user's home
DOCS_DIR = os.path.join(os.path.expanduser("~"), "Documents")
SAVE_DIR = os.path.join(DOCS_DIR, "faces", PERSON_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_IMAGES = 50
CAM_INDEX = 0  # 0 = default webcam
# ------------------------------

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam. Check CAM_INDEX or permissions.")

count = 0
print(f" Saving images to: {SAVE_DIR}")
print("Press SPACE to capture | Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame. Exiting...")
        break

    # Mirror the camera view (like a selfie)
    frame = cv2.flip(frame, 1)

    # Show instructions on the frame
    text = f"{PERSON_NAME}: {count}/{MAX_IMAGES}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE bar
        filename = os.path.join(SAVE_DIR, f"{PERSON_NAME}_{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f" Saved: {filename}")
        count += 1
        if count >= MAX_IMAGES:
            print(" Reached max images. Exiting...")
            break

    elif key in (ord("q"), ord("Q")):  # Q to quit
        print("Quit requested.")
        break

cap.release()
cv2.destroyAllWindows()
