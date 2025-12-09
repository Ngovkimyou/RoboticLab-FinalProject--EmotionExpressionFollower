# convert videos to frames
import cv2, os

# ===== CONFIG =====
emotion = "neutral"                       # change for each emotion
video_file = f"extra_videos/{emotion}.mp4"
frame_skip = 5                            # extract 1 frame every 5 frames
max_frames = 1000                         # avoid too many frames
# ===================

# Final training folder (NO NEED TO MOVE MANUALLY)
output_folder = f"data/train/{emotion}"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_file}")

idx = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save only every frame_skip-th frame
    if idx % frame_skip == 0:
        out_path = os.path.join(output_folder, f"{emotion}_{saved:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1

        if saved >= max_frames:
            print("Reached max_frames limit. Stopping.")
            break

    idx += 1

cap.release()
print(f"Saved {saved} frames for emotion: {emotion}")
print("Frames saved into:", output_folder)
