import os
import shutil
import random
import stat

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise"]

# ============ SAFE DELETE FUNCTION (Fix WinError 5) ============
def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# ============ STEP 1 — CLEAN TEST DIRECTORY ============

print("Cleaning test folder...")

for emo in EMOTIONS:
    folder = os.path.join(TEST_DIR, emo)
    if os.path.exists(folder):
        shutil.rmtree(folder, onerror=remove_readonly)
    os.makedirs(folder, exist_ok=True)

print("✔ Test folder cleaned")

# ============ STEP 2 — ENSURE TRAIN FOLDERS EXIST ============

for emo in EMOTIONS:
    os.makedirs(os.path.join(TRAIN_DIR, emo), exist_ok=True)

# ============ STEP 3 — SPLIT DATA ============

print("Splitting dataset (80% train / 20% test)...")

for emo in EMOTIONS:
    train_folder = os.path.join(TRAIN_DIR, emo)
    test_folder = os.path.join(TEST_DIR, emo)

    images = [f for f in os.listdir(train_folder)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if len(images) == 0:
        print(f"⚠ No images found for {emo}. Skipping...")
        continue

    random.shuffle(images)

    test_count = max(1, int(len(images) * 0.20))

    test_images = images[:test_count]

    for img in test_images:
        src = os.path.join(train_folder, img)
        dst = os.path.join(test_folder, img)
        shutil.move(src, dst)

    print(f"{emo}: {len(images)} images → {test_count} moved to test")

print("\n🎉 DONE! Dataset split into train/test successfully.")
