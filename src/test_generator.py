# This script tests the data generators for training and testing datasets.
from data_pipeline import get_data_generators
import os

train_path = os.path.join("data", "train")
test_path  = os.path.join("data", "test")

train_gen, test_gen, train_count, test_count = get_data_generators(train_path, test_path)

print("Train samples:", train_count)
print("Test samples:", test_count)

batch_x, batch_y = next(train_gen)
print("Batch X shape:", batch_x.shape)
print("Batch Y shape:", batch_y.shape)
