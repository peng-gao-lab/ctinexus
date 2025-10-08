import os
import random
import shutil

data_dir = "data/annotation"


def demo_test_split(test_size=10):
	"""Split the dataset into demo and test sets."""

	parent_dir = os.path.dirname(data_dir)
	demo_dir = os.path.join(parent_dir, "demo")
	test_dir = os.path.join(parent_dir, "test")

	# Create directories if they don't exist
	os.makedirs(demo_dir, exist_ok=True)
	os.makedirs(test_dir, exist_ok=True)

	# Get all JSON files in the data directory
	all_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

	# Shuffle the files
	random.shuffle(all_files)

	# Split the files into demo and test sets
	demo_files = all_files[:-test_size]
	test_files = all_files[-test_size:]

	# Copy files to the respective directories
	for file in demo_files:
		src = os.path.join(data_dir, file)
		dst = os.path.join(demo_dir, file)
		shutil.copy(src, dst)

	for file in test_files:
		src = os.path.join(data_dir, file)
		dst = os.path.join(test_dir, file)
		shutil.copy(src, dst)


if __name__ == "__main__":
	demo_test_split()
