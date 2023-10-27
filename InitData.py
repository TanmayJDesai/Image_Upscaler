import os
import shutil
import random

#ALL IMAGES LOCATED HERE
source_dir = "/Users/tanmaydesai/Desktop/IMAGES"

'''
I am loading an 80/20 split for train/test 
    - traindir and testdir are two directories with paths to folders on my desktop 
      where I will be loading in my data based on the split.
'''
train_dir = "/Users/tanmaydesai/Desktop/TrainingData"
test_dir = "/Users/tanmaydesai/Desktop/TestingData"

#Rewriting the split ratio
split_ratio = 0.8

#Ensure we can work with all data
image_files = [f for f in os.listdir(source_dir) if f.endswith(".jpg")]

#NO BIAS HERE BOISS
random.shuffle(image_files)

#Based on the split ratio decide the exact number of images in each set
split_index = int(len(image_files) * split_ratio)

# Split into training and testing sets
train_files = image_files[:split_index]
test_files = image_files[split_index:]

#Move to appropriate directories
for file in train_files:
    src_file = os.path.join(source_dir, file)
    dest_file = os.path.join(train_dir, file)
    shutil.copy(src_file, dest_file)

for file in test_files:
    src_file = os.path.join(source_dir, file)
    dest_file = os.path.join(test_dir, file)
    shutil.copy(src_file, dest_file)

print ("done")