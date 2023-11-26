'''FINAL MODEL BUILT HERE'''
import os
import glob

files = glob.glob('/Users/tanmaydesai/Desktop/TrainingData')
for f in files:
    os.remove(f)
