import os

def count_images_in_directory(directory):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    image_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(image_extensions):
            image_count += 1

    return image_count

def checkImagesinSource():
    directory_path = "/Users/tanmaydesai/Desktop/IMAGES"
    num_images = count_images_in_directory(directory_path)
    return num_images

def checkImagesinTraining():
    directory_path = "/Users/tanmaydesai/Desktop/TrainingData"
    num_images = count_images_in_directory(directory_path)
    return num_images

def checkImagesinTesting():
    directory_path = "/Users/tanmaydesai/Desktop/TestingData"
    num_images = count_images_in_directory(directory_path)
    return num_images

def checkImagesinDownscale():
    directory_path = "/Users/tanmaydesai/Desktop/DownscaledImages"
    num_images = count_images_in_directory(directory_path)
    return num_images

source_images_count = checkImagesinSource()
training_images_count = checkImagesinTraining()
testing_images_count = checkImagesinTesting()
downscale_images_count = checkImagesinDownscale()

print(f"Source Images Count: {source_images_count}")
print(f"Training Images Count: {training_images_count}")
print(f"Testing Images Count: {testing_images_count}")
print(f"Downscaled Images Count: {downscale_images_count}")
