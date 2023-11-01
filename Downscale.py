from PIL import Image
import os

def downscaler(source_dir, output_dir):
    # The target resolution is (128, 128)
    target_resolution = (128, 128)

    # Dictionary maintains the mapping between original and downscaled images
    image_mapping = {}

    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(source_dir, filename)
            img = Image.open(image_path)
            downscale_img = img.resize(target_resolution, Image.BILINEAR)
            output_path = os.path.join(output_dir, filename)
            downscale_img.save(output_path)
            image_mapping[filename] = os.path.relpath(output_path, output_dir)

    return image_mapping


source_dir = "/Users/tanmaydesai/Desktop/TrainingData"
downscaled_output_dir = "/Users/tanmaydesai/Desktop/DownscaledImages"
downscale_mapping = downscaler(source_dir, downscaled_output_dir)

print("done")
