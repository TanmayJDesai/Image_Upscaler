import os
from Downscale-Images import downscaler
from Upscale-Images import upscaler

# Set the source directory for your images
source_dir = "/Users/tanmaydesai/Desktop/TrainingData"


# Set the output directories for downscaled and upscaled images
downscaled_output_dir = "/Users/tanmaydesai/Desktop/DownscaledImages"
upscaled_output_dir = "/Users/tanmaydesai/Desktop/UpscalesImages"

# Perform downscaling and upscaling and get the mappings
downscale_mapping = downscale_to_128x128(source_dir, downscaled_output_dir)
upscale_mapping = upscale_to_256x256(source_dir, upscaled_output_dir)

# Create a dictionary that pairs downscaled images with their respective upscaled images
image_pairing = {}
for filename in downscale_mapping:
    downscaled_path = os.path.join(downscaled_output_dir, downscale_mapping[filename])
    upscaled_path = os.path.join(upscaled_output_dir, upscale_mapping[filename])
    image_pairing[downscaled_path] = upscaled_path

# image_pairing now contains the mapping of downscaled images to their respective upscaled images
