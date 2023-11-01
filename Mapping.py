import os

source_dir = "/Users/tanmaydesai/Desktop/TrainingData"
downscaled_output_dir = "/Users/tanmaydesai/Desktop/DownscaledImages"

#get names
image_filenames = [filename for filename in os.listdir(source_dir) if filename.endswith(".jpg")]

#pair downscale with normal nice images
image_pairing = {}
for filename in image_filenames:
    high_res_path = os.path.join(source_dir, filename)
    downscaled_path = os.path.join(downscaled_output_dir, filename)
    image_pairing[high_res_path] = downscaled_path

print("Pairings for the top 10 images:")
for i in range(10):
    high_res_path = list(image_pairing.keys())[i]
    downscaled_path = image_pairing[high_res_path]
    print(f"High-Res: {high_res_path}, Downscaled: {downscaled_path}")

print("done")
