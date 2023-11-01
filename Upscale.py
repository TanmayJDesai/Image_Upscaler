from PIL import Image
import os

def upscaler(source_dir, output_dir):
    target_resolution = (256, 256)

    #Dict maintains map between orig and upscaled images
    image_mapping = {}

    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(source_dir, filename)
            img = Image.open(image_path)
            upscale_img = img.resize(target_resolution, Image.ANTIALIAS)
            output_path = os.path.join(output_dir, filename)
            upscale_img.save(output_path)
            image_mapping[filename] = os.path.relpath(output_path, output_dir)

    return image_mapping

print("done")