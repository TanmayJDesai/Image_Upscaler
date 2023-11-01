from PIL import Image
import os

def downscaler(source_dir, output_dir):
    #The shitty pics are 128 128 which im pairing with 256 256 and then ill make my model take the 128 test images and try to make them 256 256
    target_resolution = (128, 128)

    #Dict maintains map between orig and downscaled images
    image_mapping = {}

    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(source_dir, filename)
            img = Image.open(image_path)
            downscale_img = img.resize(target_resolution, Image.ANTIALIAS)
            output_path = os.path.join(output_dir, filename)
            downscale_img.save(output_path)
            image_mapping[filename] = os.path.relpath(output_path, output_dir)

    return image_mapping

print("done")