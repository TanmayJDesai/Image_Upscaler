import os
from preprocessing import train_ds, valid_ds

#just get it to a 0,1 pixel normalization
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image

dataset = "/Users/tanmaydesai/Desktop/IMAGES"

test_img_paths = sorted(
    [
        os.path.join(dataset, fname)
        for fname in os.listdir(dataset)
        if fname.endswith(".jpg")
    ]
)

training_set = train_ds.map(scaling)
valid_set = valid_ds.map(scaling)