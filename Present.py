import tensorflow as tf
from Mapping import test_img_paths
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from preprocessing import upscale_factor
import matplotlib.pyplot as plt
import PIL

PSNR = 0.0
Test_Signal_Ratio = 0.0

import PIL.Image

def get_lowres_image_modified(input_image, scaling_factor):
    width, height = input_image.size
    new_width = width // scaling_factor
    new_height = height // scaling_factor

    if new_width % 2 == 0:
        new_width -= 1

    if new_height % 2 == 0:
        new_height -= 1

    return input_image.resize((new_width, new_height), PIL.Image.NEAREST)

import numpy as np
import PIL.Image
from keras.preprocessing.image import img_to_array

def upscale_image_modified(model, input_img):
    ycbcr = input_img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y_array = img_to_array(y)
    y_array = y_array.astype("float32") / 255.0

    if y_array.shape[0] % 2 == 0:
        y_array[0, 0] -= 0.1

    input_data = np.expand_dims(y_array, axis=0)
    predicted_data = model.predict(input_data)

    predicted_y = predicted_data[0]
    predicted_y *= 255.0

    predicted_y = predicted_y.clip(0, 255)
    predicted_y = predicted_y.reshape((np.shape(predicted_y)[0], np.shape(predicted_y)[1]))
    predicted_y_img = PIL.Image.fromarray(np.uint8(predicted_y), mode="L")

    cb_resized = cb.resize(predicted_y_img.size, PIL.Image.BICUBIC)
    cr_resized = cr.resize(predicted_y_img.size, PIL.Image.BICUBIC)

    # Merge the YCbCr channels and convert back to RGB
    final_image = PIL.Image.merge("YCbCr", (predicted_y_img, cb_resized, cr_resized)).convert("RGB")

    return final_image


def plot_results_modified(image, file_prefix, plot_title):
    image_array = img_to_array(image)
    image_array = image_array.astype("float32") / 255.0

    fig, ax = plt.subplots()
    im = ax.imshow(image_array[::-1], origin="lower")

    plt.title(plot_title)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    plt.savefig(str(file_prefix) + "-" + plot_title + ".png")
    plt.show()


from keras import Input, Model, layers


def get_model(upscale_factor=3, channels=1):
    conv_args = {"activation": "relu", "kernel_initializer": "Orthogonal", "padding": "same"}

    inputs = Input(shape=(None, None, channels))
    x = inputs

    for filters in [64, 64, 32, channels * (upscale_factor ** 2)]:
        x = layers.Conv2D(filters, 3, **conv_args)(x)

    outputs = layers.Lambda(lambda x: tf.nn.depth_to_space(x, upscale_factor))(x)

    return Model(inputs, outputs)


model = get_model(upscale_factor=upscale_factor, channels=1)

def process_and_plot_images(model, img_paths, start, end):
    PSNR, Test_Signal_Ratio = 0, 0

    for index, test_img_path in enumerate(img_paths[start:end]):
        img = load_img(test_img_path)
        lowres_input = get_lowres_image_modified(img, upscale_factor)
        highres_img = img.resize((w := lowres_input.size[0] * upscale_factor, h := lowres_input.size[1] * upscale_factor))
        prediction = upscale_image_modified(model, lowres_input)
        lowres_img = lowres_input.resize((w, h))


        PSNR += tf.image.psnr(img_to_array(lowres_img), img_to_array(highres_img), max_val=255)
        Test_Signal_Ratio += tf.image.psnr(img_to_array(prediction), img_to_array(highres_img), max_val=255)

        for i, img in enumerate([lowres_img, prediction, highres_img]):
            plot_results_modified(img, index, ["BAD IMAGE", "Digital MPEG compression (YcBcR Color Family)", "IMAGE AFTER MODEL"][i])

    return PSNR, Test_Signal_Ratio

start, end = 50, 60
PSNR, Test_Signal_Ratio = process_and_plot_images(model, test_img_paths, start, end)


print("Avg. Peak Signal to Noise Ratio (LOW RESOLUTION) %.4f" % (PSNR / 10))
print("Avg. Peak Signal to Noise Ratio (FINAL IMAGE) %.4f" % (Test_Signal_Ratio / 10))