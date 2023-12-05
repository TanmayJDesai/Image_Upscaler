import tensorflow as tf
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, array_to_img, img_to_array
from IPython.display import display
from Mapping import training_set, valid_set
from preprocessing import input_size, upscale_factor

Final_Paths = sorted(
    [
        os.path.join("/Users/tanmaydesai/Desktop/IMAGES", fname)
        for fname in os.listdir("/Users/tanmaydesai/Desktop/IMAGES")
        if fname.endswith(".jpg")
    ]
)

def process_input_img(INPUT, SIZE):
    yuv_image = tf.image.rgb_to_yuv(INPUT)

    last_dimension_axis = len(yuv_image.shape) - 1
    y, u, v = tf.split(yuv_image, 3, axis=last_dimension_axis)

    resized_y = tf.image.resize(y, [SIZE, SIZE], method="area")

    return resized_y

def process_target_img(INPUT):
    yuv_image = tf.image.rgb_to_yuv(INPUT)
    last_dimension_axis = len(yuv_image.shape) - 1
    y, u, v = tf.split(yuv_image, 3, axis=last_dimension_axis)
    return y

training_set = training_set.map(
    lambda x: (process_input_img(x, input_size), process_target_img(x))
)
training_set = training_set.prefetch(buffer_size=32)

valid_set = valid_set.map(
    lambda x: (process_input_img(x, input_size), process_target_img(x))
)
valid_set = valid_set.prefetch(buffer_size=32)

for batch in training_set.take(1):
    for img in batch[0]:
        display(array_to_img(img))
    for img in batch[1]:
        display(array_to_img(img))


def custom_upscale_model(factor=3, num_channels=1):
    # Define convolutional layer arguments
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }

    inputs = keras.Input(shape=(None, None, num_channels))
    conv1 = layers.Conv2D(64, 5, **conv_args)(inputs)
    conv2 = layers.Conv2D(64, 3, **conv_args)(conv1)
    conv3 = layers.Conv2D(32, 3, **conv_args)(conv2)
    conv4 = layers.Conv2D(num_channels * (factor ** 2), 3, **conv_args)(conv3)

    outputs = tf.nn.depth_to_space(conv4, factor)

    model = keras.Model(inputs, outputs)
    return model

def display_enhanced_results(img, prefix, title):
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.title(title)
    plt.yticks(visible=False)
    plt.xticks(visible=False)

    plt.savefig(str(prefix) + "-" + title + ".png")
    plt.show()


def generate_lowres_img(img, factor):
    return img.resize(
        (img.size[0] // factor, img.size[1] // factor),
        PIL.Image.BICUBIC,
    )

def predict_enhanced_image(model, img):
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input_img = np.expand_dims(y, axis=0)
    out = model.predict(input_img)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img

class PSNRMonitor(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.test_img = generate_lowres_img(load_img(Final_Paths[0]), upscale_factor)

    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 20 == 0:
            prediction = predict_enhanced_image(self.model, self.test_img)
            display_enhanced_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))

def custom_upscale_model(factor=3, num_channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }

    # Input layer for the model
    inputs = keras.Input(shape=(None, None, num_channels))

    conv1 = layers.Conv2D(64, 5, **conv_args)(inputs)

    conv2 = layers.Conv2D(64, 3, **conv_args)(conv1)

    conv3 = layers.Conv2D(32, 3, **conv_args)(conv2)

    conv4 = layers.Conv2D(num_channels * (factor ** 2), 3, **conv_args)(conv3)

    outputs = tf.nn.depth_to_space(conv4, factor)

    model = keras.Model(inputs, outputs)
    return model

def build_and_train_model(training_set, valid_set, upscale_factor, epochs=100):
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    checkpoint_filepath = "/tmp/checkpoint"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )

    model = custom_upscale_model(factor=upscale_factor, num_channels=1)
    model.summary()

    callbacks = [PSNRMonitor(), early_stopping_callback, model_checkpoint_callback]
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_fn)

    model.fit(
        training_set, epochs=epochs, callbacks=callbacks, validation_data=valid_set, verbose=2
    )

    model.load_weights(checkpoint_filepath)

build_and_train_model(training_set, valid_set, upscale_factor)
