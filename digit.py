import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

tf.random.set_seed(3)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

input_img_path = input('\nEnter the path of the image to be predicted: ')
input_img = cv2.imread(input_img_path)
greyscale = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
input_img_resize = cv2.resize(greyscale, (28, 28))
input_img_resize = input_img_resize / 255
image_reshaped = np.reshape(input_img_resize, [1, 28, 28])
input_prediction = model.predict(image_reshaped)
input_pred_label = np.argmax(input_prediction)
print("\nThe Handwritten Digit is recognised as: ", input_pred_label)
