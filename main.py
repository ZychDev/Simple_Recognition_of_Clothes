import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#load data
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#shrink data down pixel value = 255 we want to shrink that down
train_images = train_images/255.0
test_images = test_images/255.0

#show data img in matplot
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.show()


#create model
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

#model compile options
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])#train model
#epoch = how many times the model gonan see information
model.fit(train_images, train_labels, epochs=10)

#test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)