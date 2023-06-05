import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

tf.compat.v1.disable_eager_execution()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the data to match the input shape expected by ART (batch size, width, height, channels)
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.legacy.Adam(),
              metrics=['accuracy'])

# Create an ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))
model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=1)
x_test_subset = x_test[:100]
y_test_subset = y_test[:100]
# Initialize the FGSM attack
attack = FastGradientMethod(estimator=classifier, eps=0.1)

# Generate adversarial examples
x_test_adv = attack.generate(x=x_test_subset)
# Evaluate the accuracy of the classifier on the adversarial examples
predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
accuracy = np.sum(predictions == np.argmax(y_test_subset, axis=1)) / len(y_test_subset)
print("Accuracy on adversarial examples: {:.2f}%".format(accuracy * 100))