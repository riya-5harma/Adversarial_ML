import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import KerasClassifier

tf.compat.v1.disable_eager_execution()
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape input data
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Convert labels to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes=10)

# Create and compile the model
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Create ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Create BIM attack instance
attack = BasicIterativeMethod(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=100)

# Generate adversarial examples
x_test_adv = attack.generate(x=x_test[:10])

# Evaluate the accuracy of the adversarial examples
predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
accuracy = np.mean(predictions == np.argmax(y_test[:10], axis=1))
print(f"Accuracy on adversarial examples: {accuracy}")
