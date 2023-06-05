import numpy as np
from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Reshape the data to fit the PyTorch classifier
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

# Define the PyTorch classifier
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Define the attack
attack = BasicIterativeMethod(estimator=classifier, eps=0.3, eps_step=0.1, max_iter=40)

# Generate adversarial examples
x_test_adv = attack.generate(x=x_test)

# Evaluate the accuracy of the classifier on the adversarial examples
accuracy = np.sum(np.argmax(classifier.predict(x=x_test_adv), axis=1) == np.argmax(y_test, axis=1)
) / len(y_test)
print(f"Accuracy on adversarial examples: {accuracy * 100:.2f}%")
