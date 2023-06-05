
from art.defences.trainer import AdversarialTrainer
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


# Adversarial training parameters
epochs = 1  # Number of training epochs

# Create an adversarial trainer
adv_trainer = AdversarialTrainer(classifier, attack)

# Adversarial training loop
for epoch in range(epochs):
    adv_trainer.fit(x_train, y_train)

# Evaluate the model on test data
test_loss, test_acc = classifier.model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
