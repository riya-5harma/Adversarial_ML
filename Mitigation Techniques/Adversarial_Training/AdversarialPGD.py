
from art.defences.trainer import AdversarialTrainer
import numpy as np
from tensorflow.keras.datasets import mnist

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier

tf.compat.v1.disable_eager_execution()
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the dataset
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.legacy.Adam(),
              metrics=['accuracy'])
# Create ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Create Projected Gradient Descent attack
attack = ProjectedGradientDescent(classifier, norm=np.inf, eps=0.1, eps_step=0.1, max_iter=10)

# Perform the attack
x_test_adv = attack.generate(x_test)

# Evaluate the success rate
preds = np.argmax(classifier.predict(x_test_adv), axis=1)

# Calculate the accuracy
accuracy = np.sum(preds == y_test) / len(y_test)
print("Accuracy on adversarial examples: {:.2f}%".format(accuracy * 100))
# Create an adversarial trainer
trainer = AdversarialTrainer(classifier, attack)

# Train the model with adversarial training
trainer.fit(x_train, y_train, nb_epochs=1, batch_size=32)

# Evaluate the model on clean and adversarial test examples
# clean_acc = np.mean(np.argmax(classifier.predict(x_test), axis=1) == y_test)

adv_acc = np.mean(np.argmax(classifier.predict(x_test_adv), axis=1) == y_test)

# print(f"Clean accuracy: {clean_acc:.4f}")
print(f"Adversarial accuracy: {adv_acc:.4f}")
