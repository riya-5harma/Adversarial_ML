import numpy as np
import tensorflow as tf
from art.attacks.evasion import CarliniLInfMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist

tf.compat.v1.disable_eager_execution()
# Step 2: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Reduce the size of the train dataset
train_size = 10000  # Replace with your desired train dataset size
x_train = x_train[:train_size]
y_train = y_train[:train_size]

# Reduce the size of the test dataset
test_size = 100  # Replace with your desired test dataset size
x_test = x_test[:test_size]
y_test = y_test[:test_size]
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
# Create ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))


# Step 5: Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Step 6: Create the attack instance
attack = CarliniLInfMethod(classifier=classifier, targeted=True, max_iter=10, confidence=0.5, verbose=True, batch_size=1)

# Step 7: Craft the adversarial examples
x_test_adv = attack.generate(x=x_test, y=np.argmax(y_test, axis=1))

# Step 8: Evaluate the attack success
predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Attack success rate: {accuracy * 100}%")