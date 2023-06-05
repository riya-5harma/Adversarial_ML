import numpy as np
from art.attacks.evasion import CarliniLInfMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Normalize the pixel values to [0, 1]
x_train = x_train / max_pixel_value
x_test = x_test / max_pixel_value

# Define your custom model (replace with your own model architecture)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create the ART classifier
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value))

# Train your model on the MNIST dataset (replace with your own training process)
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Select a target image and its true label
target_image = x_test[0]  # Replace with the desired target image
true_label = y_test[0]   # Replace with the true label of the target image

# Create the Carlini and Wagner L_inf attack object
attack = CarliniLInfMethod(classifier=classifier, targeted=False, eps=0.3, max_iter=100)

# Generate adversarial examples using the attack
x_test_adv = attack.generate(x=target_image.reshape((1, 28, 28)), y=true_label)

# Convert the adversarial example back to the original pixel range [0, 255]
x_test_adv = x_test_adv * max_pixel_value

# Evaluate the model on the adversarial example
predictions = model.predict(np.expand_dims(x_test_adv, axis=0))
predicted_label = np.argmax(predictions)

print("Original label:", true_label)
print("Predicted label on adversarial example:", predicted_label)
