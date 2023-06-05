import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the images

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

# Train your model on the MNIST dataset (replace with your own training process)
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Select a target image and its true label
target_image = x_test[0]  # Replace with the desired target image
true_label = y_test[0]   # Replace with the true label of the target image

# Define the loss function for the Carlini and Wagner L_inf attack
def loss_fn(inputs, target):
    logits = model(inputs)
    loss = tf.keras.losses.sparse_categorical_crossentropy(target, logits, from_logits=True)
    return loss

# Perform the Carlini and Wagner L_inf attack
def carlini_wagner_linf_attack(model, loss_fn, target_image, true_label, epsilon=0.3, max_iterations=100):
    adv_image = tf.Variable(target_image, dtype=tf.float32, trainable=True)

    for i in range(max_iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            loss = loss_fn(adv_image, true_label)
        
        gradients = tape.gradient(loss, adv_image)
        perturbation = tf.sign(gradients)
        adv_image.assign_add(epsilon * perturbation)
        
        # Clip the adversarial image within the valid range
        adv_image.assign(tf.clip_by_value(adv_image, 0, 1))

    return adv_image.numpy()

# Generate the adversarial example using the Carlini and Wagner L_inf attack
adversarial_example = carlini_wagner_linf_attack(model, loss_fn, target_image, true_label)

# Evaluate the model on the adversarial example
predictions = model.predict(np.expand_dims(adversarial_example, axis=0))
predicted_label = np.argmax(predictions)

print("Original label:", true_label)
print("Predicted label on adversarial example:", predicted_label)
