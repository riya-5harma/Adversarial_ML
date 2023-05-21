
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
tf.enable_eager_execution()
# Load the dataset (e.g., MNIST)
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize the pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create and train a base model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

# Function to generate adversarial examples using FGSM
def fgsm_attack(model, image, epsilon):
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(test_labels[0], predictions)

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    perturbed_image = image + epsilon * signed_grad
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)

    return perturbed_image.numpy()

# Choose an image for the attack
image_index = 158
original_image = test_images[image_index]

# Generate the adversarial example using FGSM
epsilon = 0.2
perturbed_image = fgsm_attack(model, original_image, epsilon)

# Plot the original and perturbed images for comparison
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(perturbed_image[0], cmap='gray')
plt.title("Perturbed Image (FGSM)")

plt.show()

# Evaluate the model on the adversarial example
perturbed_image = tf.squeeze(perturbed_image, axis=0)
perturbed_image = np.expand_dims(perturbed_image, axis=0)
predictions = model.predict(perturbed_image)
predicted_label = np.argmax(predictions)

print("Original Label:", test_labels[image_index])
print("Predicted Label on Perturbed Image:", predicted_label)
