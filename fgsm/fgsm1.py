# To load your own dataset in the code provided, you can follow these steps:

# 1. Prepare your dataset: Organize your dataset in a suitable format. For example, if you have images, make sure they are stored in a folder structure where each class has its own subfolder containing the corresponding images.

# 2. Load the dataset: Use libraries like `scikit-learn` or `tensorflow.keras.preprocessing.image` to load and preprocess your dataset.

# Here's an updated version of the previous code snippet, modified to demonstrate how to load a custom dataset:

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset
train_data_dir = 'C:/Users/chira/OneDrive/Desktop/fgsm/dataset/train.csv'
test_data_dir = 'C:/Users/chira/OneDrive/Desktop/fgsm/dataset/test.csv'

# Specify the input size and number of classes in your dataset
input_size = (28, 28)
num_classes = 10

# Data preprocessing and augmentation
train_data_generator = ImageDataGenerator(rescale=1./255,
                                         rotation_range=10,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         shear_range=0.1,
                                         zoom_range=0.1,
                                         horizontal_flip=False,
                                         fill_mode='nearest')

test_data_generator = ImageDataGenerator(rescale=1./255)

# Load the training data
train_generator = train_data_generator.flow_from_directory(train_data_dir,
                                                          target_size=input_size,
                                                          batch_size=32,
                                                          class_mode='categorical')

# Load the test data
test_generator = test_data_generator.flow_from_directory(test_data_dir,
                                                        target_size=input_size,
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=False)

# Create and train a base model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(input_size[0], input_size[1], 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5)

# Function to generate adversarial examples using FGSM
def fgsm_attack(model, image, epsilon):
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = tf.keras.losses.categorical_crossentropy(test_generator.labels[0], predictions)

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    perturbed_image = image + epsilon * signed_grad
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)

    return perturbed_image.numpy()

# Choose an image for the attack (modify as per your dataset)
image_index = 0
original_image = test_generator[0][0][image_index]

# Generate the adversarial example using FGSM
epsilon = 0.1
perturbed_image = fgsm_attack(model, original_image, epsilon)

# Plot the original and perturbed images for comparison
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(perturbed_image[0])
plt.title("Perturbed Image (FGSM)")

plt.show()

# Evaluate the model on the adversarial example
perturbed_image = tf.squeeze(perturbed_image, axis=0)
perturbed_image = np.expand_dims(perturbed_image, axis=0)
predictions = model.predict(perturbed_image)
