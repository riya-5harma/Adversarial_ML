
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.compat.v1.enable_eager_execution()
# Load the CSV dataset
dataset_path = "dataset/train.csv"
df = pd.read_csv(dataset_path)

# Separate features and labels
X = df.drop(columns=["label"]).values
y = df["label"].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create and train a base model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

# Function to generate adversarial examples using FGSM
def fgsm_attack(model, x, epsilon):
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_test[0], predictions)

    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
    perturbed_x = x + epsilon * signed_grad
    perturbed_x = tf.clip_by_value(perturbed_x, 0, 1)

    return perturbed_x.numpy()

# Choose an example for the attack (modify as per your dataset)
example_index = 6
original_example = X_test[example_index]

# Generate the adversarial example using FGSM
epsilon = 0.5
perturbed_example = fgsm_attack(model, original_example, epsilon)
import matplotlib.pyplot as plt
# Plot the original and perturbed examples
plt.subplot(1, 2, 1)
plt.imshow(original_example.reshape(28, 28), cmap='gray')
plt.title("Original Example")

plt.subplot(1, 2, 2)
plt.imshow(perturbed_example.reshape(28, 28), cmap='gray')
plt.title("Perturbed Example (FGSM)")

plt.tight_layout()
plt.show()

# Evaluate the model on the adversarial example
perturbed_example = np.squeeze(perturbed_example, axis=0)
perturbed_example = np.expand_dims(perturbed_example, axis=0)
predictions = model.predict(perturbed_example)
predicted_label = np.argmax(predictions)

print("Original Label:", y_test[example_index])
print("Predicted Label on Perturbed Example:", predicted_label)