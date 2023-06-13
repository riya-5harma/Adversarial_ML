import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from art.attacks.evasion import DeepFool
from art.estimators.classification import KerasClassifier
from art.utils import get_file
tf.compat.v1.disable_eager_execution()
# Load the ResNet50 model
model = ResNet50(weights='imagenet')

# Load and preprocess a sample image
image_path = get_file('sample_image.jpg', 'https://cdn.britannica.com/70/192570-138-848FB7B3/penguin-species-places-Galapagos-Antarctica.jpg?w=800&h=450&c=crop')
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(image)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Create an ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 255))

# Initialize the Deep Fool attack
attack = DeepFool(classifier)

# Generate adversarial examples
x_adv = attack.generate(x)

# Make predictions on the original and adversarial examples
predictions = model.predict(x)
adv_predictions = model.predict(x_adv)

# Decode and print the predictions
original_label = decode_predictions(predictions, txop=3)[0]
adversarial_label = decode_predictions(adv_predictions, top=3)[0]

print("Original Image Predictions:")
for class_name, class_description, class_probability in original_label:
    print(f"- {class_description}: {class_probability:.2%}")

print("\nAdversarial Image Predictions:")
for class_name, class_description, class_probability in adversarial_label:
    print(f"- {class_description}: {class_probability:.2%}")
