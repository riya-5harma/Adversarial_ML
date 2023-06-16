import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from art.attacks.evasion import CarliniLInfMethod
from art.estimators.classification import KerasClassifier
from PIL import Image
from art.utils import get_file
tf.compat.v1.disable_eager_execution()
# Load pre-trained ResNet-50 model
model = VGG19(weights='imagenet')

# Define input image path
image_path = get_file('sample_image.jpg', 'https://cdn.britannica.com/70/192570-138-848FB7B3/penguin-species-places-Galapagos-Antarctica.jpg?w=800&h=450&c=crop')

# Load and preprocess the input image
image = Image.open(image_path).resize((224, 224))  # Resize image to match input size of ResNet-50
x = np.array(image)
x = preprocess_input(x)

# Create ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 255))

# Create the Carlini and Wagner L_inf attack object
attack = CarliniLInfMethod(classifier)
x_adv = attack.generate(x.reshape((1, 224, 224, 3)))
preds = model.predict(x_adv)
label = decode_predictions(preds, top=3)[0] 
original_label = decode_predictions(model.predict(x.reshape((1, 224, 224, 3))), top=3)[0]
print("Original Image Predictions:")
for class_name, class_description, class_probability in original_label:
        print(f"- {class_description}: {class_probability:.2%}")

print("\nAdversarial Image Predictions:")
for class_name, class_description, class_probability in label:
        print(f"- {class_description}: {class_probability:.2%}")