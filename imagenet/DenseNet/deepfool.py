import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions

from art.attacks.evasion import DeepFool
from art.estimators.classification import KerasClassifier
from art.utils import get_file
tf.compat.v1.disable_eager_execution()
# Load the ResNet50 model
model = DenseNet121(weights='imagenet')

epsilons = [0, .05, .1, .15, .2, .25, .3]
# Load and preprocess a sample image
image_path = get_file('sample_image.jpg', 'https://cdn.britannica.com/70/192570-138-848FB7B3/penguin-species-places-Galapagos-Antarctica.jpg?w=800&h=450&c=crop')
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(image)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Create an ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 255))

# Initialize the Deep Fool attack


# Generate adversarial examples

for epss in epsilons:
    attack = DeepFool(classifier,epsilon=epss)
    adversarial_image = attack.generate(x)  
    preds_original = classifier.predict(x)
    preds_adv = classifier.predict(adversarial_image)
    original_label = decode_predictions(preds_original, top=3)[0]
    adversarial_label = decode_predictions(preds_adv, top=3)[0]
    
    print("For epsilon : ",epss)
    print("Original Image Predictions:")
    for class_name, class_description, class_probability in original_label:
        print(f"- {class_description}: {class_probability:.2%}")

    print("\nAdversarial Image Predictions:")
    for class_name, class_description, class_probability in adversarial_label:
        print(f"- {class_description}: {class_probability:.2%}")
