import numpy as np
import imageio
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import get_file
from tensorflow.keras.preprocessing.image import img_to_array, load_img

tf.compat.v1.disable_eager_execution()

epsilons = [0, .05, .1, .15, .2, .25, .3]
# Download and load a sample image from the ImageNet dataset
image_path = get_file('sample_image.jpg', 'https://cdn.britannica.com/70/192570-138-848FB7B3/penguin-species-places-Galapagos-Antarctica.jpg?w=800&h=450&c=crop')
image = load_img(image_path, target_size=(299, 299))
image = img_to_array(image)

# Preprocess the image
preprocessed_image = preprocess_input(np.expand_dims(image, axis=0))

# Load the pre-trained ResNet50 model
model = Xception(weights='imagenet')

# Create a KerasClassifier instance
classifier = KerasClassifier(model=model, clip_values=(-1.0, 1.0))

# Get the predicted label for the original image
original_preds = classifier.predict(preprocessed_image)
original_label = decode_predictions(original_preds, top=1)[0][0][1]


for epss in epsilons:
    attack = FastGradientMethod(estimator=classifier, eps=epss)
    adversarial_image = attack.generate(x=preprocessed_image) 
   
    original_preds = classifier.predict(preprocessed_image)

    adversarial_preds = classifier.predict(adversarial_image)
    original_label = decode_predictions(original_preds, top=3)[0]
    adversarial_label = decode_predictions(adversarial_preds, top=3)[0]
    
    print("\nFor Perturbation Value : ",epss)
    print("\nTrue Label:")
    for class_name, class_description, class_probability in original_label:
        print(f"- {class_description}: {class_probability:.2%}")

    print("\nAdversarial Label:")
    for class_name, class_description, class_probability in adversarial_label:
        print(f"- {class_description}: {class_probability:.2%}")