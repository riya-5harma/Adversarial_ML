import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import ProjectedGradientDescent
from art.utils import get_file

epsilons = [0, .05, .1, .15, .2, .25, .3]
tf.compat.v1.disable_eager_execution()
# Load the pre-trained ResNet50 model
model = InceptionV3(weights='imagenet')

# Wrap the model with an ART-compatible classifier
classifier = KerasClassifier(model=model, clip_values=(0, 255))

# Select an image for the attack
image_path = get_file('sample_image.jpg', 'https://cdn.britannica.com/70/192570-138-848FB7B3/penguin-species-places-Galapagos-Antarctica.jpg?w=800&h=450&c=crop')
x = image.load_img(image_path, target_size=(299, 299))
x = image.img_to_array(x)
x = np.expand_dims(x, axis=0)
preprocessed_image = preprocess_input(x)

# Get the true label of the image
true_label = np.argmax(classifier.predict(x), axis=1)
classifier = KerasClassifier(model=model, clip_values=(0, 255))

# Create the PGD attack instance


for epss in epsilons:
    
    attack = ProjectedGradientDescent(estimator=classifier, eps=epss, eps_step=0.1, max_iter=100, targeted=False)
    adversarial_image = attack.generate(x=preprocessed_image)  
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
