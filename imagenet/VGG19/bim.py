import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import BasicIterativeMethod
from art.utils import to_categorical,get_file

epsilons = [0, .05, .1, .15, .2, .25, .3]
tf.compat.v1.disable_eager_execution()
# Load the pre-trained ResNet50 model
model = VGG19(weights='imagenet')

# Wrap the model with an ART-compatible classifier
classifier = KerasClassifier(model=model, clip_values=(0, 255))

# Select an image for the attack
image_path = get_file('sample_image.jpg', 'https://cdn.britannica.com/70/192570-138-848FB7B3/penguin-species-places-Galapagos-Antarctica.jpg?w=800&h=450&c=crop')
x = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(x)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Get the true label of the image
true_label = np.argmax(classifier.predict(x), axis=1)


for epss in epsilons:
    
    attack = BasicIterativeMethod(estimator=classifier, eps=epss, eps_step=0.1, max_iter=10)
    x_adv = attack.generate(x, y=np.argmax(to_categorical(true_label, 1000), axis=1))   
    preds_original = classifier.predict(x)
    preds_adv = classifier.predict(x_adv)
    original_label = decode_predictions(preds_original, top=3)[0]
    adversarial_label = decode_predictions(preds_adv, top=3)[0]
    
    print("For epsilon : ",epss)
    print("Original Image Predictions:")
    for class_name, class_description, class_probability in original_label:
        print(f"- {class_description}: {class_probability:.2%}")

    print("\nAdversarial Image Predictions:")
    for class_name, class_description, class_probability in adversarial_label:
        print(f"- {class_description}: {class_probability:.2%}")
