import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import BasicIterativeMethod
from art.utils import to_categorical

epsilons = [0, .05, .1, .15, .2, .25, .3]
tf.compat.v1.disable_eager_execution()
# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Wrap the model with an ART-compatible classifier
classifier = KerasClassifier(model=model, clip_values=(0, 255))

# Select an image for the attack
image_path = 'example.jpg'
x = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(x)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Get the true label of the image
true_label = np.argmax(classifier.predict(x), axis=1)

# # Configure the BIM attack
# attack = BasicIterativeMethod(estimator=classifier, eps=0.0, eps_step=0.1, max_iter=10)

# # Generate adversarial examples
# x_adv = attack.generate(x, y=np.argmax(to_categorical(true_label, 1000), axis=1))

# # Evaluate the attack
# preds_original = classifier.predict(x)
# preds_adv = classifier.predict(x_adv)

# print('Shape of preds_original:', preds_original.shape)
# print('Shape of preds_adv:', preds_adv.shape)

# label_original = decode_predictions(preds_original, top=1)[0][0][1]
# label_adv = decode_predictions(preds_adv, top=1)[0][0][1]

# print('Original Image Prediction:', label_original)
# print('Adversarial Image Prediction:', label_adv)

for epss in epsilons:
    
    attack = BasicIterativeMethod(estimator=classifier, eps=epss, eps_step=0.1, max_iter=10)
    x_adv = attack.generate(x, y=np.argmax(to_categorical(true_label, 1000), axis=1))   
    preds_original = classifier.predict(x)
    preds_adv = classifier.predict(x_adv)
    label_original = decode_predictions(preds_original, top=1)[0][0][1]
    label_adv = decode_predictions(preds_adv, top=1)[0][0][1]
    print('for epsilon value:', epss)
    print('Original Image Prediction:', label_original)
    print('Adversarial Image Prediction:', label_adv)