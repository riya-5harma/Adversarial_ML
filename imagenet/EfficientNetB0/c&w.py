import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import CarliniLInfMethod
from art.utils import get_file

# tf.compat.v1.disable_eager_execution()
model = tf.keras.applications.EfficientNetB0(weights='imagenet')
image_path = get_file('sample_image.jpg', 'https://cdn.britannica.com/70/192570-138-848FB7B3/penguin-species-places-Galapagos-Antarctica.jpg?w=800&h=450&c=crop')

image = Image.open(image_path).resize((224, 224))
x = np.array(image)
x = preprocess_input(x)
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=1000,
    clip_values=(0, 255),
    input_shape=(224, 224, 3),
    preprocessing_defences=None,
    preprocessing=(0, 1),
)
attack = CarliniLInfMethod(classifier)
x_adv = attack.generate(x.reshape((1, 224, 224, 3)))
preds = model.predict(x_adv)
label = decode_predictions(preds, top=1)[0][0][1]
original_label = decode_predictions(model.predict(x.reshape((1, 224, 224, 3))), top=1)[0][0][1]
print(f"Original Label: {original_label}")
print(f"Adversarial Label: {label}")
