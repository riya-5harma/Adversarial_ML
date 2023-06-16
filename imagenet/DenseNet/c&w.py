import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import CarliniLInfMethod
from art.utils import get_file

# tf.compat.v1.disable_eager_execution()
model = tf.keras.applications.DenseNet121(weights='imagenet')
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
label = decode_predictions(preds, top=3)[0] 
original_label = decode_predictions(model.predict(x.reshape((1, 224, 224, 3))), top=3)[0]
print("Original Image Predictions:")
for class_name, class_description, class_probability in original_label:
        print(f"- {class_description}: {class_probability:.2%}")

print("\nAdversarial Image Predictions:")
for class_name, class_description, class_probability in label:
        print(f"- {class_description}: {class_probability:.2%}")