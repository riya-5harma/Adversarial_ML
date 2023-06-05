import numpy as np
from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
from torchvision.models import resnet18
import torch.nn as nn

# Create the model
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)  # Modify the last layer to match the number of classes

# Create the classifier
classifier = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(), clip_values=(min_pixel_value, max_pixel_value),
                               input_shape=(1, 28, 28), nb_classes=10)
image = x_test[0]
true_label = np.argmax(y_test[0])
attack = BasicIterativeMethod(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=100)
adversarial_example = attack.generate(x=image, y=true_label)
predictions = classifier.predict(np.array([adversarial_example]))
predicted_label = np.argmax(predictions)
print("True label:", true_label)
print("Predicted label on original image:", classifier.predict(np.array([image])).argmax())
print("Predicted label on adversarial example:", predicted_label)
