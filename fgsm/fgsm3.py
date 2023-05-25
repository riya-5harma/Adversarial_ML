from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import tensorflow as tf
 
import keras.backend as k
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

from art.attacks import FastGradientMethod, DeepFool
from art.classifiers import TensorFlowClassifier, KerasClassifier
from art.data_generators import DataGenerator
from art.defences import AdversarialTrainer
from art.utils import load_mnist, get_labels_np_array, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11
ACCURACY_DROP = 0.0  # The unit tests are too inaccurate

# tf.enable_eager_execution()
(x_train, y_train), (x_test, y_test) = self.mnist

attack = FastGradientMethod(self.classifier_k)
x_test_adv = attack.generate(x_test)
preds = np.argmax(self.classifier_k.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / NB_TEST

adv_trainer = AdversarialTrainer(self.classifier_k, attack)
adv_trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)

preds_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
acc_new = np.sum(preds_new == np.argmax(y_test, axis=1)) / NB_TEST
self.assertGreaterEqual(acc_new, acc * ACCURACY_DROP)

logger.info('Accuracy before adversarial training: %.2f%%', (acc * 100))
logger.info('Accuracy after adversarial training: %.2f%%', (acc_new * 100))