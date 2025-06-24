from qmldrone.models.classic import ClassicalRadarClassifier
from ._base import train as base_train, test as base_test

def train(conf):
    base_train(conf, lambda c: ClassicalRadarClassifier(c))

def test(conf):
    base_test(conf, lambda c: ClassicalRadarClassifier(c))
