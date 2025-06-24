from qmldrone.models.hybrid import HybridRadarClassifier
from ._base import train as base_train, test as base_test

def train(conf, qlayer, ):
    circuit = qlayer.circuit
    weight_shapes = qlayer.weight_shapes

    base_train(
        conf, 
        lambda c: HybridRadarClassifier(c, circuit, weight_shapes)
    )

def test(conf, qlayer):
    circuit = qlayer.circuit
    weight_shapes = qlayer.weight_shapes

    base_test(
        conf, 
        lambda c: HybridRadarClassifier(c, circuit, weight_shapes)
    )
