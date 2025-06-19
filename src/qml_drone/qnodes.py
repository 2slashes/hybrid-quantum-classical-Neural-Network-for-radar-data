from abc import abstractmethod
import pennylane as qml

n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

class QuantumLayer:
    @property
    @abstractmethod
    def weight_shapes():
        pass
    
    @abstractmethod
    def circuit():
        pass

class OriginalCircuit(QuantumLayer):
    weight_shapes = {"weights": (3, n_qubits)}

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
class FullyEntangled(QuantumLayer):
    weight_shapes = {"weights": (3, n_qubits, 3)}

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]