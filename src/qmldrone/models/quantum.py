from abc import abstractmethod
import pennylane as qml

class QuantumLayer:
    @property
    @abstractmethod
    def weight_shapes():
        pass
    
    @abstractmethod
    def circuit():
        pass

n_qubits = 5
class OriginalCircuit(QuantumLayer):
    @property
    def weight_shapes(self):
        return {"weights": (3, n_qubits)}

    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
class FullyEntangled(QuantumLayer):
    @property
    def weight_shapes(self):
        return {"weights": (3, n_qubits, 3)}

    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]