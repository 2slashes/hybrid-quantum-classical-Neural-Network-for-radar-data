import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.preprocessing import LabelBinarizer
import torchvision.datasets as ds
from itertools import cycle
import os

#----------------------------STUFF---------------------------------------
num_outputs = None
def set_outputs(outputs: int):
    global num_outputs
    num_outputs = outputs

def get_outputs():
    global num_outputs
    if num_outputs is None:
        raise Exception("Number of outputs not configured. Use set_outputs to set the number of outputs.")
    return num_outputs

device = None
def get_device():
    global device
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    return device

# this should be customizable 
drone_type_map = ["DJI_Matrice_300_RTK", "DJI_Mavic_Air_2",
                  "DJI_Mavic_Mini", "DJI_Phantom_4", "Parrot_Disco"]
# same with this
roc_curve_minimum = 1e-3

radar_path = "Radar"
data_dir = "two_sided"
model_dir = "two_sided_models"
plot_dir = "two_sided_plots"
def set_root_dir(path):
    global radar_path
    radar_path = path

# unsure what this is for
model_snr = 5
#----------------------------MODEL DEFINITION----------------------------------
n_qubits = 5
dev = qml.device("lightning.gpu", wires=n_qubits)

# qnode shared within the HybridRadarClassifier class, and between the detection and classification models
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Define the QLayer
n_layers = 3
weight_shapes = {"weights": (n_layers, n_qubits)}

# This class is shared between the detection and classification models
# Detection model has 2 outputs, classification has 5 outputs, otherwise they are identical
# it might be possible that more outputs are desired for classification, maybe it should be an input
class HybridRadarClassifier(nn.Module):
  def __init__(self, conf):
        super(HybridRadarClassifier, self).__init__()
        self.outputs = get_outputs()
        # i/p shape - (batch_size, Channel_in, Height_in, Width_in) - (2, 16, 251)
        self.conv1 = nn.Conv2d(2, 16, (3,3), padding =1) # o/p shape - (16, 16, 251)
        self.IN1 = nn.InstanceNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2, padding =1) # o/p shape (16, 8, 126)
        self.conv2 = nn.Conv2d(16, 32, (5,5), padding = 2) # o/p shape (32, 8, 126)
        self.IN2 = nn.InstanceNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # o/p shape (32, 4, 63)

        #quantum layer
        self.qlayer1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer3 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer4 = qml.qnn.TorchLayer(qnode, weight_shapes)

        #fully connected layers
        self.fc1 = nn.Linear( 32 * 12 * 21 , 120)
        self.fc2 = nn.Linear( 120 , 20 )
        self.fc3 = nn.Linear( 20 , self.outputs)  # o/p shape should be (batch_size,5,1,1)

        self.drop = nn.Dropout2d(p=0.5)
        self.relu = nn.LeakyReLU()

  def forward(self, x):
        # i/p shape - (batch_size, Channel_in, Height_in, Width_in) - (2, 16, 251)
        x = self.pool1(self.relu(self.IN1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool2(self.relu(self.IN2(self.conv2(x))))
        x = self.drop(x)
        #flatten
        x = x.view(-1, 32 * 12 * 21) # x = torch.flatten(x, start_dim=1
        #FC layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # qlayers connected parallely
        # Make sure the values are normalised to lie in the range [0,pi]
        # since the qnodes are using angle embedding
        x = F.normalize(x) * np.pi
        x_1, x_2, x_3, x_4 = torch.split(x, 5, dim=1)
        #print(f"input {x_1=}")
        x_1 = self.qlayer1(x_1)
        x_2 = self.qlayer2(x_2)
        x_3 = self.qlayer3(x_3)
        x_4 = self.qlayer4(x_4)
        #print(f"output {x_1=}")
        x = torch.cat([x_1, x_2, x_3, x_4], axis=1)

     # if we want qlayers connected serially?
       # x = self.qlayer1(x)
       # x = self.qlayer2(x)
       # x = self.qlayer3(x)
       # x = self.qlayer4(x)

        x = self.fc3(x)
        return x
  
# ------------------TRAIN AND TEST------------------------

# this needs to be removed expeditiously
model_path = f"{radar_path}/{model_dir}"
plot_path = f"{radar_path}/{plot_dir}"
os.system(f"mkdir -p {model_path}")
os.system(f"mkdir -p {plot_path}")

def dataloader(file_extension):
    data = np.load(file_extension)
    return data


def plot_spectrogram(sp):
    fig, axs = plt.subplots(2, ncols=1, figsize=(14,3))

    axs[0].imshow(sp[0])
    axs[1].imshow(sp[1])

    plt.tight_layout()
    plt.show()


def plot_multiclass_roc(target, probs, snr, plot_file=None,
                        remove_zeros=False, semilog_axes=True):
    lb = LabelBinarizer().fit(target)
    one_hot_target = lb.transform(target)

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = cycle(['#348ABD', '#b74331', '#8EBA42', '#FBC15E', '#988ED5'])
    for class_id, color in zip(range(5), colors):
        fpr, tpr, _ = roc_curve(one_hot_target[:,class_id], probs[:,class_id])
        if remove_zeros is True:
            zero_indices = np.where(np.isclose(fpr, 0))
            fpr = np.delete(fpr, zero_indices)
            tpr = np.delete(tpr, zero_indices)
        else:
            fpr[np.isclose(fpr,0)] = 1e-10
        display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(
            name=f"{drone_type_map[class_id]}",
            ax=ax,
            color=color)
    if semilog_axes is True:
        display.ax_.set_xscale("log")
        display.ax_.set_xlim(roc_curve_minimum, 1.0)
    display.ax_.set_xlabel("False Positive Rate")
    display.ax_.set_ylabel("True Positive Rate")
    line = plt.plot(np.geomspace(roc_curve_minimum,1.0), np.geomspace(roc_curve_minimum,1.0),
                    color='k', linestyle="dashed", label="Chance")
    display.ax_.set_title(f"ROC curves for SNR {snr}dB")
    display.ax_.legend()
    plt.tight_layout()
    if plot_file is not None:
        plt.savefig(plot_file)
    else:
        plt.show()

# train and test functions generalized to work with either model
def train(conf, trainLoader, device):

    net = HybridRadarClassifier(conf).to(device)

    optim = torch.optim.AdamW(net.parameters(), lr=conf["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    for x in range(conf["epochs"]):

        net.train()

        for i, data in enumerate(trainLoader):
            # plot_spectrogram(torch.squeeze(data[0]))
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optim.zero_grad()

            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()
            loss_val = loss.item()

        print('Train Epoch: {} Loss: {:.6f}'.format(x, loss_val))
        if x > conf['min_epochs'] and loss_val < conf['loss_threshold']:
            break

    if conf['save_model'] is True:
        print(f"Saving model state to {conf['model_path']}")
        torch.save(net.state_dict(), conf['model_path'])

def test(conf, testLoader, device, plot_dir=None):

    # load model state
    net = HybridRadarClassifier(conf).to(device)
    net.load_state_dict(torch.load(conf['model_path']))
    net.eval()

    correct = 0
    total = 0
    testloss = 0
    predicted = None
    probabilities = None
    target = None

    loss_fn = nn.CrossEntropyLoss().to(device)

    confm = np.zeros((net.outputs, net.outputs), dtype=int)
    for i, data in enumerate(testLoader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        if target is None:
            target = labels
        else:
            target = torch.cat((target, labels))

        outputs = net(inputs.float())
        loss = loss_fn(outputs, labels)
        testloss += loss.item()

        partial_probabilities = nn.Softmax(dim=1)(outputs.data)
        if probabilities is None:
            probabilities = partial_probabilities
        else:
            probabilities = torch.cat((probabilities, partial_probabilities))

        _, partial_predicted = torch.max(outputs.data, 1)
        if predicted is None:
            predicted = partial_predicted
        else:
            predicted = torch.cat((predicted, partial_predicted))

    target = target.cpu()
    probabilities = probabilities.cpu()
    predicted = predicted.cpu()

    roc_plot_file = None
    conf_plot_file = None
    if plot_dir is not None:
        roc_plot_file = f"{plot_dir}/hybrid_classifier_roc_model-{model_snr}_signal-{conf['SNR']}.pdf"
        conf_plot_file = f"{plot_dir}/hybrid_classifier_conf_model-{model_snr}_signal-{conf['SNR']}.pdf"
    plot_multiclass_roc(target.cpu(), probabilities.cpu(), conf['SNR'], plot_file=roc_plot_file)

    confm = confusion_matrix(target, predicted)
    print(f"{confm=}")
    if conf['plot_confusion'] is True:
        plt.close()
        fig, ax = plot_confusion_matrix(conf_mat=confm,
                                        show_normed=True,
                                        colorbar=True,
                                        class_names=drone_type_map)
        ax.set_title(f"Confusion matrix for SNR {conf['SNR']}dB")
        plt.tight_layout()
        if conf_plot_file is not None:
            plt.savefig(conf_plot_file)
        else:
            plt.show()

    f1 = multiclass_f1_score(predicted, target, num_classes=5, average='micro')
    print(f"{f1=}")

#----------------------TRAINING CODE-----------------------
def train_model():
    for snr, threshold in zip([5], # these numbers seem arbitrary
                            [0.002]):
        conf = {}
        conf['f_s'] = 10_000
        conf['SNR'] = snr
        conf['batch_size'] = 8
        conf['epochs'] = 2
        conf['min_epochs'] = 50
        conf['learning_rate'] = 0.001 # these should be parameters
        conf['loss_threshold'] = threshold
        conf['save_model'] = True
        conf['model_path'] = f"{model_path}/hybrid-parallel-model-{snr}.pt"

        trainset_root = f"{radar_path}/{data_dir}/trainset/{conf['f_s']}fs/{conf['SNR']}SNR"
        trainds = ds.DatasetFolder(trainset_root, dataloader, extensions=("npy",))
        trainLoader = torch.utils.data.DataLoader(trainds, conf["batch_size"], shuffle=True, num_workers=2)

        print(f"SNR: {snr} dB")
        train(conf, trainLoader, get_device())

#---------------------EVALUATION------------------------------
def eval_model():
    model_snr = 5
    for snr in [20, 15, 10, 5, 0, -5]:
        conf = {}
        conf['f_s'] = 10_000
        conf['SNR'] = snr
        conf['batch_size'] = 8
        conf['epochs'] = 10
        conf['learning_rate'] = 0.001
        conf['plot_confusion'] = True
        conf['model_path'] = f"{model_path}/hybrid-parallel-model-{model_snr}.pt"

        testset_root = f"{radar_path}/{data_dir}/testset/{conf['f_s']}fs/{conf['SNR']}SNR"
        testds = ds.DatasetFolder( testset_root, dataloader, extensions=("npy",))
        testLoader = torch.utils.data.DataLoader( testds, conf["batch_size"], shuffle=True, num_workers=2)

        print(f"SNR: {snr} dB")
        test(conf, testLoader, get_device(), plot_dir=plot_path)