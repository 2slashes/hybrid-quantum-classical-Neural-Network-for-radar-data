FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

WORKDIR /root
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install pennylane
RUN pip3 install custatevec_cu12
RUN pip3 install pennylane-lightning-gpu
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install -U scikit-learn
RUN pip3 install pandas
RUN pip3 install wandb
RUN pip3 install mlxtend
RUN pip3 install torcheval
RUN pip3 install pyyaml

WORKDIR /workspace