import torch
from models.alexnet import AlexNet
from models.lenet import LeNet

def main():

    alexnet = AlexNet()
    trainable_params = sum(p.numel() for p in alexnet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in alexnet.parameters())
    print(f"AlexNet Trainable parameters: {trainable_params}")
    print(f"AlexNet Total params: {total_params:,}")


    lenet = LeNet()
    trainable_params = sum(p.numel() for p in lenet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lenet.parameters())
    print(f"LeNet Trainable parameters: {trainable_params}")
    print(f"LeNet Total params: {total_params:,}")

if __name__ == "__main__":
    main()