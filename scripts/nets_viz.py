import pandas as pd
import json
import matplotlib.pyplot as plt

lenet_logs = '/home/YSaxephone/sp26-nmep-hw3/output/lenet/metrics.json'
alexnet_logs = '/home/YSaxephone/sp26-nmep-hw3/output/alexnet/metrics.json'
resnet_logs = '/home/YSaxephone/sp26-nmep-hw3/output/resnet18/metrics.json' 

lenet_records = []
alexnet_records = []
resnet_records = []

with open(lenet_logs, 'r') as f:
    for line in f:
        lenet_records.append(json.loads(line))

with open(alexnet_logs, 'r') as f:
    for line in f:
        alexnet_records.append(json.loads(line))

with open(resnet_logs, 'r') as f:
    for line in f:
        resnet_records.append(json.loads(line))

lenet_pd = pd.DataFrame(lenet_records)
alexnet_pd = pd.DataFrame(alexnet_records)
resnet_pd = pd.DataFrame(resnet_records)


#print(lenet_pd)
#print(alexnet_pd)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs = axs.flatten()

columns_to_plot = ["train_acc", "train_loss", "val_acc", "val_loss"]

for i, col in enumerate(columns_to_plot):
    #axs[i].plot(lenet_pd[col], label='LeNet', color='blue')
    axs[i].plot(alexnet_pd[col], label='AlexNet', color='red')
    axs[i].plot(resnet_pd[col], label='ResNet18', color='green')
    axs[i].set_title(col)
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel(col)
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig('alexnet_resnet_comparison.png', dpi=300)

