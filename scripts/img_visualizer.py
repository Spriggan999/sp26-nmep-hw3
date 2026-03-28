import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
#from config import get_config
from data.datasets import MediumImagenetHDF5Dataset
#import argparse

def main():

    mimgnet = MediumImagenetHDF5Dataset(img_size=96)


    imgs, labels = [], []

    for i in range(5):
        img, lab = mimgnet.__getitem__(i)
        imgs.append(img)
        labels.append(lab)

    grid = utils.make_grid(imgs, nrow=5, normalize=True, padding=2)

    utils.save_image(grid, '5_images_grid.png')
    print("Saved 5 images grid as 5_images_grid.png")


if __name__ == "__main__":
    main()