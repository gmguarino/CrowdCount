import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models import CSRNet
from dataset import ShanghaiTech
from utils import transforms

import sys

import os


# https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/train.py

CUDA = torch.cuda.is_available()
print("model not defined")

model = CSRNet(cfg_path="./cfg/cfg.json", from_weights=False)
print("model defined")
if CUDA:
    print("moving model to cuda")
    model.cuda()
    print("Done")


train_data = ShanghaiTech(
    roots=["./part_A_final", "./part_B_final"],
    train=True,
    tranforms=transforms
)


# print(train_data.__len__())
# im, target = train_data.__getitem__(700)
# density_map = target["map"]
# print(density_map.numpy().sum(), target['count'].item())
# plt.figure()

# plt.imshow(im.numpy().astype('int'))

# plt.figure()
# plt.imshow(density_map.numpy(), cmap='jet', interpolation="bilinear")
# plt.show()

EPOCHS = 50
BATCH_SIZE = 1

def main():

    trainloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    criterion = nn.MSELoss(reduction='none')
    if CUDA:
        criterion.cuda()

    optimizer = optim.SGD(
        params=model.parameters(),
        lr=1e-7,
        momentum=0.95,
        weight_decay=5*1e-4
    )
    print("starting training")

    for epoch in range(EPOCHS):
        for idx, (image, target) in enumerate(trainloader):
            print("getting batch")

            dmap = target["map"]
            count = target["count"]
            print(dmap.shape)

            if CUDA:
                dmap = dmap.cuda()
                count = count.cuda()
                image = image.cuda()

            out = model(image)
            loss = criterion(out, dmap)
            print(loss.shape)
            break
        break
if __name__ == '__main__':
    main()