import os
import argparse

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex as iou

from utils.Utils import *
from utils.TrainUtils import *
from unet.model.UNet import UNet
from utils.DataUtils import get_manual_data_loaders, get_oxford_data_loaders
from utils.ShowResults import ShowResults

ignore_index = 250

os.makedirs("./saves/", exist_ok=True)

"""
for fast testing:
python main.py --subset --subset_size=8 --epochs=1 (takes ~25 seconds)

for training:
python main.py --epochs=100 --verbose --save_models --device="cuda"
"""

# TODO: Update the README.md file with the following information:
# 1. how to use the command line arguments


def main(args):
    # load the dataset
    train_loader, val_loader = get_oxford_data_loaders(args)
    print(
        f"train_loader: {len(train_loader.dataset)}\tval_loader: {len(val_loader.dataset)}")

    # create the model
    resolution = train_loader.dataset[0][0].shape[-1]
    model = UNet(resolution=resolution, out_channels=args.num_classes,
                 base_channels=32, channel_mult=[1, 2, 4], num_blocks=4, bilinear=False, pooling=True)

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    metric = iou(num_classes=args.num_classes,
                 ignore_index=ignore_index).to(args.device)

    if args.show_model:
        print_module_summary(
            model, inputs=[torch.rand(2, 3, resolution, resolution)])

    # train the model
    dict_log = train(model, train_loader, val_loader, criterion, metric, args)

    # plot/show the results
    plot_stats(dict_log, modelname="Model Training",
               scale_metric=100, path="saves/train_plot")
    sr = ShowResults(args.num_classes)
    sr.show_preds(model, train_loader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple Implementation of a UNet")

    # general
    parser.add_argument("--device", type=str, default="cpu")

    # dataset specific
    parser.add_argument("--data_path", type=str, default="data/CityScapes")
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--subset_size", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=3)

    # training specific
    parser.add_argument("--exp_name", type=str, default="UNet")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--show_model", action="store_true")
    parser.add_argument("--save_models", action="store_true")

    # hyperparameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    args = parser.parse_args()

    main(args)

    print("Done!")
