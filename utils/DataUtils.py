import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np


def get_manual_data_loaders(args):
    data_path = args.data_path
    transform_imgs = transforms.Compose([transforms.Normalize(-1, 2),
                                        transforms.ColorJitter(
                                            0.2, 0.2, 0.2, 0.2),
                                        transforms.Normalize(0.5, 0.5),])

    train_dataset = CityscapesDownsampled(img_path=f"{data_path}/imgs_train.pth", label_path=f"{data_path}/labels_train.pth",
                                          transform=transform_imgs)
    val_dataset = CityscapesDownsampled(
        img_path=f"{data_path}/imgs_val.pth", label_path=f"{data_path}/labels_val.pth")

    # subset the datasets for faster testing time
    if args.subset:
        print("Created subset of dataset")
        train_dataset = CityscapesSubset(train_dataset, list(range(
            args.subset_size)), img_path=f"{data_path}/imgs_train.pth", label_path=f"{data_path}/labels_train.pth")
        val_dataset = CityscapesSubset(val_dataset, list(range(
            args.subset_size)), img_path=f"{data_path}/imgs_val.pth", label_path=f"{data_path}/labels_val.pth")
    # currently unused
    # test_dataset = CityscapesDownsampled(img_path=f"{data_path}/imgs_test.pth",label_path=f"{data_path}/labels_test.pth")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)

    return train_loader, val_loader


# Custom transformation to convert the mask image to class labels
class MaskToTensor(object):
    """
    For oxford dataset:
    0, 2 -> segment
    2 -> background
    """

    def __call__(self, mask: torch.tensor):
        new_mask = torch.zeros_like(mask, dtype=torch.long)
        unique_values = sorted(mask.unique())
        for i, value in enumerate(unique_values):
            if i in [0, 2]:
                c = 1
            else:
                c = 0
            new_mask[mask == value] = c
        return new_mask


def get_oxford_data_loaders(args):
    size = (300, 300)
    transform_imgs = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(-1, 2),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Normalize(0.5, 0.5),])

    transform_masks = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        MaskToTensor(),
    ])

    dataset = datasets.OxfordIIITPet(
        root="data/oxford_data",
        download=True,
        transform=transform_imgs,
        target_transform=transform_masks,
        split="trainval",
        target_types="segmentation"
    )

    if args.subset:
        print(f"Created subset of dataset of size {args.subset_size}")
        dataset = torch.utils.data.Subset(
            dataset, list(np.random.randint(0, len(dataset), args.subset_size)))

    train_length = int(len(dataset)*0.8)
    val_length = len(dataset) - train_length

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)

    return train_loader, val_loader


class CityscapesDownsampled(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, transform=None, target_transform=None):
        self.ignore_index = 250
        self.img_path = img_path
        self.label_path = label_path
        self.imgs = torch.load(img_path)
        self.labels = torch.load(label_path)
        self.transform = transform
        self.target_transform = target_transform
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                            'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
                            'train', 'motorcycle', 'bicycle']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index, ...]
        seg = self.labels[index, ...]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            seg = self.target_transform(seg)

        return img, seg


class CityscapesSubset(CityscapesDownsampled):
    def __init__(self, dataset, indices, **params):
        super().__init__(**params)
        self.train_dataset_test = torch.utils.data.Subset(dataset, indices)

    def __len__(self):
        return len(self.train_dataset_test)

    def __getitem__(self, index):
        img = self.imgs[index, ...]
        seg = self.labels[index, ...]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            seg = self.target_transform(seg)

        return img, seg
