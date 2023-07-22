import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms


class ShowResults:
    def __init__(self, n_classes, n_results=1) -> None:
        self.n_results = n_results

        self.n_classes = n_classes
        self.colors = [[0,   0,   0],
                       [70, 143, 67],
                       [72, 179, 189],
                       [70, 70, 70],
                       [102, 102, 156],
                       [190, 153, 153],
                       [153, 153, 153],
                       [250, 170, 30],
                       [220, 220, 0],
                       [107, 142, 35],
                       [152, 251, 152],
                       [0, 130, 180],
                       [220, 20, 60],
                       [255, 0, 0],
                       [0, 0, 142],
                       [0, 0, 70],
                       [0, 60, 100],
                       [0, 80, 100],
                       [0, 0, 230],
                       [119, 11, 32]]

        # only use self.n_classes colors
        self.label_colours = dict(zip(range(self.n_classes), self.colors))

    def show_preds(self, model, loader, args, ignore_index=250):
        model.eval()
        model = model.to(args.device)
        imgs, segs = next(iter(loader))
        preds = model(imgs.to(args.device))
        num_samples = min(self.n_results, int(imgs.shape[0]))
        for img_id in range(num_samples):
            pred = preds.argmax(dim=1)[img_id].cpu()
            img, seg = imgs[img_id], segs[img_id].squeeze()
            pred[seg == ignore_index] = ignore_index
            # Visualizes the three arguments in a plot
            self.plot_triplet(img, seg, pred)

    def plot_triplet(self, img, seg, pred):
        """
        shows a triplet of: image + ground truth + predicted segmentation
        """
        plt.subplots(ncols=3, figsize=(18, 10))
        plt.subplot(131)
        plt.axis("off")
        plt.title("Original Image")
        self.img_show(img, mean=torch.tensor([0.5]), std=torch.tensor([0.5]))
        plt.subplot(132)
        plt.axis("off")
        plt.title("Ground Truth")
        self.seg_show(seg)
        plt.subplot(133)
        plt.axis("off")
        plt.title("Predicted")
        self.seg_show(pred)
        plt.show()

    def img_show(self, img, mean, std):
        """
        shows an image on the screen.
        mean of 0 and variance of 1 will show the images unchanged in the screen
        """
        unnormalize = transforms.Normalize(
            (-mean / std).tolist(), (1.0 / std).tolist())
        npimg = unnormalize(img).numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def seg_show(self, seg):
        """
        shows an image on the screen. mean of 0 and variance of 1 will show the images unchanged in the screen
        """
        seg = self.decode_segmap(seg.squeeze_())
        plt.imshow(seg)

    def decode_segmap(self, temp):
        # convert gray scale to color
        temp = temp.numpy()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        # 19npossible colors
        for l in range(self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
