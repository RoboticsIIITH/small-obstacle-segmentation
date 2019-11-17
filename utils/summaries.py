import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence,decode_segmap
import numpy as np
from tensorboardX.utils import figure_to_image
from matplotlib import pyplot as plt

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step,num_image=3):
        grid_image = make_grid(image[:num_image].clone().cpu().data, num_image, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:num_image], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), num_image, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:num_image], 1).detach().cpu().numpy(),
                                                       dataset=dataset), num_image, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)


    def vis_grid(self, writer, dataset, image, target, pred, pred_softmax, global_step, split):
        image = image.squeeze()
        image = image.astype(np.uint8)
        image = np.moveaxis(image, 0, 2)
        target = target.squeeze()
        seg_mask = target == 2
        target = decode_segmap(target, dataset=dataset)
        pred = pred.squeeze()
        pred_softmax = np.max(pred_softmax.squeeze(), axis=0)
        pred_softmax = seg_mask * pred_softmax
        pred = decode_segmap(pred, dataset=dataset).squeeze()

        fig = plt.figure(figsize=(7, 20), dpi=150)
        ax1 = fig.add_subplot(411)
        ax1.imshow(image)
        ax2 = fig.add_subplot(412)
        ax2.imshow(pred)
        ax3 = fig.add_subplot(413)
        ax3.imshow(target)
        ax4 = fig.add_subplot(414)
        ax4.imshow(pred_softmax, cmap='plasma')
        writer.add_image(split, figure_to_image(fig), global_step)
        plt.clf()