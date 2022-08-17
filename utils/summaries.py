import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from .util import tensor2im


class TensorboardSummary:
    def __init__(self, opt):
        self.experiment_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.experiment_name)

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.experiment_dir))
        return writer

    def make_figure_grid(self, images):
        fig = plt.figure()
        # fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i, (label, img) in enumerate(images.items()):
            ax = fig.add_subplot(1, 3, i+1)
            ax.set_title(label)
            ax.axis('off')
            ax.imshow(tensor2im(img), cmap='gray')

        return fig

    def visualize_image(self, writer, images, global_step):
        fig = self.make_figure_grid(images)
        writer.add_figure(f'image', fig, global_step)
