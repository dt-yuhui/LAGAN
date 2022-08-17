import os
import sys
import torch
from tqdm import tqdm

from options import get_opt, print_options
from utils.summaries import TensorboardSummary
from models.la_module import ClassifyModel

from dataloaders import get_transform, is_image_file
from torch.utils import data
import cv2


def make_dataset(dataset_dir, phase, max_dataset_size=float('inf')):
    img_dir = os.path.join(dataset_dir, phase, 'img')
    images = []
    assert os.path.isdir(dataset_dir), f'{dataset_dir} is not a valid directory'

    for root, _, fnames in sorted(os.walk(img_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    print(f'Find {len(images)} images in {dataset_dir}')
    return images[:min(max_dataset_size, len(images))]


class ClassifyDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.path = make_dataset(opt.dataroot, opt.phase)
        self.size = len(self.path)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        image, label = self._make_img_gt_pair(index)

        transformed = self.transform(image=image)
        image = transformed['image']

        return {'image': image, 'label': label}

    def __len__(self):
        return self.size

    def _make_img_gt_pair(self, index):
        image_path = self.path[index]
        _image = cv2.imread(image_path, 0)
        if image_path.split('/')[-3] == 'A':
            _gt = 0
        else:
            _gt = 1

        return _image, _gt


def ClassifyDataloader(opt):
    dataset = ClassifyDataset(opt)
    loader = data.DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_threads, drop_last=True)

    return loader


def training_loop():
    opt, parser = get_opt()
    print_options(opt, parser)

    dataloader = ClassifyDataloader(opt)
    step_per_epoch = len(dataloader)
    model = ClassifyModel(opt)

    summary = TensorboardSummary(opt)
    writer = summary.create_summary()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        with tqdm(dataloader, desc=f'train_epoch_{epoch}', file=sys.stdout) as tbar:
            for i, data in enumerate(tbar):
                global_step = i + step_per_epoch * (epoch - 1)

                model.optimize_parameters(data)
                model.evaluate()

                losses = model.get_current_losses()
                # print training losses on tensorboard
                for loss_name, loss in losses.items():
                    writer.add_scalar(f'train/{loss_name}', loss, global_step)
        model.scheduler.step()

    model.save(opt.save_epoch)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True

    training_loop()
    print('Training was successfully finished')
