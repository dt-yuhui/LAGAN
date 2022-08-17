import os
from torch.utils import data
from dataloaders import get_transform, make_dataset
import cv2


class CustomDataset(data.Dataset):
    def __init__(self, opt, name):
        super().__init__()
        self.opt = opt
        self.name = name
        self.phase = opt.phase
        self.path = make_dataset(opt, mode=name)
        if name == 'B':
            self.size = len(self.path[0])
        else:
            self.size = len(self.path)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # get abnormal ones
        if self.name == 'B':
            image, gt = self._make_img_gt_pair(index)
            transformed = self.transform(image=image, mask=gt)
            image, gt = transformed['image'], transformed['mask']
            return {'image': image, 'mask': gt, 'image_path': self.path[0][index]}
        # get normal ones
        else:
            image_path = self.path[index % self.size]
            image = cv2.imread(image_path, 0)
            transformed = self.transform(image=image)
            image = transformed['image']
            return {'image': image, 'image_path': image_path}

    def _make_img_gt_pair(self, index):
        _image = cv2.imread(self.path[0][index], 0)
        _gt = cv2.imread(self.path[1][index], 0)

        return _image, _gt

    def __len__(self):
        return self.size


def data_sample(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def CustomDataloader(opt, name):
    dataset = CustomDataset(opt, name)
    if opt.phase == 'train':
        sampler = data_sample(dataset, shuffle=True)
    else:
        sampler = data_sample(dataset, shuffle=False)
    loader = data.DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler, num_workers=opt.num_threads, drop_last=True)

    return loader, sampler


def yield_data(loader, sampler, distributed=False):
    epoch = 0
    while True:
        if distributed:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
