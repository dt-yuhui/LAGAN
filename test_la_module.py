import os
import torch
import numpy as np
from options import get_opt
from tqdm import tqdm
from models.la_module import ClassifyModel
from train_la_module import ClassifyDataloader


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt, parser = get_opt()
    # hard-code some parameters for test

    Model = ClassifyModel(opt)
    Model.load(opt.save_epoch)
    model = Model.model
    print(f'load the pretrained {opt.model} successfully')

    dataloader = ClassifyDataloader(opt)

    total = len(dataloader)
    correct = 0
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader) as tbar:
            for i, data in enumerate(tbar):
                img = data['image'].to('cuda')
                gt = data['label'].item()
                pred_logit = model(img)
                pred_label = np.argmax(pred_logit.cpu().data.numpy(), axis=-1)
                if gt == pred_label:
                    correct += 1

    print(f'acc: {correct / total}, correct: {correct}, total: {total}')


