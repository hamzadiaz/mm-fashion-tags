import torch
import numpy as np


def load_torch_model(path, evaluate=True):
    model = torch.load(path)
    if evaluate:
        model.eval()
    return model


def decode_model_prediction(fine_or_coarse, pred, type):
    if fine_or_coarse == 'fine' and type == 'cate':
        type = 'Category'
        top_k = [1, 3]
        label_path = 'labels/category_fine.txt'
    elif fine_or_coarse == 'fine' and type == 'attr':
        type = 'Attribute'
        top_k = [3, 5]
        label_path = 'labels/attribute_fine.txt'
    elif fine_or_coarse == 'coarse' and type == 'attr':
        type = 'Attribute'
        top_k = [3, 5, 10]
        label_path = 'labels/attribute_coarse.txt'
    else:
        return
    if len(pred.shape) == 1:
        pred = torch.unsqueeze(pred, 0)
    pred = pred.data.cpu().numpy()
    attr_cloth_file = open(label_path).readlines()
    attr_idx2name = {}
    for i, line in enumerate(attr_cloth_file[2:]):
        attr_idx2name[i] = line.strip('\n').split()[0]

    indexes = np.argsort(pred[0])[::-1]
    for k in top_k:
        idxes = indexes[:k]
        print(f'[ Top%d {type} Prediction ]' % k)
        for idx in idxes:
            print(f'{attr_idx2name[idx].ljust(15)} \t\t{pred[0, idx]:.3f}')