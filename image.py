from PIL import Image
import numpy as np
import torch
from torchvision import transforms


def load_img_to_tensor(path, im_size=(256, 256)):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    if isinstance(path, list):
        img = np.zeros((len(path),3, im_size[0], im_size[1]))
        for i, p in enumerate(path):
            img_single = Image.open(p)
            img_single = img_single.resize(im_size)
            img_single = img_single.convert('RGB')
            img_single = transform(img_single)
            img[i] = img_single.numpy()
        img_tensor = torch.Tensor(img)

    else:
        img = Image.open(path)
        img = img.resize(im_size)
        img = img.convert('RGB')
        img_tensor = transform(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)

    return img_tensor