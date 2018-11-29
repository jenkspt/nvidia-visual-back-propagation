from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from skimage import data, transform
from PIL import Image
import matplotlib.pyplot as plt

from visualbackprop import ResnetVisualizer

class FilenameDataset(Dataset):

    def __init__(self, files, transform=None):
        self.files = list(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            return self.transform(sample)
        return sample

if __name__ == "__main__":
    
    files = ['media/snake.png','media/truck.png', 'media/monkey.png']
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    with open('imagenet1000_clsid_to_human.txt', 'r') as f:
        classes = f.read().splitlines()

    ds = FilenameDataset(files, transform)
    loader = DataLoader(ds)

    model = resnet50(pretrained=True).eval()
    model_vis = ResnetVisualizer(model.eval())
    
    fig, axs = plt.subplots(3,3)
    vbp_files = ['media/snake-vbp.png', 'media/truck-vbp.png', 'media/monkey-vbp.png'] 
    for i, img, vbp_file in zip(range(3), loader, vbp_files):

        vbp = Image.open(vbp_file).convert('RGB')
        with torch.no_grad():
            x, vis = model_vis(img)
            x2 = model(img)

        class_pred = ' or '.join([classes[i] for i in x[0].topk(3)[1]])
        axs[i,1].set_title(class_pred)

        vis = vis[0].numpy().transpose(1,2,0)[:,:,0]
        vis = np.interp(vis, [vis.min(), vis.max()], [0,1])

        img = img[0].numpy().transpose(1,2,0)
        img = np.interp(img, [img.min(), img.max()], [0,1])
        
        axs[i,0].imshow(img, interpolation='bilinear')
        axs[i,1].imshow(vis, cmap='bwr', interpolation='bilinear')
        axs[i,2].imshow(vbp, interpolation='bilinear')

        axs[i,0].axis('off')
        axs[i,1].axis('off')
        axs[i,2].axis('off')
    plt.show()
