import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from os import listdir
from os.path import isfile, join

import sys
import os

from PIL import Image
import numpy as np

sys.path.append('../')
from external.googlenet.googlenet import googlenet


class CocoDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super(CocoDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.image_names = [f for f in listdir(self.img_dir) if isfile(join(self.img_dir, f))]

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        img = Image.open(join(self.img_dir, img_path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return (img_path, img)

    
if __name__ == "__main__":
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    resize_dim = 256
    crop_dim = 224
    batch_size = 50
    num_workers = 8

    transform_pipeline = transforms.Compose([transforms.Resize(resize_dim),
                                            transforms.CenterCrop(crop_dim),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    coco_dataset = CocoDataset(src_dir, transform=transform_pipeline)

    coco_loader = DataLoader(coco_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    
    model = googlenet(pretrained=True)
    model = model.cuda()
    model.eval()

    for i, (img_paths, images) in enumerate(coco_loader):
        images = images.cuda()
        output, _ = model(images)

        for j in range(len(img_paths)):
            feat_name = img_paths[j].replace('.jpg', '.npy')
            feat_name = join(dst_dir, feat_name)
            np.save(feat_name, output[j].cpu().data.numpy())
