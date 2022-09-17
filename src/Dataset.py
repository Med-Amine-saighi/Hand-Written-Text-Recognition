import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
import torch
import numpy as np
import ast
import torch
import CFG
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.15),
    transforms.Resize(size=(CFG.IMAGE_HEIGHT,CFG.IMAGE_WIDTH)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(CFG.IMAGE_HEIGHT,CFG.IMAGE_WIDTH)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class ClassificationDataset:
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        image = Image.open(self.dataframe['image_path'][item]).convert("RGB")
        targets = self.dataframe['LABEL ENCODED'][item]
        if self.transform:
            image = self.transform(image)
        image = np.array(image)
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }