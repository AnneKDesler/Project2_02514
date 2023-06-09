import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import glob
import PIL.Image as Image
from torch.utils.data import random_split
import numpy as np

data_path = '/dtu/datasets1/02514/DRIVE/'
class DRIVE(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path=data_path):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'training' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
        self.mask_paths = sorted(glob.glob(data_path + '/mask/*.gif'))
        self.manual_paths = sorted(glob.glob(data_path + '/1st_manual/*.gif'))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        manual_path = self.manual_paths[idx]
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        manual = Image.open(manual_path)
        Y = self.transform(np.array(manual))
        X = self.transform(image)
        return X, Y

train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.5, std=0.5)])
train_transform_augmented = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)])
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.5, std=0.5)])

trainset = DRIVE(train=True, transform=train_transform)
trainset_augmented = DRIVE(train=True, transform=train_transform_augmented)
testset = DRIVE(train=False, transform=test_transform)

generator1 = torch.Generator().manual_seed(42)
trainset_new, valset_new, testset_new = random_split(trainset, [0.5, 0.25, 0.25], generator=generator1)
trainset_new_augmented, _, _ = random_split(trainset, [0.5, 0.25, 0.25], generator=generator1)

batch_size = 1
train_loader = DataLoader(trainset_new_augmented, batch_size=batch_size, shuffle=True, num_workers=3)
test_loader = DataLoader(testset_new, batch_size=batch_size, shuffle=False, num_workers=3)
val_loader = DataLoader(valset_new, batch_size=batch_size, shuffle=False, num_workers=3)

if __name__ == "__main__":
    print('Loaded %d training images' % len(trainset))
    print('Loaded %d test images' % len(testset))

# IMG Size  565x584
    import matplotlib.pyplot as plt
    img, target = list(train_loader)[0]
    img, target = img.numpy().squeeze().transpose((1,2,0)), target.numpy().squeeze()
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.savefig("Testing_DRIVE_img.png")
    plt.show()
    plt.figure()
    plt.imshow(target.squeeze())
    plt.colorbar()
    plt.savefig("Testing_DRIVE_target.png")
    plt.show(block=True)