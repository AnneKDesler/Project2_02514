import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
import glob
import numpy as np
import PIL.Image as Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class DRIVE(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02514/DRIVE/training'):
        'Initialization'
        self.transform = transform
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
        
        image = np.array(Image.open(image_path),dtype=np.uint8)
        mask = np.array(Image.open(mask_path),dtype=np.uint8)
        mask = mask//255
        manual = np.array(Image.open(manual_path),dtype=np.uint8)
        manual = manual//255
        combined = np.concatenate((manual[...,None], mask[...,None]), axis=-1)
        
        transformed = self.transform(image=image, manual=manual, mask=mask)
        X = transformed["image"]
        Y = transformed["manual"]
        Z = transformed["mask"]
        return X, Y, Z
    
def get_dataloaders_DRIVE(batch_size, num_workers=8, seed=42, data_path="/dtu/datasets1/02514/DRIVE/training"):

    data_transform_val = A.Compose(
        [
            A.PadIfNeeded(min_height=576, min_width=576),
            A.CenterCrop(576, 576),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ],
        additional_targets={'manual': 'mask'}
    )
    data_transform_train = A.Compose(
        [
            A.PadIfNeeded(min_height=576, min_width=576),
            A.CenterCrop(576, 576),
            A.Normalize(mean=0.5, std=0.5),
            A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.VerticalFlip(p=0.5),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.75),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0, p=1.0),
            ToTensorV2(),
        ],
        additional_targets={'manual': 'mask'}
    )
    
    trainset = DRIVE(train=True, transform=data_transform_train, data_path=data_path)
    testset = DRIVE(train=False, transform=data_transform_val, data_path=data_path)

    generator1 = torch.Generator().manual_seed(seed)
    trainset, _, _ = random_split(trainset, [0.5, 0.25, 0.25], generator=generator1)
    generator1 = torch.Generator().manual_seed(seed)
    _, valset, testset = random_split(testset, [0.5, 0.25, 0.25], generator=generator1)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


class PH2(torch.utils.data.Dataset):
    def __init__(self, set_type, transform, split=[60,20,20], seed=42, data_path="/dtu/datasets1/02514/PH2_Dataset_images"):
        'Initialization'
        self.transform = transform
        allpaths = np.array(sorted(os.listdir(data_path)))
        
        split = np.array(split, dtype=np.float32)
        split *= len(allpaths)/np.sum(split)
        split = np.cumsum(split)
        
        np.random.seed(seed)
        random_ordering = np.random.permutation(len(allpaths))
        train_idx = random_ordering[:int(split[0])]
        val_idx = random_ordering[int(split[0]):int(split[1])]
        test_idx = random_ordering[int(split[1]):]
        
        if set_type.lower() == "train":
            allpaths = allpaths[train_idx]
        elif set_type.lower() in ["val", "validation"]:
            allpaths = allpaths[val_idx]
        elif set_type.lower() == "test":
            allpaths = allpaths[test_idx]
        else:
            raise AttributeError
        
        self.label_paths = []
        self.image_paths = []
        
        for i, name in enumerate(allpaths):
            self.image_paths.append(os.path.join(data_path, name, name+"_Dermoscopic_Image", name+".bmp"))
            self.label_paths.append(os.path.join(data_path, name, name+"_lesion", name+"_lesion.bmp"))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        mask_path = self.label_paths[idx]
        
        image = np.array(Image.open(image_path), dtype=np.uint8)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        
        transformed = self.transform(image=image, mask=mask)
        X = transformed["image"]
        Y = transformed["mask"]
        return X, Y


def get_dataloaders_PH2(batch_size, num_workers=8, seed=42, data_path="/dtu/datasets1/02514/PH2_Dataset_images"):
    
    data_transform_val = A.Compose(
        [
            A.Normalize(mean=0.5, std=0.5),
            A.PadIfNeeded(min_height=576, min_width=576),
            A.CenterCrop(576, 576),
            ToTensorV2(),
        ]
    )
    data_transform_train = A.Compose(
        [
            A.Normalize(mean=0.5, std=0.5),
            A.PadIfNeeded(min_height=576, min_width=576),
            A.CenterCrop(576, 576),
            A.Rotate(limit=45, p=1.0),
            A.VerticalFlip(p=0.5),
            A.GridDistortion(p=0.75),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0, p=1.0),
            ToTensorV2(),
        ]
    )
    
    trainset = PH2("train", data_transform_train, seed=seed, data_path=data_path)
    valset = PH2("val", data_transform_val, seed=seed, data_path=data_path)
    testset = PH2("test", data_transform_val, seed=seed, data_path=data_path)
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return trainloader, valloader, testloader


if __name__ == "__main__":
    
    torch.manual_seed(41)
    trainloader,valloader,testloader = get_dataloaders_PH2(batch_size=1)
    
    print(f"Lengths of train, val, test are: {len(trainloader)}, {len(valloader)}, {len(testloader)}")
    # for img, target in testloader:
    #     # shape roughly 575x767
    #     print(f"Img: {img.shape}, and target: {target.shape} ")
    
    import matplotlib.pyplot as plt
    
    img, target = list(trainloader)[0]
    img, target = img.numpy().transpose((0,3,2,1)).squeeze(), target.numpy().transpose((2,1,0)).squeeze()
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.savefig("Testing_PH2_img.png")
    plt.show()
    plt.figure()
    plt.imshow(target)
    plt.colorbar()
    plt.savefig("Testing_PH2_target.png")
    plt.show()
    
    train_loader, val_loader, test_loader = get_dataloaders_DRIVE(batch_size=1, seed=42)
    
    print('Loaded %d training images' % len(train_loader))
    print('Loaded %d test images' % len(test_loader))

    # IMG Size  565x584
    import matplotlib.pyplot as plt
    img, target, mask = list(train_loader)[0]
    img, target, mask = img.numpy().transpose((0,3,2,1)).squeeze(), target.numpy().transpose((2,1,0)).squeeze(), mask.numpy().transpose((2,1,0)).squeeze()
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.savefig("Testing_DRIVE_img.png")
    plt.show()
    plt.figure()
    plt.imshow(mask)
    plt.colorbar()
    plt.savefig("Testing_DRIVE_mask.png")
    plt.show()
    plt.figure()
    plt.imshow(target.squeeze())
    plt.colorbar()
    plt.savefig("Testing_DRIVE_target.png")
    plt.show()