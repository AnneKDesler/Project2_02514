import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split
import os
from src import _DATA_PATH
import numpy as np
import PIL.Image as Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


data_path = '/dtu/datasets1/02514/phc_data'
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


def get_dataloaders(batch_size, num_workers=8, seed=42, data_path="/dtu/datasets1/02514/PH2_Dataset_images"):
    
    data_transform_val = A.Compose(
        [
            A.Normalize(mean=0.5, std=0.5),
            A.CenterCrop(570, 760),
            ToTensorV2(),
        ]
    )
    data_transform_train = A.Compose(
        [
            A.Normalize(mean=0.5, std=0.5),
            A.CenterCrop(570, 760),
            A.Rotate(limit=45, p=1.0),
            A.VerticalFlip(p=0.5),
            A.GridDistortion(p=1.0),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=1.0),
            ToTensorV2(),
        ]
    )
    
    trainset = PH2("train", data_transform_train, seed=seed, data_path=data_path)
    valset = PH2("val", data_transform_val, seed=seed, data_path=data_path)
    testset = PH2("test", data_transform_val, seed=seed, data_path=data_path)
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, num_workers=num_workers
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers
    )

    return trainloader, valloader, testloader


if __name__ == "__main__":
    
    torch.manual_seed(41)
    trainloader,valloader,testloader = get_dataloaders(batch_size=1)
    
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
    plt.show(block=True)