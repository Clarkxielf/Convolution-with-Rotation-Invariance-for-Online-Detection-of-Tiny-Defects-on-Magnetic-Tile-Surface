import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def data_loader(path, batch_size=256, workers=0, pin_memory=True):
    traindir = os.path.join(path, 'train_Crop')
    valdir = os.path.join(path, 'val_Crop')
    testdir = os.path.join(path, 'test_Crop')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = 0.4521771967411041,std = 0.16110114753246307),
            transforms.RandomHorizontalFlip(),
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = 0.4521771967411041,std = 0.16110114753246307)
        ])
    )
    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = 0.4521771967411041,std = 0.16110114753246307)
        ])
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

# train_loader, _, _ = data_loader('./Data_processing/Result_Data_split')
#
# for i, (input, target) in enumerate(train_loader):
#     print()