'''
求取训练集数据的均值和标准差
'''

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

'''
验证代码正确性
CIFAR100   mean = tensor([0.5071, 0.4865, 0.4409]),std = tensor([0.2673, 0.2564, 0.2762])
CIFAR10    mean = tensor([0.4914, 0.4822, 0.4465]),std = tensor([0.2470, 0.2435, 0.2616])
MNIST      mean = 0.13066047430038452,std = tensor([0.3081])
'''
# train_dataset = datasets.MNIST(download=True, root='./', train=True, transform=transforms.ToTensor())

train_path = './Result_Split_data/train_Crop'
train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)


def get_mean_std_value(loader, List):

    '''shape [n_samples(batch), channels, height, width]'''
    for data, label in loader:
        List.append(data)

    Data = torch.cat(List, dim=0)
    mean = torch.mean(Data, dim=[0, 2, 3], keepdim=True)
    std = torch.mean((Data-mean)**2, dim=[0, 2, 3])**0.5

    return mean.squeeze(), std.squeeze()

mean, std = get_mean_std_value(train_loader, List=[])
print('mean = {},std = {}'.format(mean, std))

'''
输出结果
mean = 0.4521771967411041,std = 0.16110114753246307
'''