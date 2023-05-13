'''
将文件夹Magnetic-tile-defect-datasets.-master划分为数据集

训练集：验证集：测试集=6:2:2
'''

import os
import random
import torch
import glob
import numpy as np
from shutil import copy, rmtree

def mk_file(file_path: str):
    if os.path.exists(file_path):
        '''如果文件夹存在，则先删除原文件夹再重新创建'''
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    '''保证随机可复现'''
    np.random.seed(2022)

    '''将数据集中60%的数据划分到训练集中;将数据集中20%的数据划分到验证集中'''
    train_split_rate = 0.6
    val_split_rate = 0.2

    '''指向未数据划分处理的文件夹'''
    origin_path = './Magnetic-tile-defect-datasets.-master'
    assert os.path.exists(origin_path), "path '{}' does not exist.".format(origin_path)

    defect_class = [cla for cla in os.listdir(origin_path) if os.path.isdir(os.path.join(origin_path, cla))]

    save_path = './Result_Split_data'
    for p in ['train', 'val', 'test']:
        '''建立保存的文件夹'''
        path = os.path.join(save_path, p)
        for cla in defect_class:
            '''建立每个类别对应的文件夹'''
            mk_file(os.path.join(path, cla))


    for cla in defect_class:
        cla_path = os.path.join(origin_path, cla)
        images = glob.glob(cla_path+'/Imgs/'+'*.jpg')
        images.sort()

        '''随机采样数据索引'''
        # eval_index = random.sample(images, k=int(num*split_rate))
        np.random.shuffle(images)
        num = len(images)
        train_num = int(num*train_split_rate)
        val_num = int(num*val_split_rate)
        train_index = images[:train_num]
        eval_index = images[train_num:train_num+val_num]
        # test_index = images[train_num+val_num:]

        for index, image_path in enumerate(images):
            if image_path in train_index:
                '''将分配至训练集中的文件复制到相应目录'''
                new_path = os.path.join(save_path, 'train', cla)
                img_path_png = image_path.split('jpg')[0]+'png'
                copy(image_path, new_path)
                copy(img_path_png, new_path)
            elif image_path in eval_index:
                '''将分配至验证集中的文件复制到相应目录'''
                new_path = os.path.join(save_path, 'val', cla)
                img_path_png = image_path.split('jpg')[0] + 'png'
                copy(image_path, new_path)
                copy(img_path_png, new_path)
            else:
                '''将分配至测试集中的文件复制到相应目录'''
                new_path = os.path.join(save_path, 'test', cla)
                img_path_png = image_path.split('jpg')[0] + 'png'
                copy(image_path, new_path)
                copy(img_path_png, new_path)

                print("\r[{}] processing [{}/{}]".format(cla, 2*(index + 1), 2*len(images)), end="")  # processing bar

        print("\nprocessing done!")


if __name__ == '__main__':
    main()