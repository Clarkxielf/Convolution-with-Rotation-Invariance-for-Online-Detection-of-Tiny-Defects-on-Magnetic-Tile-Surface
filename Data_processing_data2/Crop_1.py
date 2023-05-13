'''
对采集的图片裁剪到一致大小
'''
import cv2
import os
import glob
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

path = './Result_Split_data/test'
# for p in ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Free', 'MT_Uneven']:
for p in ['MT_Blowhole', 'MT_Crack', 'MT_Free']:

    '''新建文件夹'''
    save_path = os.path.join(path+'_Crop', p)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    '''读取采集的图片'''
    Img_path_png = glob.glob(path+'/'+p+'/*.png')
    Img_path_png.sort()

    delete_path = []
    if p != 'MT_Free':
        for img_path_png in Img_path_png:
            img = cv2.imread(img_path_png)
            if img.sum()<255*2*2*3:
                delete_path.append(img_path_png)

    for dp in delete_path:
        Img_path_png.remove(dp)

    # '''train'''
    # num = 2000
    # if len(Img_path_png)>num:
    #     random.shuffle(Img_path_png)
    #     Img_path_png = Img_path_png[:num]
    # else:
    #     Img_path_png = Img_path_png*(num//len(Img_path_png))
    #     Img_path_png += Img_path_png[:num-len(Img_path_png)]


    for i, img_path_png in enumerate(Img_path_png):
        img = cv2.imread(img_path_png)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()

        img_h, img_w = img.shape[:2]
        H, W = 64, 64
        if p != 'MT_Free':
            top_left_x, top_left_y = 0, 0
            while img[top_left_x:top_left_x+H, top_left_y:top_left_y+W].sum()<255*2*2*3 or img[top_left_x:top_left_x+H, top_left_y:top_left_y+W].sum()>255*32*32*3:
                top_left_x = int(np.random.randint(0, img_h - H, 1))
                top_left_y = int(np.random.randint(0, img_w - W, 1))
        else:
            top_left_x = int(np.random.randint(0, img_h - H, 1))
            top_left_y = int(np.random.randint(0, img_w - W, 1))

        '''对png图片进行裁剪'''
        # save_img = img[top_left_x:top_left_x+H, top_left_y:top_left_y+W]
        # plt.imshow(save_img)
        # plt.axis('off')
        # plt.imsave(save_path +'/{}_'.format(0+i)+ img_path_png.split('/')[-1], save_img)
        # plt.show()

        '''png图片指导jpg图片裁剪'''
        img_path_jpg = img_path_png.split('png')[0]+'jpg'
        img = cv2.imread(img_path_jpg)

        save_img = img[top_left_x:top_left_x+H, top_left_y:top_left_y+W]
        # plt.imshow(save_img)
        # plt.axis('off')
        plt.imsave(save_path +'/{}_'.format(0+i)+ img_path_jpg.split('/')[-1], save_img)
        # plt.show()

        print("\r[{}] processing [{}/{}]".format(p, i + 1, len(Img_path_png)), end="")  # processing bar
    print()

    print("processing done!")