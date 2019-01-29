import os
import torch
import numpy as np
import scipy.misc as m
import re
import glob

from torch.utils import data


class CELEBA(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=(32, 32), augmentations=None):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes =  40
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876]) # TODO(compute this mean)
        self.files = {}
        self.labels = {}

        self.label_file = self.root+"/Anno/list_attr_celeba.txt"
        label_map = {}
        with open(self.label_file, 'r') as l_file:
            labels = l_file.read().split('\n')[2:-1]
        for label_line in labels:
            f_name = re.sub('jpg', 'png', label_line.split(' ')[0])
            label_txt = list(map(lambda x:int(x), re.sub('-1','0',label_line).split()[1:]))
            label_map[f_name]=label_txt

        self.all_files = glob.glob(self.root+'/Img/img_align_celeba_png/*.png')
        with open(root+'//Eval/list_eval_partition.txt', 'r') as f:
            fl = f.read().split('\n')
            fl.pop()
            if 'train' in self.split:
                selected_files = list(filter(lambda x:x.split(' ')[1]=='0', fl))
            elif 'val' in self.split:
                selected_files =  list(filter(lambda x:x.split(' ')[1]=='1', fl))
            elif 'test' in self.split:
                selected_files =  list(filter(lambda x:x.split(' ')[1]=='2', fl))
            selected_file_names = list(map(lambda x:re.sub('jpg', 'png', x.split(' ')[0]), selected_files))
        
        base_path = '/'.join(self.all_files[0].split('/')[:-1])
        self.files[self.split] = list(map(lambda x: '/'.join([base_path, x]), set(map(lambda x:x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
        self.labels[self.split] = list(map(lambda x: label_map[x], set(map(lambda x:x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
        self.class_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                                'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',      
                                'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',       
                                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
                                'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
                                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        if len(self.files[self.split]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))

        print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        label = self.labels[self.split][index]
        img = m.imread(img_path)

        if self.augmentations is not None:
            img = self.augmentations(np.array(img, dtype=np.uint8))

        if self.is_transform:
            img = self.transform_img(img)

        return [img] + label

    def transform_img(self, img):
        """transform
        Mean substraction, remap to [0,1], channel order transpose to make Torch happy
        """
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt


    local_path = 'CELEB_A_PATH'
    dst = CELEBA(local_path, is_transform=True, augmentations=None)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)

    for i, data in enumerate(trainloader):
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])

        f, axarr = plt.subplots(bs,4)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
            axarr[j][2].imshow(instances[j,0,:,:])
            axarr[j][3].imshow(instances[j,1,:,:])
        plt.show()
        a = raw_input()
        if a == 'ex':
            break
        else:
            plt.close()
