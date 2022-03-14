

from torchvision import datasets
import os
import numpy as np
import random
import csv,json
import torch
import pdb

class ReIDFolder(datasets.ImageFolder):

    def __init__(self, root,bone_folder,mask_folder, transform):
        super(ReIDFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.bone_folder = bone_folder
        self.mask_folder = mask_folder
        self.targets = targets
        self.img_num = len(self.samples)
        print(self.img_num)

    def _get_cam_id(self, path):
        camera_id = []
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        return int(camera_id)-1

    def _get_pos_sample(self, target, index, path):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        if len(pos_index)==0:  # in the query set, only one sample
            return path
        else:
            rand = random.randint(0,len(pos_index)-1)
        return self.samples[pos_index[rand]][0]

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        return self.samples[neg_index[rand]][0]
    
    def load_bone_data(self, img_name, flip=False):
        bone_img = np.load(os.path.join(self.bone_folder, img_name + ".npy"))
        bone = torch.from_numpy(bone_img).float()  # h, w, c
        bone = bone.transpose(2, 0)  # c,w,h
        bone = bone.transpose(2, 1)  # c,h,w
        if flip:
            bone = bone.flip(dims=[-1])
        return bone

    def load_mask_data(self, img_name, flip=False):
        mask = torch.Tensor(np.load(os.path.join(self.mask_folder, img_name + ".npy")).astype(int))
        if flip:
            mask = mask.flip(dims=[-1])
        mask = mask.unsqueeze(0).expand(3, -1, -1)
        return mask

    #@staticmethod
    def load_key_points(annotations_file_path):
        with open(annotations_file_path, "r") as f:
            f_csv = csv.reader(f, delimiter=":")
            next(f_csv)
            annotations_data = {}
            for row in f_csv:
                img_name = row[0]
                key_points_y = json.loads(row[1])
                key_points_x = json.loads(row[2])
                annotations_data[img_name] = torch.cat([
                    torch.Tensor(key_points_y).unsqueeze_(-1),
                    torch.Tensor(key_points_x).unsqueeze_(-1)
                ], dim=-1)
            return annotations_data


    def __getitem__(self, index):
        #pdb.set_trace()
        path, target = self.samples[index]
        sample = self.loader(path)
        name = os.path.basename(path)
        #print(name)
        flip = False
        pos_path = self._get_pos_sample(target, index, path)
        neg_path = self._get_neg_sample(target)
        #print(pos_path)
        #print(neg_path)
        pos = self.loader(pos_path)
        neg = self.loader(neg_path)
       
        bone = self.load_bone_data(name, flip)
        mask = self.load_mask_data(name,flip)
        cam = self._get_cam_id(path)
        if self.transform is not None:
            sample = self.transform(sample)
            pos = self.transform(pos)
            neg = self.transform(neg)
        if self.target_transform is not None:
            target = self.target_transform(target)


        return sample, target, pos, neg, bone, mask, cam

