
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
from os import path
import random
import numpy as np
from PIL import Image
import time
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import copy
from scipy.special import comb
from scipy.stats import beta
np.set_printoptions(suppress=True, precision=4, linewidth=65535)
import matplotlib.pyplot as plt

def expand_control_points(point_array):
    point_array_expand = copy.deepcopy(point_array)
    size = point_array.shape[1]
    assert size >= 3
    for i in range(1,size-3):
        p0, p1, p2 = point_array[:,i], point_array[:,i+1], point_array[:,i+2]
        norm1, norm2 = np.linalg.norm(p0-p1), np.linalg.norm(p2-p1)
        pc = p1 - 0.5*np.sqrt(norm1*norm2)*((p0-p1)/norm1 + (p2-p1)/norm2)
        point_array_expand[:,i+1] = pc
    return point_array_expand

def bernstein(t, i, n):
    return comb(n,i) * t**i * (1-t)**(n-i)

def bezier_curve(t, point_array, bias=0):
    t = np.clip(t, 0, 1)
    n = point_array.shape[1]-1
    p = np.array([0.,0.]).reshape(2,1)
    size = len(t) if isinstance(t, np.ndarray) else 1
    p = np.zeros((2, size))
    new_point_array = np.diff(point_array, n=bias, axis=1)
    for i in range(n+1-bias):
        p += new_point_array[:,i][:,np.newaxis] * bernstein(t, i, n-bias) * n**bias
    return p


class Bezier(object):
    def __init__(self, time_list, x_list, y_list, v0, vf=(0.000001,0.000001)):
        t0, x0, y0 = time_list[0], x_list[0], y_list[0]
        t_span = time_list[-1] - time_list[0]
        time_array = np.array(time_list)
        x_array, y_array = np.array(x_list), np.array(y_list)
        time_array -= t0
        x_array -= x0
        y_array -= y0
        time_array /= t_span

        point_array = np.vstack((x_array, y_array))
        n = point_array.shape[1]+1
        v0, vf = np.array(v0), np.array(vf)
        p0 = point_array[:, 0] + v0/n
        pf = point_array[:,-1] - vf/n

        point_array = np.insert(point_array, 1, values=p0, axis=1)
        point_array = np.insert(point_array,-1, values=pf, axis=1)

        point_array_expand = expand_control_points(point_array)

        self.t0, self.t_span = t0, t_span
        self.x0, self.y0 = x0, y0
        self.p0 = np.array([x0, y0]).reshape(2,1)
        self.point_array = point_array
        self.point_array_expand = point_array_expand
    

    def position(self, time, expand=True):
        time = np.clip(time, self.t0, self.t0+self.t_span)
        t = (time - self.t0) / self.t_span
        p = self.point_array_expand if expand else self.point_array
        position = bezier_curve(t, p, bias=0)
        return position + self.p0
    
    def velocity(self, time, expand=True):
        time = np.clip(time, self.t0, self.t0+self.t_span)
        t = (time - self.t0) / self.t_span
        p = self.point_array_expand if expand else self.point_array
        return bezier_curve(t, p, bias=1)
    
    def acc(self, time, expand=True):
        time = np.clip(time, self.t0, self.t0+self.t_span)
        t = (time - self.t0) / self.t_span
        p = self.point_array_expand if expand else self.point_array
        return bezier_curve(t, p, bias=2)
    
def angle_normal(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

def xy2uv(x, y):
    pixs_per_meter = 200./25.
    u = (200-x*pixs_per_meter).astype(int)
    v = (y*pixs_per_meter+400//2).astype(int)
    #mask = np.where((u >= 0)&(u < 200))[0]
    return u, v

def search(l,item):
    length = len(l)
    for i in range(length):
        if l[i]>=item:
            return i
    # print('Not Found')
    return -1

def check_nav_valid(img):
    sum_white = len(img[img==255])
    return sum_white<205000
    
def get_filelist(path, index:int):
    files = glob.glob(path+'/'+str(index)+'/ipm/*.png')
    file_names = []
    for file in files:
        file_name = file.split('/')[-1][:-4]
        # check for img pic validity
        # if cv2.imread(path+'/'+str(index)+'/img/'+file_name+'.png') is None:
        #     print('not found:',file_name)
        #     continue
        # # check for nav pic validity
        # pic = cv2.imread(path+'/'+str(index)+'/nav/'+file_name+'.png')
        # if not check_nav_valid(pic):
        #     continue

        file_names.append(file_name)
    file_names.sort()
    return file_names

def spin(xy,deg):
    x,y = xy
    rad = deg/180*np.pi

    # rotation equation:
    # [x] = [cost -sint][x']
    # [y] = [sint  cost][y']
    
    x_ = x*np.cos(rad)-y*np.sin(rad)
    y_ = x*np.sin(rad)+y*np.cos(rad)
    # res = ()
    return [x_,y_]

def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0


class CARLADataset(Dataset):
    def __init__(self, data_index, dataset_path, eval_mode=False):
        self.data_index = data_index
        self.eval_mode = eval_mode
        img_height = 128
        img_width = 256
        
        label_transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]

        seg_transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        
        img_transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        nav_transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.RandomRotation(30, interpolation=Image.BICUBIC, expand=False),
            # transforms.RandomHorizontalFlip(0.1),
            # transforms.RandomVerticalFlip(0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        self.label_transforms = transforms.Compose(label_transforms)
        self.seg_transforms = transforms.Compose(seg_transforms)
        self.img_transforms = transforms.Compose(img_transforms)
        self.nav_transforms = transforms.Compose(nav_transforms)
        
        self.dataset_path = dataset_path
        self.files_dict = {}
        self.total_len = 0
        self.turning_list = {}
        
        for index in self.data_index:
            # self.files_dict[index] = get_filelist(self.dataset_path, index)
            self.read_img(index)
            # print(len(self.files_dict[index]))
            yaw_list = self.read_pos(index)[1]
            # print(len(yaw_list))

            self.turning_list[index]=self.find_turning(yaw_list)[0]
    
    def read_img(self, index):
        files = glob.glob(self.dataset_path+str(index)+'/pm/*.png')
        file_names = []
        for file in files:
            file_name = file.split('/')[-1][:-4]
            # check for nav pic validity
            file_names.append(file_name)
        file_names.sort()
        self.files_dict[index] = file_names

    def check_nav_valid(self, img):
        sum_white = len(img[img==255])
        return sum_white<205000

    def read_pos(self,index):
        xy_list = []
        yaw_list = []
        file_path = self.dataset_path+str(index)+'/state/pos.txt'
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                x = float(sp_line[1])
                y = float(sp_line[2])
                yaw = float(sp_line[5])
                xy_list.append(np.array([x,y]))
                yaw_list.append(yaw)
        return xy_list, yaw_list

    def find_turning(self, yaw_list, yaw_threshold=1.5):
        turning_offset = 0 # 转弯开始对yaw角变化的提前量
        
        # 列表转np数组
        yaw_array = np.array(yaw_list)

        # 使用NumPy的差分函数计算偏航角变化率
        diff_yaw = np.abs(np.diff(yaw_array))

        # 根据阈值，将偏航角变化率超过阈值的位置视为转弯点，下标值存入列表
        # 防止 180 附近突变
        turning_indexes = np.where((diff_yaw < 10) & (diff_yaw >= yaw_threshold))[0] - turning_offset # np数组

        # 将转弯点 time 时间戳存入列表
        # turning_list = [yaw_list[i] for i in turning_indexes]

        # 直行点下标
        # straight_indexes = np.array([i for i in range(len(yaw_list)) if i not in turning_indexes]) # np数组
        
        # 将直行点 time 时间戳存入列表
        # straight_list = [yaw_list[i] for i in straight_indexes]
        
        # return turning_indexes, straight_indexes, turning_list, straight_list
        return turning_indexes, None


    def __getitem__(self, index):
        data_index = random.sample(self.data_index, 1)[0]
        if random.uniform(0.0, 1.0) < 0.6: # Choosing Turning
            time_point_index = random.choice(self.turning_list[data_index])
            # print('time_point_index:',time_point_index)
            time_point_index += random.randint(-100,25)
        else: 
            time_point_index = random.randint(0, len(self.files_dict[data_index])-1)
        time_point_index = min(max(time_point_index,0),len(self.files_dict[data_index])-1)

        file_name = self.files_dict[data_index][time_point_index]
        fake_file_name = random.sample(self.files_dict[data_index], 1)[0]

        img_path = self.dataset_path + str(data_index)+'/img/'+file_name+'.png'
        img = Image.open(img_path).convert("RGB")
        # nav
        nav_path = self.dataset_path + str(data_index)+'/nav/'+file_name+'.png'
        nav = Image.open(nav_path).convert("RGB")

        fake_nav_path = self.dataset_path + str(data_index)+'/nav/'+fake_file_name+'.png'
        fake_nav = Image.open(fake_nav_path).convert("RGB")
        # label
        label_path = self.dataset_path + str(data_index)+'/pm/'+file_name+'.png'
        label = Image.open(label_path).convert('L')
        # seg
        # print('3')

        seg_path = self.dataset_path + str(data_index)+'/segimg/'+file_name+'.png'
        seg = Image.open(seg_path).convert('RGB')
            

        # mirror the inputs
        mirror = True if random.uniform(0.0, 1.0) > 0.5 else False
        if mirror:
            img = Image.fromarray(np.array(img)[:, ::-1, :], 'RGB')
            nav = Image.fromarray(np.array(nav)[:, ::-1, :], 'RGB')
            label = Image.fromarray(np.array(label)[:, ::-1], 'L')
            seg = Image.fromarray(np.array(seg)[:, ::-1], 'RGB')
            #     break

            # except:
            #     pass

        
        img = self.img_transforms(img)
        nav = self.nav_transforms(nav)
        fake_nav = self.nav_transforms(fake_nav)
        label = self.label_transforms(label)
        seg = self.label_transforms(seg)
            
        if not self.eval_mode:
            
            input_img = torch.cat((img, nav), 0)
            input_seg = torch.cat((seg, nav), 0)

            fake_input_img = torch.cat((img, fake_nav), 0)
            return {'img_nav': input_img, 'label': label, 'fake_nav_with_img':fake_input_img, 'seg_nav':input_seg}
        else:
            return {'img': img, 'nav': nav, 'fake_nav':fake_nav, 'label': label, 'file_name':file_name,'seg':seg}

    def __len__(self):
        return 160000


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    random.seed(datetime.now())
    torch.manual_seed(999)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="mu-log_var-test", help='name of the dataset')
    parser.add_argument('--width', type=int, default=400, help='image width')
    parser.add_argument('--height', type=int, default=200, help='image height')
    parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
    parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
    parser.add_argument('--max_dist', type=float, default=25., help='max distance')
    parser.add_argument('--max_t', type=float, default=3., help='max time')
    parser.add_argument('--beta1', type=float, default=1, help='beta parameter')
    parser.add_argument('--beta2', type=float, default=1, help='beta parameter')
    opt = parser.parse_args()

    test_loader = DataLoader(CARLADataset([21], dataset_path='../datacollect/DATASET/CARLA/Segmentation/', eval_mode=True),
                         batch_size=1, shuffle=False, num_workers=1, pin_memory=True,persistent_workers=True)

    cnt = 0
    for i, batch in enumerate(test_loader):
        # return {'img': img, 'nav': nav, 'fake_nav':fake_nav, 'label': label, 'file_name':file_name,'seg':seg}
        img = batch['img'].clone().data.numpy().squeeze()*127+128
        nav = batch['nav'].clone().data.numpy().squeeze()*127+128
        
        img = np.transpose(img,(1,2,0))
        nav = np.transpose(nav,(1,2,0))

        # print(img.shape)
        print('file_name',batch['file_name'])
        # img = Image.fromarray(np.transpose(img, (2, 1, 0)).astype('uint8')).convert("RGB")
        # nav = Image.fromarray(np.transpose(nav, (2, 1, 0)).astype('uint8')).convert("RGB")

        cv2.imwrite('./img_%d.jpg'%cnt,img)
        cv2.imwrite('./nav_%d.jpg'%cnt,nav)



        cnt+=1
        if cnt >= 5:
            break