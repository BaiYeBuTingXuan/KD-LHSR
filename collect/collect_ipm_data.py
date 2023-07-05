import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import simulator
simulator.load('/data2/wanghejun/CARLA_0.9.9')
from simulator import config
import carla
import numpy as np
import argparse
import time
import cv2
from tqdm import tqdm

from collect_pm_data import sensor_dict
from ff.collect_ipm import InversePerspectiveMapping
from ff.carla_sensor import Sensor, CarlaSensorMaster


longitudinal_length = 25.0 # 纵向长度，开25m？

# for GaussianBlur
ksize = 21

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-n', '--num', type=int, default=100000, help='total number')
args = parser.parse_args()
data_index = args.data

save_path = './DATASET/CARLA/NoTrafficLightNoNPC/'+str(data_index)+'/'

def mkdir(path):
    if not os.path.exists(save_path+path):
        os.makedirs(save_path+path)
        
mkdir('ipm/')

def read_pm_time_stamp(dir_path):
    img_name_list = os.listdir(dir_path)
    time_stamp_list = []
    for img_name in img_name_list:
        time_stamp_list.append( eval(img_name.split('.png')[0]) )
    time_stamp_list.sort()
    return time_stamp_list

def read_image(time_stamp):
    img_path = save_path + 'pm/'
    file_name = str(time_stamp) + '.png'
    image = cv2.imread(img_path + file_name)
    return image


def read_state():
    state_path = save_path + 'state/'

    # read pose
    pose_file = state_path + 'pos.txt'
    time_stamp_list = []
    time_stamp_pose_dict = dict()
    file = open(pose_file, 'r') 
    while 1:
        line = file.readline()
        if not line:
            break
        if line == '\n':
            continue
        # print(line)

        line_list = line.split()

        index = eval(line_list[0])

        transform = carla.Transform()
        transform.location.x = eval(line_list[1])
        transform.location.y = eval(line_list[2])
        transform.location.z = eval(line_list[3])
        transform.rotation.pitch = eval(line_list[4])
        transform.rotation.yaw = eval(line_list[5])
        transform.rotation.roll = eval(line_list[6])

        time_stamp_list.append(index)
        time_stamp_pose_dict[index] = transform

    file.close()

    return time_stamp_list, time_stamp_pose_dict

class Param(object):
    def __init__(self):
        self.longitudinal_length = longitudinal_length
        self.ksize = ksize
        self.longitudinal_length = longitudinal_length
       
def read_pcd(time_stamp):
    pcd_path = save_path + 'pcd/'
    file_name = str(time_stamp) + '.npy'
    pcd = np.load(pcd_path+file_name)
    return pcd



pix_width = 0.05
map_x_min = 0.0
map_x_max = 10.0
map_y_min = -10.0
map_y_max = 10.0
lim_z = -1.5

width = int((map_x_max - map_x_min) / pix_width)
height = int((map_y_max - map_y_min) / pix_width)

u_bias = int(np.abs(map_y_max) / (map_y_max - map_y_min) * height)
v_bias = int(np.abs(map_x_max) / (map_x_max - map_x_min) * width)


def project(x, y):
    u = -x/pix_width + u_bias
    v = -y/pix_width + v_bias
    result = np.array([u, v])
    mask = np.where((result[0] < width) & (result[1] < height))
    result = result[:, mask[0]]
    return result.astype(np.int16)

def get_cost_map2(img, point_cloud, show=False):


    img2 = np.zeros((width, height, 1), np.uint8)
    img2.fill(255)
    # cv2.imwrite("D:\documents\SRTP\\test\\answeripm\ipmans4.jpg", img)

    # res = np.where((point_cloud[0] > map_x_min) & (point_cloud[0] < map_x_max) & (point_cloud[1] > map_y_min) & (point_cloud[1] < map_y_max))
    res = np.where((point_cloud[2] > lim_z) & (point_cloud[0] > map_x_min) & (point_cloud[0] < map_x_max) & (
                point_cloud[1] > map_y_min) & (point_cloud[1] < map_y_max))
    point_cloud = point_cloud[:, res[0]]
    u, v = project(point_cloud[0], point_cloud[1])
    img2[u, v] = 0

    kernel = np.ones((21, 21), np.uint8)
    img2 = cv2.erode(img2, kernel, iterations=1)
    img = cv2.addWeighted(img, 0.5, img2, 0.5, 0)
    kernel_size = (15, 15)
    sigma = 15
    img = cv2.GaussianBlur(img, kernel_size, sigma)
    if show:
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


def main():
    time_stamp_list = read_pm_time_stamp(save_path+'pm/')

    param = Param()
    sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
    sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)
    inverse_perspective_mapping = InversePerspectiveMapping(param, sensor_master)

    start = 0
    end = len(time_stamp_list)
    
    for i in tqdm(range(start, end)):
            time_stamp = time_stamp_list[i]
            pm_image = read_image(time_stamp)
            pcd = read_pcd(time_stamp)
    
            #tick1 = time.time()
            ipm_image = inverse_perspective_mapping.getIPM(pm_image)
            #tick2 = time.time()
    
            img = get_cost_map2(ipm_image, pcd)
            cv2.imwrite(save_path+'ipm/'+str(time_stamp)+'.png', img)
            #cv2.imshow('ipm_image', img)
            #cv2.waitKey(16)
            #print('time total: ' + str(tick2-tick1))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        #exit(0)
        pass