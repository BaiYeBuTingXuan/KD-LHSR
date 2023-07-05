#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 处理机器人的实际运行轨迹，并将它与全局规划路径进行匹配，截出图片
# 但是奇怪的是在本程序的实现中，实际上只截了最后一个点也只匹配了路径中的最后一个点（应该只是对最后一个点进行测试，具体如果用于应该用需要修改）
# 好像没有体现DSW的完整过程

import cv2
import math
import time
import numpy as np
from PIL import Image
import threading
from utils.manual_gps import dist_p2p, manual_gps_y, manual_gps_x, find_nn, gen_manual_gps


def avg(values):  # 求平均值
    summ = sum(values)
    return summ / len(values)


def normal(values, is_y=False):  # values中的值减去一个定值(从名字上看像减去平均值，但是提前给出了)
    # avg_value = values[0]
    if is_y:
        avg_value = (11589596.333751025 + 11589605.59585501) / 2
    else:
        avg_value = (3047362.377753752 + 3047362.981469983) / 2
    return [(item - avg_value) for item in values]


def filt_gps(x_list, y_list, ts_list):  # 处理真实位置的投影，由文章中5.3.1.2中可知机器人轨迹要通过一定比例投影到地图中
    filt_x = [-x_list[0] * scale_x + x_offset]
    filt_y = [y_list[0] * scale_y + y_offset]
    last_ts = float(ts_list[0])  # 存储上一个位置的时间
    for i in range(len(x_list) - 1):
        # ts = float(ts_list[i])
        next_s = float(ts_list[i + 1])
        dt = max(0.01, next_s - last_ts)  # 时间差
        dist = dist_p2p(x_list[i], y_list[i], x_list[i + 1], y_list[i + 1])  # 两点距离
        v = dist / dt  # 平均速度

        if v > 0.0:
            last_ts = next_s

        if v < 5 and v > 0.0:  # 要求速度合理
            x = x_list[i]
            y = y_list[i]
            _x = -x * scale_x + x_offset
            _y = y * scale_y + y_offset
            filt_x.append(_x)
            filt_y.append(_y)
        else:  # 如果速度不合理的话，就取和上个位置一样的位置
            filt_x.append(filt_x[-1])
            filt_y.append(filt_y[-1])

    return filt_x, filt_y


data_index = 4
scale_x = 6.0  # x轴扩大倍率
scale_y = 6.0  # y轴扩大倍率
x_offset = 3300  # x轴偏移量
y_offset = 2600  # y轴偏移量


class NavMaker:  # NavMaker类
    def __init__(self, reader, imu):  # 初始化函数
        self.reader = reader
        self.imu = imu
        self.x = []
        self.y = []
        self.t = []
        self.last_angle = 0.
        self.gps_x, self.gps_y = gen_manual_gps(manual_gps_x, manual_gps_y)  # 把全局规划路径进行离散
        self.map = Image.open('map-mark.png')
        self.size_y = 800
        self.size_x = 400  # 截下来的局部路径图片大小
        self.stop_read_event = threading.Event()  # 事件管理标志，默认为false

        self.read_cyclic = threading.Thread(  # 将self.read_data作为线程目标
            target=self.read_data, args=()
        )
        self.nav = Image.open('nav.png').convert('RGB')
        self.yaw = 0.0

    def start(self):
        self.stop_read_event.clear()  # 清空事件状态
        self.read_cyclic.start()

    def close(self):  # 停止读入
        self.stop_read_event.set()

    def read_data(self):
        while not self.stop_read_event.is_set():  # 读入未停止时
            x, y, t = self.reader.get()  # 读取机器人实际轨迹
            # print('get data', x, y)
            ax, ay, az, yaw, pitch, roll, w = self.imu.get()
            self.yaw = yaw
            self.get_gps(x, y, t)
            self.get_nav()
            time.sleep(0.05)  # 过0.05秒再执行下个线程

    def get(self):
        return self.nav

    def get_gps(self, x, y, t):
        nx = normal([x])[0]
        ny = normal([y], True)[0]
        self.x.append(nx)
        self.y.append(ny)
        self.t.append(t)

    def make_nav(self, angle, index):  # 对固定点在导航图上的路径图片进行裁剪，并进行旋转
        img2 = self.map.crop((self.gps_y[index] - self.size_y, self.gps_x[index] - self.size_x,
                              self.gps_y[index] + self.size_y, self.gps_x[index] + self.size_x))
        im_rotate = img2.rotate(angle)
        size_y2 = 100
        size_x2 = 80
        img3 = im_rotate.crop((
                              img2.size[0] // 2 - size_y2, img2.size[1] // 2 - 2 * size_x2, img2.size[0] // 2 + size_y2,
                              img2.size[1] // 2))
        # 截下旋转后偏中间部分的图片
        # img4 = cv2.cvtColor(np.asarray(img3),cv2.COLOR_RGB2BGR)
        # cv2.imshow("OpenCV",img4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img3.convert('RGB')

    def get_nav(self):  # 处理机器人实际运行采集的数据，截出最后一个点的导航图
        if len(self.x) < 3:  # 如果点数太少，相当于路径太短或者采集失误
            # print('No GPS')
            return

        if len(self.x) < 50:
            nx, ny, ts = self.x, self.y, self.t
            dy = (ny[-1] - ny[-2])
            dx = (nx[-1] - nx[-2])
            self.last_angle = 180. * math.atan2(dy, dx) / math.pi  # 计算倒数第二个点向第一个点的角度为多少（用角度表示而非弧度）
        else:
            nx, ny, ts = self.x[-100:], self.y[-100:], self.t[-100:]  # 如果太长只取后一百个

        fter_x, fter_y = filt_gps(nx, ny, ts)  # 处理实际运行数据
        nn_x, nn_y, nn_index = find_nn(fter_x[-1], fter_y[-1], self.gps_x, self.gps_y)  # 找到规划路径中离实际运行最后一个点最近的点以及下标
        # dy = (self.gps_y[min(nn_index+5,len(fter_x)-1)] - self.gps_y[max(0,nn_index-5)])
        # dx = (self.gps_x[min(nn_index+5,len(fter_x)-1)] - self.gps_x[max(0,nn_index-5)])
        # angle = 180.*math.atan2(dy, dx)/math.pi

        # input_angle = 0.5*self.last_angle+0.5*angle
        # self.last_angle = input_angle
        # print('yaw', self.yaw*180./math.pi, input_angle)
        input_angle = self.yaw * 180. / math.pi
        self.nav = self.make_nav(-input_angle + 80., nn_index)  # 把规划路径的那个点附近的导航图截出来
