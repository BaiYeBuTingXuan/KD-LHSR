#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 离散导航图上的全局规划路径，但没有说怎么获取，程序中的数据是人工截取的（可能实际操作也是人工采集）

import math

# 似乎是一组人工采集的gps数据
manual_gps_y = [2400, 2300, 3815, 3630, 3325, 3240, 2855, 2570, 2420, 2465, 2115, 2260, 2095, 2075, 2200, 1655, 1820,
                2605, 2460]
manual_gps_x = [3200, 3430, 4000, 4520, 4400, 4600, 4665, 4555, 4240, 4090, 3970, 3555, 3490, 3395, 3055, 2820, 2335,
                2615, 3045]


def dist_p2p(x1, y1, x2, y2):  # 计算两点直线距离
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def find_nn(x, y, x_list, y_list):  # 找到(x_list，y_list)中离(x,y)最近的点以及下标
    assert len(x_list) == len(y_list)  # 要求x_list和y)list大小一致
    nn_dist = 9999999
    nn_x = None
    nn_y = None
    nn_index = 0
    for i in range(len(x_list)):
        dist = dist_p2p(x, y, x_list[i], y_list[i])

        if dist < nn_dist:
            nn_dist = dist
            nn_x = x_list[i]
            nn_y = y_list[i]
            nn_index = i
    return nn_x, nn_y, nn_index


def gen_manual_gps(x, y):
    # 传进来的参数没用过，使用的数据在文件一开始
    # GPS导航上的数据差分，原本的数据是每五米一个点，差分为每一米分为一个点
    step_len = 1.0  # 5.0 meters per setp
    gps_y = []
    gps_x = []
    for i in range(len(manual_gps_y) - 1):
        x1 = manual_gps_x[i]
        y1 = manual_gps_y[i]
        x2 = manual_gps_x[i + 1]
        y2 = manual_gps_y[i + 1]
        dist = dist_p2p(x1, y1, x2, y2)
        n = int(dist / step_len)
        for j in range(n):
            gps_x.append(x1 + (x2 - x1) * j / n)
            gps_y.append(y1 + (y2 - y1) * j / n)

    return gps_x, gps_y
