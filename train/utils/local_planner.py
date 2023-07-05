#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本篇代码主要实现引导区域数据和雷达障碍区域混合，通过高斯模糊生成图片，然后计算路径的cost，找出最优路径发现，返回转弯半径，并与端对端方法进行比较

import cv2
import numpy as np
import time

pix_width = 0.05  # 像素大小
map_x_min = 0.0
map_x_max = 10.0  # x轴范围
map_y_min = -10.0
map_y_max = 10.0  # y轴范围
lim_z = -1.5  # point_cloud所得数据的z轴范围

width = int((map_x_max - map_x_min) / pix_width)  # 像素宽度
height = int((map_y_max - map_y_min) / pix_width)  # 像素高度

u_bias = int(np.abs(map_y_max) / (map_y_max - map_y_min) * height)  # y轴正轴的像素个数
v_bias = int(np.abs(map_x_max) / (map_x_max - map_x_min) * width)  # x轴正轴像素个数


def read_pcd(file_path):  # 数据读入，数据格式应如下 每行为x y z intensity 其中x需为正值，返回[x,y,z]的一个3*n的数组和intensity
    x = []
    y = []
    z = []
    intensity = []  # 强度
    with open(file_path, 'r') as file:  # 打开文件
        lines = file.readlines()  # 分行储存为列表
        [lines.pop(0) for _ in range(11)]  # _指代临时变量，此指执行删除列表的第一个元素11次，并将其存在一个列表中（但是这里并未用变量保存下来）
        for line in lines:
            sp_line = line.split()  # 以空格分开
            if float(sp_line[0]) < 0:
                continue
            x.append(float(sp_line[0]))
            y.append(float(sp_line[1]))
            z.append(float(sp_line[2]))
            intensity.append(float(sp_line[3]))  # 转换为浮点数
    return np.array([x, y, z]), intensity  # 返回[x,y,z]的一个3*n的数组和intensity


def project(x, y):  # 将坐标转换为像素位置，像素坐标原点位于左上角，那么原坐标原点位置位于中下方，向左为y正轴，向右为y负轴，向上为x正轴
    u = -x / pix_width + u_bias  # 像素位置中的横坐标
    v = -y / pix_width + v_bias  # 像素位置中的纵坐标
    result = np.array([u, v])
    mask = np.where((result[0] < width) & (result[1] < height))  # 挑选出合法的坐标
    result = result[:, mask[0]]
    return result.astype(np.int16)  # 转换为Int16


def get_cost_map(trans_pc, point_cloud, show=False):  # 生成数据合成后的引导区域，img是预计俯视路径的数据生成的costmap，img2是激光雷达的点云数据生成的costmap
    img = np.zeros((width, height, 1), np.uint8)  # 生成width*height*1的数组，数据类型为unsigned integer 8
    img.fill(0)  # img初始为黑色，表示全部为不可行区域
    img2 = np.zeros((width, height, 1), np.uint8)  # 生成width*height*1的数组，数据类型为unsigned integer 8
    img2.fill(255)  # img2初始为白色，表示全部为可行区域，应该是因为雷达数据输入的是障碍位置的坐标

    res = np.where(
        (trans_pc[0] > map_x_min) & (trans_pc[0] < map_x_max) & (trans_pc[1] > map_y_min) & (trans_pc[1] < map_y_max))
    trans_pc = trans_pc[:, res[0]]  # 取出合法范围内的坐标
    u, v = project(trans_pc[0], trans_pc[1])  # 转换为像素坐标
    img[u, v] = 255  # 把img中的合法坐标位置设置为白色，表示可行区域

    # res = np.where((point_cloud[0] > map_x_min) & (point_cloud[0] < map_x_max) & (point_cloud[1] > map_y_min) & (point_cloud[1] < map_y_max))
    res = np.where((point_cloud[2] > lim_z) & (point_cloud[0] > map_x_min) & (point_cloud[0] < map_x_max) & (
                point_cloud[1] > map_y_min) & (point_cloud[1] < map_y_max))
    point_cloud = point_cloud[:, res[0]]  # 取出合法范围内的坐标
    u, v = project(point_cloud[0], point_cloud[1])
    img2[u, v] = 0  # 把img2中的合法坐标位置设置为黑色，表示障碍区域

    kernel = np.ones((21, 21), np.uint8)  # 生成21*21的数组初始为1，数据类型为unsigned integer 8
    img2 = cv2.erode(img2, kernel, iterations=1)
    # erode函数用于图像的腐蚀（在此处就是把白色部分的边界变成黑色，可能是因为路径边缘本身也应该是不可通行的）
    # 其原理是kernel为自定义的一个二维大小，对于每一个白色像素，如果其旁边kernel范围内的像素有一个原来是黑色的话就设置它为黑色，这样白色区域边界就会变小
    # iterations是迭代次数
    img = cv2.addWeighted(img, 0.5, img2, 0.5, 0)  # 把两幅图片以一比一的比重合为一副图片,最后一个0表示啥颜色都不添加
    kernel_size = (15, 15)
    sigma = 15
    # 高斯模糊函数，就是对中心点邻域像素值进行‘加权平均’后将值赋予中心像素点，领域大小由kernel决定，sigma是x方向上的高斯核标准偏差
    # 具体功能就是让图片变得模糊一点边界不会那么清晰
    img = cv2.GaussianBlur(img, kernel_size, sigma);
    if show:  # show默认为flase,表示是否要在GUI显示混合好的图片
        cv2.imshow('Result', img)  # 显示图片
        cv2.waitKey(0)  # 控制显示图片的时长，一般在imshow的时候,如果设置 waitKey(0),代表按任意键继续
        cv2.destroyAllWindows()
    return img


m = 50  # 把角分为50份，也是50个可能的行进方向，具体在中文论文的第5.4节中
max_theta = 2 * np.pi / 3  # 接下来行进转弯路径角的最大角为120度
L = 8.0  # 转弯路径长度
Length = 1.448555  # 计算偏角时的转弯路径长度
vel = 0.5
n = 100  # 取点个数
# 曲率位于[-0.2~0.2]之间

collision_penalty = 1.0  # 碰撞惩罚系数
half_width = 10  # 半车宽度
collision_threshhold = 70  # 碰撞阈值


def gen_r():  #
    rs = []
    for i in range(m):
        theta = i * 2.0 * max_theta / m - max_theta
        if np.abs(theta - 0.) < 0.00001:  # 如果角度为0的话，设半径为9999999999999
            rs.append(9999999999999)
            continue
        r = L / theta
        rs.append(r)  # 转弯半径
        rs.append(-r)
    return rs


# 以下为DWA实现部分
def get_cmd(img, show=False, save=False, file_name=None):  # 找到cost最大的方向及碰撞概论最小，并返回
    rs = gen_r()
    for i in range(m):
        theta = i * 2.0 * max_theta / m - max_theta  # 如果角度为0的话，设半径为99999（有点奇怪这边又算了一遍，除了极值以外没有区别）
        if np.abs(theta - 0.) < 0.00001:
            rs.append(99999)
            continue
        r = L / theta
        rs.append(r)  # 转弯半径
        rs.append(-r)

    best_cost = 0
    best_r = 99999
    best_u = []
    best_v = []

    for r in rs:
        theta = L / np.abs(r)
        cost = 0
        indexs = np.arange(n)
        xs = np.abs(r * np.sin(indexs * theta / n))
        ys = r * (1 - np.cos(indexs * theta / n))  # 将转弯路径上分为n个点,得到x坐标和y坐标
        u, v = project(xs, ys)  # 转换为像素位置
        v2 = np.clip(v + half_width, 0, height - 1)  # v+half_width限制在0~height-1中
        v3 = np.clip(v - half_width, 0, height - 1)  # v-half_width限制在0~height-1中

        mask = np.where(img[u, v] < collision_threshhold)[0]
        mask2 = np.where(img[u, v2] < collision_threshhold)[0]
        mask3 = np.where(img[u, v3] < collision_threshhold)[0]
        all_collision = len(mask) + len(mask2) + len(mask3)  # 取出半径上碰撞范围内的点（即颜色小于70），然后计算他们的总长

        cost = sum(img[u, v] / 255.0) + sum(img[u, v2] / 255.0) + sum(
            img[u, v3] / 255.0) - collision_penalty * all_collision
        # 计算cost，公式为半径左右加上本身的颜色综合减去碰撞乘法
        # img[u, v] = 0

        if best_cost < cost:  # 找到最大的cost
            best_cost = cost
            best_r = r
            best_u = u
            best_v = v

    # img[best_u,best_v] = 0

    if show:
        # cv2.imshow('Result', img)
        # cv2.waitKey(100)
        cv2.imwrite(str(time.time()) + '.png', img)  # 以时间戳为名字保存img
        # cv2.destroyAllWindows()

    direct = -1.0 if best_r > 0 else 1.0  # 判断路径是左半边还是右半边
    yaw = direct * np.arctan2(Length, abs(best_r))  # 计算偏角(转弯角度，这里化曲为直，将Length看作线段)具体为 arctan(Length/abs(best_r))
    return yaw  # , rwpf(best_r)


k_k = 1.235
k_theta = 0.456
k_e = 0.11  # 0.1386
max_steer = np.deg2rad(30)


def pi2pi(theta, theta0=0.0):
    return (theta + np.pi) % (2 * np.pi) - np.pi  # 把角表示为[-pie,pie)范围内的角


def rwpf(radius, distance=0.5):  # 与端对端方法进行对比，判断指令的有效性
    kr = 1. / radius  # 移动曲率 表示轮子速度 radius是转弯的半径（参考文献方法）
    vr = v = 1.0

    thetar = kr * distance  # 转弯角度
    # 计算目标位置相对于当前位置x轴坐标的增值和y轴坐标的增值
    if kr < 0.001:  # 如果转弯角度太小，就默认为直线行驶即往x轴正方向行驶
        xr, yr = distance, 0
    else:
        xr = np.sin(distance * kr) / kr
        yr = (1 - np.cos(distance * kr)) / kr

    dx = - xr
    dy = - yr
    tx = np.cos(thetar)
    ty = np.sin(thetar)
    e = dx * ty - dy * tx  # 计算偏差，理论为0
    theta_e = pi2pi(- thetar)  # 把角表示为[-pie,pie)范围内的角

    alpha = 1.8
    # 与端对端方法进行对比，判断指令的有效性，具体在参考文献27
    w1 = k_k * vr * kr * np.cos(theta_e)
    w2 = - k_theta * np.fabs(vr) * theta_e
    w3 = (k_e * vr * np.exp(-theta_e ** 2 / alpha)) * e
    w = (w1 + w2 + w3) * 0.8
    # print(dx, dy, current_state.theta, xr, yr, thetar)
    if v < 0.02:
        steer = 0
    else:
        steer = np.arctan2(w * Length, v) * 2 / np.pi * max_steer

    return steer
