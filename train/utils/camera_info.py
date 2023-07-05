#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本代码主要是对雷达数据的处理

import cv2
import numpy as np

width = 1280
height = 720

fx = 711.642238
fy = 711.302135
s = 0.0
x0 = 644.942373
y0 = 336.030580

cameraMat = np.array([
    [fx, s, x0],
    [0., fy, y0],
    [0., 0., 1.]
])

distortionMat = np.array([-0.347125, 0.156284, 0.001037, -0.000109, 0.000000])

theta_y = 10.0 * np.pi / 180.  # pie/18

pitch_rotationMat = np.array([
    [np.cos(theta_y), 0., np.sin(theta_y)],
    [0., 1., 0.],
    [-np.sin(theta_y), 0., np.cos(theta_y)],
])

rotationMat = np.array([
    [-0.0024, -1.0000, -0.0033],
    [0.0746, 0.0031, -0.9972],
    [0.9972, -0.0026, 0.0746],
])

translationMat = np.array([0.0660, 0.1263, 0.2481])

theta_x = np.arctan2(rotationMat[2][1], rotationMat[2][2])

"""
theta_y = np.arctan2(-rotationMat[2][0], np.sqrt(rotationMat[2][1]**2 + rotationMat[2][2]**2))
theta_z = np.arctan2(rotationMat[1][0], rotationMat[0][0])

RxMat = np.array([
    [1.0, 0.0, 0.0],
    [0.0,  np.cos(theta_x),  -np.sin(theta_x)],
    [0.0,  np.sin(theta_x),  np.cos(theta_x)],
])

RyMat = np.array([
    [np.cos(theta_y),  0.0,  np.sin(theta_y)],
    [0.0,  1.0,  0.0],
    [-np.sin(theta_y),  0.0, np.cos(theta_y)],
])
    
RzMat = np.array([
    [np.cos(theta_z), -np.sin(theta_z), 0.0],
    [np.sin(theta_z),  np.cos(theta_z), 0.0],
    [0.0,  0.0,  1.0],
])
    
R = np.dot(np.dot(RzMat, RyMat), RxMat)
print(R, rotationMat)
print(theta_x*180./np.pi, theta_y*180./np.pi, theta_z*180./np.pi)
"""
rotationMat = np.dot(rotationMat, np.linalg.inv(pitch_rotationMat))  #


# 点云数据point_cloud(一组三维向量)
# rotationMat 乘数矩阵
# translationMat 常数矩阵
# 方阵用于向量处理
def lidar2camera(point_cloud, rotationMat=rotationMat, translationMat=translationMat, file_name='merge', data_index=1):
    # 把激光雷达数据投射到俯视视角
    img = np.zeros((720, 1280, 3), np.uint8)
    trans_pc = np.dot(rotationMat, point_cloud) + np.tile(translationMat, (point_cloud.shape[1], 1)).T  # 将数据乘上乘数矩阵加常数矩阵
    image_uv = np.array([  # 用z处理x,y的数据
        trans_pc[0] * fx / trans_pc[2] + x0,

        trans_pc[1] * fy / trans_pc[2] + y0
    ])
    total = image_uv.shape[1]  # 数据个数
    for i in range(total):
        point = (int(image_uv[0][i]), int(image_uv[1][i]))
        if point[0] > width or point[0] < 0 or point[1] > height or point[1] < 0:  # 数据范围需要在图片内
            continue
        # cv2.circle(img, point, 2, (i/total*255, 0, 255-i/total*255), 8)
        cv2.circle(img, point, 2, (255, 255, 255), 8)
        # 用白点表示数据
    return img
    # cv2.imwrite('/media/wang/DATASET/label'+str(data_index)+'/'+file_name+'.png',img)


def lidar2camera(point_cloud, rotationMat=rotationMat, translationMat=translationMat, file_name='merge', data_index=1):
    # 好像又写了一遍
    img = np.zeros((720, 1280, 3), np.uint8)
    trans_pc = np.dot(rotationMat, point_cloud) + np.tile(translationMat, (point_cloud.shape[1], 1)).T
    image_uv = np.array([
        trans_pc[0] * fx / trans_pc[2] + x0,
        trans_pc[1] * fy / trans_pc[2] + y0
    ])
    total = image_uv.shape[1]
    for i in range(total):
        point = (int(image_uv[0][i]), int(image_uv[1][i]))
        if point[0] > width or point[0] < 0 or point[1] > height or point[1] < 0:
            continue
        # cv2.circle(img, point, 2, (i/total*255, 0, 255-i/total*255), 8)
        cv2.circle(img, point, 2, (255, 255, 255), 8)
    return img


def lidar2camera_test(img, point_cloud, rotationMat=rotationMat, translationMat=translationMat, file_name='merge',
                      data_index=1):
    trans_pc = np.dot(rotationMat, point_cloud) + np.tile(translationMat, (point_cloud.shape[1], 1)).T
    image_uv = np.array([
        trans_pc[0] * fx / trans_pc[2] + x0,
        trans_pc[1] * fy / trans_pc[2] + y0
    ])
    total = image_uv.shape[1]
    for i in range(total):
        point = (int(image_uv[0][i]), int(image_uv[1][i]))
        if point[0] > width or point[0] < 0 or point[1] > height or point[1] < 0:
            continue
        cv2.circle(img, point, 2, (i / total * 255, 0, 255 - i / total * 255), 8)  # 为了检验点画的对不对，每个点都画成不同的颜色，用于图片输出方便观看
    cv2.imwrite('test_output/test_' + file_name + '.png', img)


def camera2lidar(image_uv):
    # 把俯视视角图片转换为激光雷达数据
    rotation = np.linalg.inv(np.dot(cameraMat, rotationMat))
    translation = np.dot(cameraMat, translationMat)
    translation = np.dot(rotation, translation)
    R = rotation
    T = translation
    roadheight = -1.55  # 道路高度常数

    u = image_uv[0]
    v = image_uv[1]

    zi = (T[2] + roadheight) / (R[2][0] * u + R[2][1] * v + R[2][2])
    xl = (R[0][0] * u + R[0][1] * v + R[0][2]) * zi - T[0]
    yl = (R[1][0] * u + R[1][1] * v + R[1][2]) * zi - T[1]
    trans_pc = np.array([
        xl,
        yl,
        [roadheight] * image_uv.shape[1]
    ])
    return trans_pc
