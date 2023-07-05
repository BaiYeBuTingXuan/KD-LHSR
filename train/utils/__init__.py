#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils模块为自定义包
import time
import numpy as np
import cv2
from PIL import Image


# SensorManager的子类
class Singleton(object):
    _instance = None

    # 重写内置的创建对象的静态方法
    # 分配空间，返回对象的引用
    # cls表示类本身
    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


# 调试用
def debug(info, info_type='debug'):
    if info_type == 'error':
        print('\033[1;31m ERROR:', info, '\033[0m')
    elif info_type == 'success':
        print('\033[1;32m SUCCESS:', info, '\033[0m')
    elif info_type == 'warning':
        print('\033[1;34m WARNING:', info, '\033[0m')
    elif info_type == 'debug':
        print('\033[1;35m DEBUG:', info, '\033[0m')
    else:
        print('\033[1;36m MESSAGE:', info, '\033[0m')


# 参数记录写在params.md中
def write_params(log_path, parser, description=None):
    opt = parser.parse_args()
    options = parser._optionals._actions
    with open(log_path + 'params.md', 'w+') as file:
        file.write('# Params\n')
        file.write('********************************\n')
        file.write('Time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        if description is not None:
            file.write('**Description**: ' + description + '\n')

        file.write('| Param | Value | Description |\n')
        file.write('| ----- | ----- | ----------- |\n')
        for i in range(len(parser._optionals._actions)):
            option = options[i]
            if option.dest != 'help':
                file.write('|**' + option.dest + '**|' + str(opt.__dict__[option.dest]) + '|' + option.help + '|\n')
        file.write('********************************\n\n')


# 图片数据向量化
def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # ARGB是一种色彩模式，也就是RGB色彩模式附加上Alpha（透明度）通道，常见于32位位图的存储结构。
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # r, g, b, and alpha 4 layers in total
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    # 以序号2的轴（维度大小为4）作roll运动3个单位，ARGB->RGBA
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# 添加透明度通道
def add_alpha_channel(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    alpha_channel[:, :int(b_channel.shape[0] / 2)] = 100
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA
