# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# __init__.py将文件夹当成Python模块使用
import os
import sys
import glob
import random

# 配置参数存储在字典"config"中
config = {
    # 本地客户端
    'host': 'localhost',
    # TCP通信端口
    'port': 2000,
    # 超时设定
    'timeout': 5.0,
    # RGB相机参数
    'camera': {
        # 长宽
        'img_length': 640,
        'img_width': 360,
        # 视角宽度
        'fov': 120,
        # 帧数
        'fps': 30,
    },
    # 激光雷达
    'lidar': {
        'channels': 64,
        # 转每分
        'rpm': 30,
        'sensor_tick': 0.05,
        'pps': 100000,
        'range': 5000,  # <= 0.9.5
        'lower_fov': -30,
        'upper_fov': 10,
    },
    # 惯性测量
    'imu': {
        'fps': 400,
    },
    # 卫星导航
    'gnss': {
        'fps': 20,
    },
}


# 环境添加load
def load(path='/data2/wanghejun/CARLA_0.9.13'):
    # 尝试运行添加路径
    try:
        sys.path.append(path + '/PythonAPI')
        # Python版本问题修改
        # sys.path.append(glob.glob(path + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        #     sys.version_info.major,
        #     sys.version_info.minor,
        #     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
        sys.path.append(path + '/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
    # 出现错误的处理
    except Exception as e:
        print(e)
        print('Fail to load carla library')


# world设置天气weather
def set_weather(world, weather):
    world.set_weather(weather)
    return weather


# world生成指定生成点的任意类型车辆vehicle
def add_vehicle(world, blueprint, vehicle_type='vehicle.bmw.grandtourer'):
    bp = random.choice(blueprint.filter(vehicle_type))
    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)
    transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, transform)
    return vehicle
