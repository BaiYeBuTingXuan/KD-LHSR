#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 实现了车动态行进时截下局部导航图，但似乎没有实现与全局规划路线的点匹配（可能在主程序里面实现）

import cv2
import random
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw

scale = 12.0
x_offset = 800  # 1500#2500
y_offset = 1000  # 0#3000


def get_random_destination(spawn_points):  # 从点序列中随机取一个并返回
    return random.sample(spawn_points, 1)[0]


def get_map(waypoint_tuple_list):  #
    origin_map = np.zeros((6000, 6000, 3), dtype="uint8")
    origin_map.fill(255)
    # array到image转变
    # 建立一个6000*6000全白的RGB图片
    origin_map = Image.fromarray(origin_map)  # 转换为图片，但是不知道为什么没有加convert('RGB')
    """
    for i in range(len(waypoint_tuple_list)):
        _x1 = waypoint_tuple_list[i][0].transform.location.x
        _y1 = waypoint_tuple_list[i][0].transform.location.y
        _x2 = waypoint_tuple_list[i][1].transform.location.x
        _y2 = waypoint_tuple_list[i][1].transform.location.y

        x1 = scale*_x1+x_offset
        x2 = scale*_x2+x_offset
        y1 = scale*_y1+y_offset
        y2 = scale*_y2+y_offset
        draw = ImageDraw.Draw(origin_map)
        draw.line((x1, y1, x2, y2), 'white', width=12)
    """
    return origin_map


def draw_route(agent, destination, origin_map):  # 画出导航路线
    start_waypoint = agent._map.get_waypoint(agent._vehicle.get_location())
    end_waypoint = agent._map.get_waypoint(destination.location)

    route_trace = agent._trace_route(start_waypoint, end_waypoint)  # 取出起点到终点的路线
    route_trace_list = []
    for i in range(len(route_trace)):  # 对每个点进行处理
        x = scale * route_trace[i][0].transform.location.x + x_offset
        y = scale * route_trace[i][0].transform.location.y + y_offset
        route_trace_list.append(x)
        route_trace_list.append(y)
    draw = ImageDraw.Draw(origin_map)
    draw.line(route_trace_list, 'red', width=30)  # 在路线上画出粗30的红线，并体现在原图中
    return origin_map


def get_nav(vehicle, plan_map, town=1):  # 截出车目前行进位置点的导航图
    if town == 1:
        x_offset = 800
        y_offset = 1000
    elif town == 2:
        x_offset = 1500
        y_offset = 0
    x = int(scale * vehicle.get_location().x + x_offset)
    y = int(scale * vehicle.get_location().y + y_offset)
    _nav = plan_map.crop((x - 400, y - 400, x + 400, y + 400))

    # r = 10
    # draw = ImageDraw.Draw(_nav)
    # draw.ellipse((_nav.size[0]//2-r, _nav.size[1]//2-r, _nav.size[0]//2+r, _nav.size[1]//2+r), fill='green', outline='green', width=10)

    im_rotate = _nav.rotate(vehicle.get_transform().rotation.yaw + 90)
    nav = im_rotate.crop(
        (_nav.size[0] // 2 - 150, _nav.size[1] // 2 - 2 * 120, _nav.size[0] // 2 + 150, _nav.size[1] // 2))
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)
    return nav


def get_big_nav(vehicle, plan_map):  # 截出车目前行进位置点的导航图,比get_nav得出的图要大一点
    x = int(scale * vehicle.get_location().x + x_offset)
    y = int(scale * vehicle.get_location().y + y_offset)
    _nav = plan_map.crop((x - 400, y - 400, x + 400, y + 400))

    r = 20
    draw = ImageDraw.Draw(_nav)
    # 画一个绿色的点（在中间）
    draw.ellipse((_nav.size[0] // 2 - r, _nav.size[1] // 2 - r, _nav.size[0] // 2 + r, _nav.size[1] // 2 + r),
                 fill='green', outline='green', width=10)

    im_rotate = _nav.rotate(vehicle.get_transform().rotation.yaw + 90)
    # nav = im_rotate
    nav = im_rotate.crop((0, 0, _nav.size[0], _nav.size[1] // 2))
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)
    return nav


def replan(agent, destination, origin_map):  # 重新画图，更改原图，终点，并把路线标成红色
    agent.set_destination((destination.location.x,
                           destination.location.y,
                           destination.location.z))
    plan_map = draw_route(agent, destination, origin_map)
    return plan_map


def replan2(agent, destination, origin_map):  # 重新画图，更改原图，终点，并把路线标成红色,与上面那个函数是两种方式，但实现功能不知道有什么差别
    agent.set_destination(agent.vehicle.get_location(), destination.location, clean=True)
    plan_map = draw_route(agent, destination, origin_map)
    return plan_map


def close2dest(vehicle, destination, dist=30):  # 判断距离是否<dist
    return destination.location.distance(vehicle.get_location()) < dist
