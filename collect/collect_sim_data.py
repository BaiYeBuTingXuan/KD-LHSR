#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import sys
import logging

# sys.path为环境列表模块，可动态修改解释器和相关环境
from os.path import join, dirname

# 0代表第一优先级，insert添加指定路径到导入模块的搜索文件夹，__file__返回相对路径（在sys.path中时），".."表示上第二级目录
sys.path.insert(0, join(dirname(__file__), '..'))

# 自定义的simulator和util配置模块
import simulator

# ./CarlaUE4.sh 路径
simulator.load('/data2/wanghejun/CARLA_0.9.13')
# PythonAPI中的carla模块
import carla
from carla import VehicleLightState as vls

sys.path.append('/data2/wanghejun/CARLA_0.9.13/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent
from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_random_destination, get_map, get_nav, replan, close2dest

# import os
import cv2
import time
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm

global_img = None
global_pcd = None
global_nav = None
global_control = None
global_pos = None
global_acceleration = None
global_angular_velocity = None
global_vel = None
MAX_SPEED = 15
MIN_DISTANCE = 500
CLOSE_DISTANCE = 10

plan_times = 0

# 参数剖析器
argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
argparser.add_argument(
    '-n', '--number-of-vehicles',
    metavar='N',
    default=50,
    type=int,
    help='number of vehicles (default: 10)')
argparser.add_argument(
    '-w', '--number-of-walkers',
    metavar='W',
    default=100,
    type=int,
    help='number of walkers (default: 50)')
argparser.add_argument(
    '--safe',
    action='store_true',
    help='avoid spawning vehicles prone to accidents')
# 蓝图过滤参数
argparser.add_argument(
    '--filterv',
    metavar='PATTERN',
    default='vehicle.*',
    help='vehicles filter (default: "vehicle.*")')
argparser.add_argument(
    '--filterw',
    metavar='PATTERN',
    default='walker.pedestrian.*',
    help='pedestrians filter (default: "walker.pedestrian.*")')
argparser.add_argument(
    '--tm-port',
    metavar='P',
    default=8000,
    type=int,
    help='port to communicate with TM (default: 8000)')
argparser.add_argument(
    '--sync',
    action='store_true',
    help='Synchronous mode execution')
argparser.add_argument(
    '--hybrid',
    action='store_true',
    help='Enanble')
argparser.add_argument(
    '--car-lights-on',
    action='store_true',
    default=False,
    help='Enanble car lights')
argparser.add_argument(
    '--num',
    type=int,
    default=100000,
    help='Total Number'
)
argparser.add_argument(
    '--data',
    type=int,
    default=0,
    help='data index'
)
argparser.add_argument(
    '--town',
    type=str,
    default='Town02',
    help='Visual Town in CARLA'
)
argparser.add_argument(
    '-i', '--ignore_traffic_lights',
    type=bool,
    metavar='I',
    default=True,
    help='Ignore_traffic_lights or not'
)

args = argparser.parse_args()

data_index = args.data

# 保存总路径
save_path = './DATASET/CARLA/NoTrafficLightNoNPC/' + str(data_index) + '/'

# 从字典config参数表设置client（到server）
# 构造函数Client(self,host(IP),port(TCP port),worker_threads)
# 创建carla.client用于和server进行通信
client = carla.Client(config['host'], config['port'])

# 保存总路径下创建子文件夹函数
def mkdir(path):
    os.makedirs(save_path + path, exist_ok=True)


mkdir('')
mkdir('img/')
mkdir('pcd/')
mkdir('nav/')
mkdir('state/')
mkdir('cmd/')

# 文件写入指针
cmd_file = open(save_path + 'cmd/cmd.txt', 'w+')
pos_file = open(save_path + 'state/pos.txt', 'w+')
vel_file = open(save_path + 'state/vel.txt', 'w+')
acc_file = open(save_path + 'state/acc.txt', 'w+')
angular_vel_file = open(save_path + 'state/angular_vel.txt', 'w+')


# 保存数据，以index为索引
def save_data(index):
    # 全局变量
    # 作为暂时写入的数据
    global global_img, global_pcd, global_nav, global_control, global_pos, global_vel, global_acceleration, global_angular_velocity
    # RGB和NAV
    cv2.imwrite(save_path + 'img/' + str(index) + '.png', global_img)
    cv2.imwrite(save_path + 'nav/' + str(index) + '.png', global_nav)
    # np.save将mat转写为np数组，保存在.npy文件中
    np.save(save_path + 'pcd/' + str(index) + '.npy', global_pcd)
    # 写入各.txt文件中
    cmd_file.write(index + '\t' +
                   str(global_control.throttle) + '\t' +
                   str(global_control.steer) + '\t' +
                   str(global_control.brake) + '\n')
    pos_file.write(index + '\t' +
                   str(global_pos.location.x) + '\t' +
                   str(global_pos.location.y) + '\t' +
                   str(global_pos.location.z) + '\t' +
                   str(global_pos.rotation.pitch) + '\t' +
                   str(global_pos.rotation.yaw) + '\t' +
                   str(global_pos.rotation.roll) + '\t' + '\n')
    vel_file.write(index + '\t' +
                   str(global_vel.x) + '\t' +
                   str(global_vel.y) + '\t' +
                   str(global_vel.z) + '\t' + '\n')
    acc_file.write(index + '\t' +
                   str(global_acceleration.x) + '\t' +
                   str(global_acceleration.y) + '\t' +
                   str(global_acceleration.z) + '\t' + '\n')
    angular_vel_file.write(index + '\t' +
                           str(global_angular_velocity.x) + '\t' +
                           str(global_angular_velocity.y) + '\t' +
                           str(global_angular_velocity.z) + '\t' + '\n')


# np.frombuffer动态数组便于数据采集
# 图像反馈函数 将相机data存入全局变量global_img中，NumPy数组形状(h, w, 4)
def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    # (samples, channels, height, width)
    # 4 channels: R, G, B, A
    array = np.reshape(array, (data.height, data.width, 4))  # RGBA format
    global_img = array


# 激光雷达反馈函数 将激光雷达data的points中选择符合两个坐标条件的
# 存入全局变量global_pcd中
def lidar_callback(data):
    global global_pcd
    # (samples, (dim1, dim2, dim3)) NumPy数组形状(1, 3) 点集points
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 4])
    # np.stack用于堆叠数组
    # 分别取第0，1，2列坐标，对应x，y，z，变成3行的二维数组point_cloud
    # 取反？
    point_cloud = np.stack([-lidar_data[:, 1], -lidar_data[:, 0], -lidar_data[:, 2]])
    mask = \
        np.where((point_cloud[0] > 1.0) | (point_cloud[0] < -4.0) | (point_cloud[1] > 1.2) | (point_cloud[1] < -1.2))[0]
    # np.where(condition)返回符合第一行y和第二行坐标x条件的序号（以元组的形式） 取元组第一位保存到mask中
    # 取满足条件mask列的坐标
    point_cloud = point_cloud[:, mask]
    # 在满足第一行条件下（只取一列）
    # 取满足第三行坐标条件的序号（最多一列）z坐标 安装高度2.5 0.3~2.8 => -2.2~0.3
    mask = np.where((point_cloud[2] > -0.3) & (point_cloud[2] < 2.2))[0]
    # mask = np.where(point_cloud[2] > -1.95)[0]
    point_cloud = point_cloud[:, mask]
    global_pcd = point_cloud
    # world.set_weather(carla.WeatherParameters.ClearNoon)


def main():
    # 全局变量 卫星导航图  汽车控制 位置 油门 角速度（方向盘）
    global plan_times
    global global_nav, global_control, global_pos, global_vel, global_acceleration, global_angular_velocity

    client.set_timeout(config['timeout'])

    # world = client.get_world()
    # client请求创建一个新世界，load_word(map_name)
    world = client.load_world('Town01')
    """
    weather = carla.WeatherParameters(
        cloudiness=random.randint(0,50),
        precipitation=0,
        sun_altitude_angle=random.randint(40,90)
    )
    set_weather(world, weather)
    """

    # 设置天气为晴天
    # WeatherParameters见P82
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 返回生成的角色的蓝图
    blueprint = world.get_blueprint_library()

    # 包含道路信息和航点管理的类carla.map
    world_map = world.get_map()

    # 在simulator构造函数中封装好的生成车辆函数，返回carla.Actor
    vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.audi.a2')
    # Enables or disables the simulation of physics on this actor.
    # 启用物理模拟
    vehicle.set_simulate_physics(True)

    # 传感器配置参数字典
    # 配置项：传感器安装位置 数据保存的响应函数名
    # @TODO: lidar filter
    sensor_dict = {
        'camera': {
            'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback': image_callback,
        },
        'lidar': {
            'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback': lidar_callback,
        },
    }
    # SensorManager::__init__(self, world, blueprint, vehicle, param_dict)
    # 传感器安置与初始化
    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()
    time.sleep(0.3)

    # 返回地图创建者的建议点位表用于车辆生成
    spawn_points = world_map.get_spawn_points()

    # 返回openDRIVE文件的拓扑的最小图元祖列表
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)

    # 使用carla自带的行为规划器进行导航
    agent = BasicAgent(vehicle, target_speed=MAX_SPEED, opt_dict={'ignore_traffic_lights': args.ignore_traffic_lights})

    # port = 8000
    # tm = client.get_trafficmanager(port)
    # vehicle.set_autopilot(True,port)
    # tm.ignore_lights_percentage(vehicle,100)

    # 随机目的地
    destination = get_random_destination(spawn_points)
    plan_map = replan(agent, destination, copy.deepcopy(origin_map))

    # # spawn_npc
    # vehicles_list = []
    # walkers_list = []
    # all_id = []

    # synchronous_master = False

    # try:
    #     world = client.get_world()

    #     # 交通管理器：处理所有设置为自动驾驶模式的车辆
    #     # 交通管理器独立于world，属于client的成员
    #     traffic_manager = client.get_trafficmanager(args.tm_port)
    #     # 设置车辆与其他车辆保持的最小距离
    #     traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    #     # 根据参数进行设置：混合物理模式、服务器与客户端同步异步、是否渲染等
    #     if args.hybrid:
    #         traffic_manager.set_hybrid_physics_mode(True)

    #     if args.sync:
    #         settings = world.get_settings()
    #         traffic_manager.set_synchronous_mode(True)
    #         if not settings.synchronous_mode:
    #             synchronous_master = True
    #             settings.synchronous_mode = True
    #             settings.fixed_delta_seconds = 0.05
    #             world.apply_settings(settings)
    #         else:
    #             synchronous_master = False

    #     blueprints = world.get_blueprint_library().filter(args.filterv)
    #     blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

    #     if args.safe:
    #         blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    #         blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
    #         blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    #         blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    #         blueprints = [x for x in blueprints if not x.id.endswith('t2')]

    #     # 出生点
    #     spawn_points = world.get_map().get_spawn_points()
    #     number_of_spawn_points = len(spawn_points)

    #     if args.number_of_vehicles < number_of_spawn_points:
    #         # 打乱列表
    #         random.shuffle(spawn_points)
    #     elif args.number_of_vehicles > number_of_spawn_points:
    #         msg = 'requested %d vehicles, but could only find %d spawn points'
    #         logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
    #         args.number_of_vehicles = number_of_spawn_points

    #     # 调用命令 carla.command
    #     # @TODO: cannot import these directly.
    #     SpawnActor = carla.command.SpawnActor
    #     SetAutopilot = carla.command.SetAutopilot
    #     SetVehicleLightState = carla.command.SetVehicleLightState
    #     # 未知类，默认值为int 0
    #     FutureActor = carla.command.FutureActor

    #     # --------------
    #     # Spawn vehicles
    #     # --------------
    #     # 命令列表batch
    #     batch = []
    #     for n, transform in enumerate(spawn_points):
    #         if n >= args.number_of_vehicles:
    #             break
    #         # 根据属性选择生成角色蓝图
    #         blueprint = random.choice(blueprints)
    #         if blueprint.has_attribute('color'):
    #             color = random.choice(blueprint.get_attribute('color').recommended_values)
    #             blueprint.set_attribute('color', color)
    #         if blueprint.has_attribute('driver_id'):
    #             driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
    #             blueprint.set_attribute('driver_id', driver_id)
    #         blueprint.set_attribute('role_name', 'autopilot')

    #         # prepare the light state of the cars to spawn
    #         light_state = vls.NONE
    #         if args.car_lights_on:
    #             light_state = vls.Position | vls.LowBeam | vls.LowBeam

    #         # spawn the cars and set their autopilot and light state all together
    #         # 【独立】命令 carla.command.~ 列表：
    #         # SpawnActor: world.spawn_actor(blueprint, transform)产生角色及其出生位置
    #         # SetAutopilot: vehicle.set_autopilot(actor_id, enabled, port)设置自动驾驶
    #         # SetVehileLightState: vehicle.set_light_state(actor_id, light_state)设置车辆灯光状态
    #         batch.append(SpawnActor(blueprint, transform)
    #                      .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
    #                      .then(SetVehicleLightState(FutureActor, light_state)))

    #     # client部署batch的所有命令，同步模式为synchronous_master(False)
    #     for response in client.apply_batch_sync(batch, synchronous_master):
    #         if response.error:
    #             logging.error(response.error)
    #         else:
    #             vehicles_list.append(response.actor_id)

    #     # -------------
    #     # Spawn Walkers
    #     # -------------
    #     # some settings
    #     percentagePedestriansRunning = 0.0  # how many pedestrians will run
    #     percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
    #     # 1. take all the random locations to spawn
    #     spawn_points = []
    #     for i in range(args.number_of_walkers):
    #         spawn_point = carla.Transform()
    #         loc = world.get_random_location_from_navigation()
    #         if (loc != None):
    #             spawn_point.location = loc
    #             spawn_points.append(spawn_point)
    #     # 2. we spawn the walker object
    #     batch = []
    #     walker_speed = []
    #     for spawn_point in spawn_points:
    #         walker_bp = random.choice(blueprintsWalkers)
    #         # set as not invincible
    #         if walker_bp.has_attribute('is_invincible'):
    #             walker_bp.set_attribute('is_invincible', 'false')
    #         # set the max speed
    #         if walker_bp.has_attribute('speed'):
    #             if (random.random() > percentagePedestriansRunning):
    #                 # walking
    #                 walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
    #             else:
    #                 # running
    #                 walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
    #         else:
    #             print("Walker has no speed")
    #             walker_speed.append(0.0)
    #         batch.append(SpawnActor(walker_bp, spawn_point))
    #     results = client.apply_batch_sync(batch, True)
    #     walker_speed2 = []
    #     for i in range(len(results)):
    #         if results[i].error:
    #             logging.error(results[i].error)
    #         else:
    #             walkers_list.append({"id": results[i].actor_id})
    #             walker_speed2.append(walker_speed[i])
    #     walker_speed = walker_speed2
    #     # 3. we spawn the walker controller
    #     batch = []
    #     walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    #     for i in range(len(walkers_list)):
    #         batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    #     results = client.apply_batch_sync(batch, True)
    #     for i in range(len(results)):
    #         if results[i].error:
    #             logging.error(results[i].error)
    #         else:
    #             walkers_list[i]["con"] = results[i].actor_id
    #     # 4. we put altogether the walkers and controllers id to get the objects from their id
    #     for i in range(len(walkers_list)):
    #         all_id.append(walkers_list[i]["con"])
    #         all_id.append(walkers_list[i]["id"])
    #     all_actors = world.get_actors(all_id)

    #     # wait for a tick to ensure client receives the last transform of the walkers we have just created
    #     if not args.sync or not synchronous_master:
    #         world.wait_for_tick()
    #     else:
    #         world.tick()

    #     # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    #     # set how many pedestrians can cross the road
    #     world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    #     for i in range(0, len(all_id), 2):
    #         # start walker
    #         all_actors[i].start()
    #         # set walk to random point
    #         all_actors[i].go_to_location(world.get_random_location_from_navigation())
    #         # max speed
    #         all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

    #     print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

    #     # example of how to use parameters
    #     traffic_manager.global_percentage_speed_difference(30.0)

    # finally:

    #     if args.sync and synchronous_master:
    #         settings = world.get_settings()
    #         settings.synchronous_mode = False
    #         settings.fixed_delta_seconds = None
    #         world.apply_settings(settings)

    # 迭代
    try:
        for cnt in tqdm(range(args.num)):
            if close2dest(vehicle, destination, dist=CLOSE_DISTANCE):
                # print(plan_times)
                destination = get_random_destination(spawn_points)
                
                # 防止规划新目标点距离过近
                # @TODO: 导航与车辆行驶实时大图
                while close2dest(vehicle, destination, dist=MIN_DISTANCE):
                    print("New destination too close!")
                    destination = get_random_destination(spawn_points)
                
                plan_map = replan(agent, destination, copy.deepcopy(origin_map))
                #global nav
                # plan_map.show()
                plan_times = plan_times+1
                # print("plan_times=",plan_times)

            if vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            control = agent.run_step()
            control.manual_gear_shift = False
            global_control = control
            vehicle.apply_control(control)
            # 获得卫星导航图
            nav = get_nav(vehicle, plan_map)

            global_nav = nav
            global_pos = vehicle.get_transform()
            global_vel = vehicle.get_velocity()
            global_acceleration = vehicle.get_acceleration()
            global_angular_velocity = vehicle.get_angular_velocity()

            cv2.imshow('Nav', nav)    
            cv2.imshow('Vision', global_img)
            cv2.waitKey(10)
            index = str(time.time())
            save_data(index)
    except KeyboardInterrupt:
        # print('\ndestroying %d vehicles' % len(vehicles_list))
        # client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # # stop walker controllers (list is [controller, actor, controller, actor ...])
        # for i in range(0, len(all_id), 2):
        #     all_actors[i].stop()

        # print('\ndestroying %d walkers' % len(walkers_list))
        # client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        sm.close_all()

        time.sleep(0.5)

    cmd_file.close()
    pos_file.close()
    vel_file.close()
    acc_file.close()
    angular_vel_file.close()

    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()

# 若作为包使用则不运行main()，__name__反映模块层次
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cmd_file.close()
        pos_file.close()
        vel_file.close()
        acc_file.close()
        angular_vel_file.close()

        cv2.destroyAllWindows()
