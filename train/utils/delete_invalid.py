import cv2
import glob
import os
from tqdm import tqdm
DELETE =False
dataset_path = "/data2/wanghejun/data/"
# index = [0,1,2,3,4]
index = [7]
def check_nav_valid(img):
    # print(img.shape) 240*300*3
    # img1 = img[-20:, :, :]
    # sum_white1 = len(img1[img1==255])
    # print(sum_white1)
    sum_white = len(img[img==255])
    # print(sum_white<205000)
    # print('****')
    return sum_white<205000

for i in index:
    files = glob.glob(dataset_path+str(i)+'/nav/*.png')
    print(f"checking for index {i}")
    cnt = 0
    for file in tqdm(files):
        file_name = file.split('/')[-1][:-4]
        nav_path = dataset_path + str(i)+'/nav/'+file_name+'.png'
        flag = os.path.exists(nav_path)
        if flag:
            nav = cv2.imread(nav_path)

        if (not flag) or (not check_nav_valid(nav)):
            cnt = cnt + 1

            if DELETE:
                img_path = dataset_path + str(i)+'/img/'+file_name+'.png'
                pcd_path = dataset_path + str(i)+'/pcd/'+file_name+'.npy'
                pm_path = dataset_path + str(i)+'/pm/'+file_name+'.png'
                ipm_path = dataset_path + str(i)+'/ipm/'+file_name+'.png'
                try:
                    os.remove(nav_path)
                except:
                    pass
                try:
                    os.remove(img_path)
                except:
                    pass
                try:
                    os.remove(pcd_path)
                except:
                    pass
                try:    
                    os.remove(pm_path)
                except:
                    pass
                try:    
                    os.remove(ipm_path)
                except:
                    pass
    print(f"find {cnt} invalid nav")