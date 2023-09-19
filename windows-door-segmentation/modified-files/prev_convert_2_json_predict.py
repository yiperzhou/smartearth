import copy
import csv
import os
import pickle
import time
import logging
from datetime import datetime
import json
import math

import matplotlib.patches as mpathes
import matplotlib.path as Path
import matplotlib.pyplot as plt
import numpy as np
import shapefile
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import AxesGrid
from PIL import Image
from shapely.geometry import Polygon, mapping, LineString
from skimage import transform
from skimage.draw import polygon
import itertools
from numpy import genfromtxt

import cv2

from model_nets import get_model
from utils.config import discrete_cmap, polygons_to_image
from post_processes.post_prosessing import get_polygons, split_prediction
from load_augmentations.augmentations import RotateNTurns



class Write_Shp_File(object):
    def __init__(self,filepath,filename,shapetype=5,transform_scale=[0,0,1,0]):
        self.shps = []        
        self.transform_scale = transform_scale
        self.w = shapefile.Writer(os.path.join(filepath,filename))
        self.w.autoBalance = 1
        self.w.shapeType = shapetype
        self.w.field('name', 'C', '40')  # 名称，表示多线型区域的名称如墙、门、窗等
        self.w.field('type', 'C', '40')  # 类别名称，表示wall、railing、opening等大类

    def write_feature(self,geom,name,featuretype):
        shp = {}
        geometry = geom.copy()
        if self.w.shapeType == 5:
            geometry.append(geom[0])
            self.w.poly([geometry])
            shp['polygon'] = self.coordinate_system_transform(geometry)
            self.w.record(name, featuretype)
            shp['record'] = [name, featuretype]
            self.shps.append(shp)
        elif self.w.shapeType == 3:
            self.w.line([geometry])            
            shp['line'] = self.coordinate_system_transform(geometry)
            self.w.record(name, featuretype)
            shp['record'] = [name, featuretype]
            self.shps.append(shp)

    def coordinate_system_transform(self,shape):
        tx,ty,scale,angle = self.transform_scale
        # M = np.array([[scale,0,tx],[0,scale,ty],[0,0,1]])
        # shp = []
        # for point in shape:
        #     point_v = np.array([point[0],point[1],1]).reshape((3,1))
        #     point_t = M.dot(point_v)
        #     shp.append([point_t[0][0],point_t[1][0]])

        if tx == 0 and ty == 0 and scale == 1 and angle == 0:
            return shape

        #numpy矩阵乘法实现方式，无for循环
        pt = np.array(shape).transpose((1,0)) #将list转换为矩阵并转置行列
        # pt[-1] = fig_height - pt[-1] #将矩阵纵坐标变为左下角零点
        pt = np.row_stack((pt,np.ones_like(pt[-1]))) #规整矩阵格式，将坐标行下添加一个全1行以方便偏移计算
        if not (tx == 0 and ty == 0 and scale == 1):
            M = np.array([[scale,0,tx],
                        [0,scale,ty],
                        [0,0,1]]) #生成变换矩阵
            pt = M.dot(pt) #矩阵相乘
        if angle != 0:
            angle = angle*math.pi/180
            #以(tx,ty)为参考点旋转
            # T = np.array([[math.cos(angle),-1*math.sin(angle),tx-tx*math.cos(angle)+ty*math.sin(angle)],
            #             [math.sin(angle),math.cos(angle),ty-tx*math.sin(angle)-ty*math.cos(angle)],
            #             [0,0,1]])
            #以(0,0)为参考点旋转
            T = np.array([[math.cos(angle),-1*math.sin(angle),0],
                        [math.sin(angle),math.cos(angle),0],
                        [0,0,1]])
            pt = T.dot(pt)
        shp = pt[:-1].transpose((1,0)).tolist() #转换矩阵为列表
    
        return shp

class Write_GeoJSON_File(object):
    def __init__(self,filepath,filename,transform_scale=None):
        self.name = os.path.join(filepath,filename+'.geojson')
        if transform_scale is None:
            self.transform_scale=[0,0,1,0]
        else:
            self.transform_scale = transform_scale
        self.geojson = {"type": "FeatureCollection","features":[]}    

    def add_feature(self,geom,geomtype,propertydict):
        feature = {"type": "Feature","properties":propertydict,"geometry":None}
        geometry = {"type":geomtype,"coordinates":None}

        if geomtype in ["Polygon","LineString"]:        
            if geomtype == "Polygon":
                if geom[0] != geom[-1]:
                    geom.append(geom[0])        
                geom = self.coordinate_system_transform(geom)
                coordinates = [geom]

            else:   
                geom = self.coordinate_system_transform(geom)
                coordinates = geom
            
            geometry["coordinates"] = coordinates
            feature["geometry"] = geometry
            self.geojson["features"].append(feature)

    def write_geojson(self):
        gjs = json.dumps(self.geojson)
        with open(self.name,'w') as f:
            f.write(gjs)

    def coordinate_system_transform(self,shape):
        tx,ty,scale,angle = self.transform_scale

        # M = np.array([[scale,0,tx],[0,scale,ty],[0,0,1]])
        # shp = []
        # for point in shape:
        #     point_v = np.array([point[0],point[1],1]).reshape((3,1))
        #     point_t = M.dot(point_v)
        #     shp.append([float(point_t[0][0]),float(point_t[1][0])])

        if tx == 0 and ty == 0 and scale == 1 and angle == 0:
            return shape

        #numpy矩阵乘法实现方式，无for循环
        pt = np.array(shape).transpose((1,0)) #将list转换为矩阵并转置行列
        # pt[-1] = fig_height - pt[-1] #将矩阵纵坐标变为左下角零点
        pt = np.row_stack((pt,np.ones_like(pt[-1]))) #规整矩阵格式，将坐标行下添加一个全1行以方便偏移计算
        if not (tx == 0 and ty == 0 and scale == 1):
            M = np.array([[scale,0,tx],
                        [0,scale,ty],
                        [0,0,1]]) #生成变换矩阵
            pt = M.dot(pt) #矩阵相乘
        if angle != 0:
            angle = angle*math.pi/180
            #以(tx,ty)为参考点旋转
            # T = np.array([[math.cos(angle),-1*math.sin(angle),tx-tx*math.cos(angle)+ty*math.sin(angle)],
            #             [math.sin(angle),math.cos(angle),ty-tx*math.sin(angle)-ty*math.cos(angle)],
            #             [0,0,1]])
            #以(0,0)为参考点旋转
            T = np.array([[math.cos(angle),-1*math.sin(angle),0],
                        [math.sin(angle),math.cos(angle),0],
                        [0,0,1]])
            pt = T.dot(pt)
        shp = pt[:-1].transpose((1,0)).tolist() #转换矩阵为列表

        return shp



### 栅格图转矢量图坐标转换
def raster2vector(height, shp_list):
    vec_list = []
    for shp in shp_list:
        vec_shp = []
        for point in shp:
            vec_shp.append([point[0], height - point[1]])
        vec_list.append(vec_shp)
    return vec_list

###判断门窗在哪面墙上
def wd_position(polygons, types, pol):
    rs = 0
    center_point = (
    int((pol[0][0] + pol[1][0] + pol[2][0] + pol[3][0]) / 4), int((pol[0][1] + pol[1][1] + pol[2][1] + pol[3][1]) / 4))
    for i, shp in enumerate(polygons):
        if types[i]['type'] == 'wall':
            X, Y = [], []
            for p in shp:
                X.append(p[0])
                Y.append(p[1])
            rr, cc = polygon(X, Y)
            for j, r in enumerate(rr):
                if center_point[0] == r and center_point[1] == cc[j]:
                    rs = i + 1
    return rs

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

###将墙的多边形转换为shapely规范[(x1,y1),(x2,y2),...]
def get_shapely_data(poly):
    poly_new = [(poly[0][0], poly[0][1]), (poly[1][0], poly[1][1]), (poly[2][0], poly[2][1]), (poly[3][0], poly[3][1])]
    return poly_new

### Setup Model
# 载入模型
def load_model(gpu_flag,model_file_dir):
    n_classes = 44
    split = [21, 12, 11]
    model = get_model('hg_furukawa_original', n_classes)

    # model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    # model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    ### CPU
    # checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    # GPU
    model_name_list = os.listdir(model_file_dir)
    if 'model_best_val_loss_var.pkl' in model_name_list:
        model_file_name = 'model_best_val_loss_var.pkl'
    else:
        model_file_name = model_name_list[0]
    model_file = os.path.join(model_file_dir,model_file_name)
    checkpoint = torch.load(model_file,map_location='cuda:0') if gpu_flag else torch.load(model_file, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    ### GPU
    if gpu_flag:
        model.cuda()
    return model, n_classes, split

###定义标注列表
def get_classes(classes_path):
    room_classes = []
    csv_file = csv.reader(open(classes_path + 'room_classes.csv', 'r'))
    for room in csv_file:
        room_classes.append(room[1])

    icon_classes = []
    csv_file = csv.reader(open(classes_path + 'icon_classes.csv', 'r'))
    for icon in csv_file:
        icon_classes.append(icon[1])

    n_rooms = len(room_classes)
    n_icons = len(icon_classes)
    return room_classes, icon_classes, n_rooms, n_icons

###载入图像
def load_image(data_file, save_path):
    pil_img = Image.open(data_file)
    pil_img.save(save_path + '0-orignal.png', quality=95)
    # print(f'np.array(pil_img).shape:{np.array(pil_img).shape}')
    img = np.moveaxis(np.array(pil_img)[:, :, :3], -1, 0)[np.newaxis, :, :, :]
    # print(f'img.shape:{img.shape}')

    ### 如果底色为黑色，则将颜色取反
    total = img[0][0].sum()
    base = img[0][0].shape[0] * img[0][0].shape[1]
    score = total / base
    # print(score)
    if score < 50:
        img[0] = 255 - img[0]
        # print(img[0][0])
        mask = img[0].sum(axis=0)
        # print(mask)
        img[0][1][mask < 250 * 3] = 255 - img[0][1][mask < 250 * 3]

    image = torch.from_numpy(img)
    np_img = np.moveaxis(image[0].cpu().data.numpy(), 0, -1)
    # print(f'image.shape:{image.shape}')
    # print(f'np_img.shape:{np_img.shape}')

    return image, np_img, pil_img

### predict
def predicting(gpu_flag,image, model, n_classes):
    image = 2 * image.data.type(torch.FloatTensor) / 255 - 1
    rot = RotateNTurns()
    with torch.no_grad():
        height = image.shape[2]
        width = image.shape[3]
        img_size = (height, width)
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width])  # (4,44,height,width)
        #     print(prediction.shape)
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            image = image.cuda() if gpu_flag else image
            rot_image = rot(image, 'tensor', forward)
            pred = model(rot_image)
            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            # We add the prediction to output
            prediction[i] = pred[0]

    # 取均值，合并四个通道，生成[44,heigh,weight]张量
    prediction = torch.mean(prediction, 0, True)
    return prediction, height, width, img_size

# 以下为展示rooms和icons
def prediction_display(prediction, n_rooms, n_icons, room_classes, icon_classes, save_path):
    rooms_pred = F.softmax(prediction[0, 21:21 + 12], 0).cpu().data.numpy()
    # np.argmax找出每个位置对应的最大值的下标组成一个数组，
    rooms_pred = np.argmax(rooms_pred, axis=0)
    icons_pred = F.softmax(prediction[0, 21 + 12:], 0).cpu().data.numpy()
    icons_pred = np.argmax(icons_pred, axis=0)

    #opencv画图测试


    # print(f'rooms_pred.shape:{rooms_pred.shape}')
    # print(f'icons_pred.shape:{icons_pred.shape}')
    # print(f'prediction.shape:{prediction.shape}')
    # print(f'heat_pred.shape:{heat_pred.shape}')

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    rseg = ax.imshow(rooms_pred, cmap='rooms', vmin=0, vmax=n_rooms - 0.1)
    cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(room_classes, fontsize=20)
    plt.savefig(save_path + '1-rseg.png')
    # plt.show()
    plt.close()

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(icons_pred, cmap='icons', vmin=0, vmax=n_icons - 0.1)
    cbar = plt.colorbar(iseg, ticks=np.arange(n_icons) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(icon_classes, fontsize=20)
    plt.savefig(save_path + '2-iseg.png')
    plt.close()
    # plt.show()

### post processing
def post_prosess(prediction, img_size, split, pil_img, save_path):
    height,width = img_size
    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    # print('prediction:{0}'.format(prediction.shape))
    # print('heatmaps:{0}'.format(heatmaps.shape))
    # print('rooms:{0}'.format(rooms.shape))
    # print('icons:{0}'.format(icons.shape))
    # polygons, types, room_polygons, room_types,wall_lines_from_conner,wall_lines_from_predict,wall_lines_from_getshort,wall_lines_from_polygon,wall_lines_from_polygon_middle,wall_line_discart,wall_points,wall_part_points = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2],pil_img)
    polygons, types,wall_lines_from_polygon = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2],pil_img)
    #pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, height, width)
    # np.save(save_path + "rooms.npy", pol_room_seg)
    # np.save(save_path + "icons.npy", pol_icon_seg)
    return polygons, types, wall_lines_from_polygon

# ### 去掉被包含的墙
def delete_innerwall(polygons, types):
    polygons_new = []
    types_new = []
    for i, pol1 in enumerate(polygons):
        for j, pol2 in enumerate(polygons):
            if i != j and types[i]['type'] == 'wall' and types[j]['type'] == 'wall':
                pol3 = get_shapely_data(pol1)
                pol4 = get_shapely_data(pol2)
                if Polygon(pol4).contains(Polygon(pol3)):
                    break
        else:
            polygons_new.append(pol1)
            types_new.append(types[i])

    return polygons_new, types_new

# 获取墙列表及其方向列表
def get_wall_and_direct(polygons, types):
    wall_polylist = []
    wall_direct = []
    for i, pol in enumerate(polygons):
        if types[i]['type'] == 'wall':  # 提取wall和railing的信息
            polyshp = get_shapely_data(pol)
            wall_polylist.append(polyshp)
            # 判断墙的方向并存进列表(第0个点与第1个点距离，同第1个点与第2个点距离比较)
            h0_1 = abs(polyshp[0][0] - polyshp[1][0])
            v0_1 = abs(polyshp[0][1] - polyshp[1][1])
            h1_2 = abs(polyshp[1][0] - polyshp[2][0])
            v1_2 = abs(polyshp[1][1] - polyshp[2][1])
            l0_1 = (h0_1 ** 2 + v0_1 ** 2) ** 0.5
            l1_2 = (h1_2 ** 2 + v1_2 ** 2) ** 0.5
            if l0_1 <= l1_2:
                wall_direct.append({'0_1_type': 'short', 'D_h': h1_2, 'D_v': v1_2, 'D_l': l1_2})
            else:
                wall_direct.append({'0_1_type': 'long', 'D_h': h0_1, 'D_v': v0_1, 'D_l': l0_1})
    return wall_polylist, wall_direct

### 判断墙加长后是否相交
def judge_intersection(index, pol1, pol2, wall_direct, wall_polylist, buffer):
    flag = True
    i = index
    h = wall_direct[i]['D_h']
    v = wall_direct[i]['D_v']
    l = wall_direct[i]['D_l']
    x_stride = int(buffer / l * h)
    y_stride = int(buffer / l * v)
    if wall_direct[i]['0_1_type'] == 'short':
        pol3 = [(pol1[0][0] - x_stride, pol1[0][1] - y_stride), (pol1[1][0] - x_stride, pol1[1][1] - y_stride),
                (pol1[2][0], pol1[2][1]), (pol1[3][0], pol1[3][1])]
        pol4 = [(pol1[0][0], pol1[0][1]), (pol1[1][0], pol1[1][1]), (pol1[2][0] + x_stride, pol1[2][1] + y_stride),
                (pol1[3][0] + x_stride, pol1[3][1] + y_stride)]
        if Polygon(pol3).intersects(Polygon(pol2)):
            wall_polylist[i][0] = pol3[0]
            wall_polylist[i][1] = pol3[1]
        elif Polygon(pol4).intersects(Polygon(pol2)):
            wall_polylist[i][2] = pol4[2]
            wall_polylist[i][3] = pol4[3]
        else:
            flag = False
    elif wall_direct[i]['0_1_type'] == 'long':
        pol3 = [(pol1[0][0] - x_stride, pol1[0][1] + y_stride), (pol1[1][0], pol1[1][1]), (pol1[2][0], pol1[2][1]),
                (pol1[3][0] - x_stride, pol1[3][1] + y_stride)]
        pol4 = [(pol1[0][0], pol1[0][1]), (pol1[1][0] + x_stride, pol1[1][1] - y_stride),
                (pol1[2][0] + x_stride, pol1[2][1] - y_stride), (pol1[3][0], pol1[3][1])]
        if Polygon(pol3).intersects(Polygon(pol2)):
            wall_polylist[i][0] = pol3[0]
            wall_polylist[i][3] = pol3[3]
        elif Polygon(pol4).intersects(Polygon(pol2)):
            wall_polylist[i][1] = pol4[1]
            wall_polylist[i][2] = pol4[2]
        else:
            flag = False
    return flag, wall_polylist

### 更新墙多边形，小缺口墙补齐
def get_new_wall(polygons, types, buffer_range):
    ###去掉被包含的墙
    polygons, types = delete_innerwall(polygons, types)
    # 获取墙列表及其方向列表
    wall_polylist, wall_direct = get_wall_and_direct(polygons, types)

    ### 将墙加长，判断是否与其他墙相交并补齐加长的长度
    # buffer_range = [1,15] #定义最大buffer的长度
    wall_polylist_new = copy.deepcopy(wall_polylist)
    for i, pol1 in enumerate(wall_polylist):
        for j, pol2 in enumerate(wall_polylist):
            if Polygon(pol1).intersects(Polygon(pol2)):
                continue
            else:
                for d in range(buffer_range[1], 0, -1):
                    flag, wall_polylist_new = judge_intersection(i, pol1, pol2, wall_direct, wall_polylist_new, d)
                    if not flag:
                        break

    ### 将新生成的墙替换原polygon
    poly_new = []
    for poly in polygons:
        pol = []
        for point in poly:
            pol.append([point[0], point[1]])
        poly_new.append(pol)

    for i, poly in enumerate(wall_polylist_new):
        poly1 = [[poly[0][0], poly[0][1]], [poly[1][0], poly[1][1]], [poly[2][0], poly[2][1]], [poly[3][0], poly[3][1]]]
        poly_new[i] = poly1

    polygons = poly_new
    return polygons, types, wall_polylist_new

#### 提取房间区域多边形
def get_XY(points):
    X, Y = np.array([]), np.array([])
    for a in points:
        y, x = a[0], a[1]
        X = np.append(X, np.round(float(x)))
        Y = np.append(Y, np.round(float(y)))
    return X, Y

def get_room_polygon(area_image):
    # ### 将数组读取为opencv图片并二值化
    area_images = np.array([area_image, area_image, area_image], np.uint8)
    area_images = np.moveaxis(area_images, 0, -1)

    gray = cv2.cvtColor(area_images, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3,3), 0)
    # image_binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    image_binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    ###从二值图像中提取轮廓
    contours = cv2.findContours(image_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    ### 将获得的多边形坐标按照shp格式整理
    area_simple = []
    for poly in contours:
        polygonx = []
        for point in poly:
            polygonx.append(point[0].tolist())
        area_simple.append(polygonx)
    return area_simple

### 以墙重新分割多边形
def get_room_area(height, width, room_polygons, room_types, wall_polylist_new, room_classes):
    room_area = []
    for i, r in enumerate(room_polygons):
        room_area_part = []
        for poly in mapping(r)['coordinates'][0]:
            room_area_part.append(poly)
        room_area.append(room_area_part)

    wall_2_room = np.zeros((height, width), dtype=np.uint8)
    for i, poly in enumerate(room_area):
        X, Y = get_XY(poly)
        rr, cc = polygon(X, Y)
        valuenum = room_types[i]['class']
        wall_2_room[rr, cc] = valuenum

    wall_area_out = np.zeros((height, width), dtype=np.uint8)
    for i, poly in enumerate(wall_polylist_new):
        # poly = [[poly[0][0],poly[0][1]],[poly[1][0],poly[1][1]],[poly[2][0],poly[2][1]],[poly[3][0],poly[3][1]]]
        X, Y = get_XY(poly)
        rr, cc = polygon(X, Y)
        wall_area_out[rr, cc] = 255

    ### 提取房间区域语义多边形(房间区域包含墙)
    shp_poly = get_room_polygon(wall_area_out)

    ### 将提取的多边形画在画布上，以去掉包含墙的信息
    wall_area_out2 = np.zeros((height, width), dtype=np.uint8)
    for i, poly in enumerate(shp_poly):
        X, Y = get_XY(poly)
        rr, cc = polygon(X, Y)
        wall_area_out2[rr, cc] = 255

    for i, row in enumerate(wall_area_out):  # 在房间语义信息中去掉墙的区域
        for j, col in enumerate(row):
            if col == 255:
                wall_area_out2[i][j] = 0

    ### 提取房间区域语义多边形(房间区域不包含墙)
    area_simple = get_room_polygon(wall_area_out2)

    ###获取房间对应的语义信息
    room_types = []
    for i, pol in enumerate(area_simple):
        mask = np.zeros((height, width), dtype=np.uint8)
        X, Y = get_XY(pol)
        rr, cc = polygon(X, Y)
        mask[rr, cc] = 1
        maskp = mask == 1
        valuep = np.unique(wall_2_room[maskp])
        if valuep.size > 0 and mask[maskp].sum() != 0:
            keyp = wall_2_room[maskp].sum() / mask[maskp].sum()
            index = np.argmin(keyp - valuep)
            room_types.append({'type': room_classes[valuep[index]], 'class': valuep[index]})
        else:
            room_types.append({'type': room_classes[-1], 'class': len(room_classes)-1})
    return area_simple, room_types

##将房间语义画为栅格图显示
def room_seg_display(height, width, area_simple, room_types, n_rooms, room_classes, save_path):
    area_simple_image = np.zeros((height, width), dtype=np.uint8)
    for i, poly in enumerate(area_simple):
        X, Y = get_XY(poly)
        rr, cc = polygon(X, Y)
        valuenum = room_types[i]['class']
        area_simple_image[rr, cc] = valuenum

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    rseg = ax.imshow(area_simple_image, cmap='rooms', vmin=0, vmax=n_rooms - 0.1)
    cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(room_classes, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path + '7-post-rseg.png')
    # plt.show()
    plt.close()

##plot the poly_wall画矢量图(任意个边和角度)
def vector_display(height, area_simple, save_path):
    fig, ax = plt.subplots()
    for pols in area_simple:
        path_data = []
        for i, point in enumerate(pols):
            if i == 0:
                path_data.append((Path.Path.MOVETO, (point[0], height - point[1])))
            else:
                path_data.append((Path.Path.LINETO, (point[0], height - point[1])))
            if i == len(pols) - 1:
                path_data.append((Path.Path.CLOSEPOLY, (pols[0][0], height - pols[0][1])))
            plt.scatter(point[0], height - point[1])

        codes, verts = zip(*path_data)
        path = Path.Path(verts, codes)
        patch = mpathes.PathPatch(path, facecolor='none', alpha=0.9)
        ax.add_patch(patch)

    plt.axis('equal')
    plt.grid()
    plt.savefig(save_path + '6-post-rseg_v.png')
    # plt.show()
    plt.close()

### write shapefile --polygon
def write_shp_file(height, polygons, types, room_classes, icon_classes, save_path,wall_line_single,transform_scale):
    polygons = raster2vector(height, polygons)
    wall_line_single = raster2vector(height, wall_line_single)
  
    ###将numpy数组转为列表
    polygons = np.array(polygons).tolist()

    wall_shapefile = Write_Shp_File(save_path,'sem_wall',shapetype=5,transform_scale=transform_scale)
    opening_shapefile = Write_Shp_File(save_path,'sem_opening',shapetype=5,transform_scale=transform_scale)
    wallline_shapefile = Write_Shp_File(save_path,'wall_line',shapetype=3,transform_scale=transform_scale)

    shps = []
    for i, pol in enumerate(polygons):
        if types[i]['type'] == 'wall':
            name = room_classes[types[i]['class']]
            if types[i]['class'] == 2 or types[i]['class'] == 8:#墙、栏杆
                wall_shapefile.write_feature(pol,name,'Wall')

        elif types[i]['type'] == 'icon':
            name = icon_classes[types[i]['class']]
            if types[i]['class'] == 1 or types[i]['class'] == 2: #窗，门
                opening_shapefile.write_feature(pol,name,'Opening')  
    else:
        shps += wall_shapefile.shps
        wall_shapefile.w.close()

        shps += opening_shapefile.shps
        opening_shapefile.w.close()

    for i, pol in enumerate(wall_line_single):
        name = 'Wall'
        wallline_shapefile.write_feature(pol,name,'Wall')
    else:
        shps += wallline_shapefile.shps
        wallline_shapefile.w.close()    

    return shps    

def write_geojson_file(height, polygons, types, room_classes, icon_classes, save_path,wall_line_single,transform_scale):
    polygons = raster2vector(height, polygons)
    wall_line_single = raster2vector(height, wall_line_single)
  
    ###将numpy数组转为列表
    polygons = np.array(polygons).tolist()

    geojson_file = Write_GeoJSON_File(save_path,'sem_all',transform_scale=transform_scale)

    for i, pol in enumerate(polygons):
        if types[i]['type'] == 'wall':
            name = room_classes[types[i]['class']]
            if types[i]['class'] == 2 or types[i]['class'] == 8:#墙、栏杆
                propertydict = {"name":name,"type":'Wall'}
                geojson_file.add_feature(pol,"Polygon",propertydict)

        elif types[i]['type'] == 'icon':
            name = icon_classes[types[i]['class']]
            if types[i]['class'] == 1 or types[i]['class'] == 2: #窗，门
                propertydict = {"name":name,"type":'Opening'}
                geojson_file.add_feature(pol,"Polygon",propertydict)
 
    for i, pol in enumerate(wall_line_single):
        propertydict = {"name":"Wall","type":'Wall'}
        geojson_file.add_feature(pol,"LineString",propertydict)    
    
    geojson_file.write_geojson()
    shps = geojson_file.geojson

    return shps 

###将多边形转化为线
def get_line_from_wallpolygons(poly):
    line = []
    side_length1 = ((poly[0][0]-poly[1][0])**2 + (poly[0][1]-poly[1][1])**2)**0.5
    side_length2 = ((poly[1][0]-poly[2][0])**2 + (poly[1][1]-poly[2][1])**2)**0.5

    if side_length1 < side_length2:
        point_1_x = (poly[0][0] + poly[1][0])/2
        point_1_y = (poly[0][1] + poly[1][1])/2
        point_2_x = (poly[2][0] + poly[3][0])/2
        point_2_y = (poly[2][1] + poly[3][1])/2
    else:
        point_1_x = (poly[0][0] + poly[3][0])/2
        point_1_y = (poly[0][1] + poly[3][1])/2
        point_2_x = (poly[1][0] + poly[2][0])/2
        point_2_y = (poly[1][1] + poly[2][1])/2

    point = [[point_1_x,point_1_y],[point_2_x,point_2_y]]
    line.append(point)
    return line

# 以下为展示后处理之后的结果
def post_prosessing_display(pol_room_seg, pol_icon_seg, n_rooms, n_icons, room_classes, icon_classes, save_path,wall_points):
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    rseg = ax.imshow(pol_room_seg, cmap='rooms', vmin=0, vmax=n_rooms - 0.1)
    cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(room_classes, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path + '3-post-rseg.png')
    # plt.show()
    plt.close()

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(pol_icon_seg, cmap='icons', vmin=0, vmax=n_icons - 0.1)
    cbar = plt.colorbar(iseg, ticks=np.arange(n_icons) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(icon_classes, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path + '4-post-iseg.png')
    # plt.show()
    plt.close()

###groud truth
def ground_truth_display(np_img, pol_icon_seg, pol_room_seg, n_icons, save_path):
    fig = plt.figure(figsize=(26, 12))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 2),
                    axes_pad=0.05,
                    )

    imagesri = [np_img, pol_icon_seg + pol_room_seg]
    for i, ax in enumerate(grid):
        ax.set_axis_off()
        im = ax.imshow(imagesri[i], cmap='rooms', vmin=0, vmax=n_icons - 0.1)
    plt.savefig(save_path + '5-compose-riseg.png')
    # plt.show()
    plt.close()

####展示墙角点及热图等中间信息
def process_display(prediction,wall_points,wall_part_points,n_rooms, pil_img,save_path):
    #展示热图
    heat_map = np.zeros(prediction[0][22].shape)
    for i in range(13):
        heat_pred = prediction[0][i].cpu().data.numpy()
        heat_map += heat_pred

    for i in range(13):
        x = np.array([point[0] for point in wall_part_points[i]])
        y = np.array([point[1] for point in wall_part_points[i]])
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(1, 1, 1)
        ax.axis('off')
        iseg = ax.imshow(1-heat_map, cmap=plt.cm.gray)
        plt.scatter(x, y, marker='o',color='r')
        plt.colorbar(iseg,shrink=1)
        plt.savefig(save_path + '8-' + str(i) + '-heat.png')
        plt.close()

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(1-heat_map, cmap=plt.cm.gray)
    plt.colorbar(iseg,shrink=1)
    plt.savefig(save_path + '8-13-heat.png')
    plt.close()


    #将所有热图上的点展示在热图叠加图上
    #展示wall_points
    x = np.array([point[0] for point in wall_points])
    y = np.array([point[1] for point in wall_points])
    # y1 = prediction[0][22].shape[0] - y

    # plt.figure(figsize=(12, 12))
    # ax = plt.subplot(1, 1, 1)
    # ax.axis('off')
    # plt.scatter(x, y1, marker='o',color='r')
    # # plt.tight_layout()
    # plt.savefig(save_path + '9-post-iseg.png')
    # # plt.show()
    # plt.close()

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(1-heat_map, cmap=plt.cm.gray)
    plt.scatter(x, y, marker='o',color='r')
    # plt.colorbar(iseg,shrink=1)
    plt.savefig(save_path + '9-post-iseg.png')
    plt.close()

    #将所有热图上的点展示在预测图上
    rooms_pred = F.softmax(prediction[0, 21:21 + 12], 0).cpu().data.numpy()
    rooms_pred = np.argmax(rooms_pred, axis=0)

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    rseg = ax.imshow(rooms_pred, cmap='rooms', vmin=0, vmax=n_rooms - 0.1)
    plt.scatter(x, y, marker='o',color='r')
    # plt.colorbar(iseg,shrink=1)
    plt.savefig(save_path + '10-post-iseg.png')
    plt.close()

    #将所有热图上的点展示在原图上
    image_gray = pil_img.convert('L')
    threshold_table = [ 0 if i < 185 else 1 for i in range(256)]

    image_2_binary = image_gray.point(threshold_table,'1')
    image_2_binary.save(save_path + 'binary.png')

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(image_2_binary)
    plt.scatter(x, y, marker='o',color='r')
    plt.savefig(save_path + '11-post-iseg.png')
    plt.close()

    image_2_binary = np.ones_like(rooms_pred)
    mask = rooms_pred == 2
    image_2_binary[mask] = 0
    mask = rooms_pred == 8
    image_2_binary[mask] = 0

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    rseg = ax.imshow(image_2_binary)
    plt.scatter(x, y, marker='o',color='r')
    plt.savefig(save_path + '12-wall_rseg.png')
    # plt.show()
    plt.close()

def predict_o(gpu_flag,model_file_dir, classes_path, data_file, transform_scale, save_path, buffer_range=[1, 15]):
    logging.basicConfig(level=logging.INFO)
    time_start = time.time()
    discrete_cmap()

    model, n_classes, split = load_model(gpu_flag,model_file_dir)
    time_lm = time.time()
    ts1 = int((time_lm - time_start))
    # print("Model loaded:%d" % ts1)
    logging.info("Model loaded:%d" % ts1)

    room_classes, icon_classes, n_rooms, n_icons = get_classes(classes_path)

    image, np_img, pil_img = load_image(data_file, save_path)
    time_ld = time.time()
    ts2 = int((time_ld - time_lm))
    print('data loaded:%d' % ts2)

    prediction, height, width, img_size = predicting(gpu_flag,image, model, n_classes)
    prediction_display(prediction, n_rooms, n_icons, room_classes, icon_classes, save_path)
    time_pd = time.time()
    ts3 = int((time_pd - time_ld))
    print('predict complete:%d' % ts3)

    polygons, types, wall_lines_from_polygon = post_prosess(prediction, img_size, split,pil_img,save_path)
    time_bqq = time.time()
    ts3 = int((time_bqq - time_pd))
    print('getpolygon complete:%d' % ts3)

    #补齐单线墙缺口
    connect_wall_line = get_connect_wall_line(wall_lines_from_polygon)

    #补齐墙缺口，形成封闭区间
    polygons, types, wall_polylist_new = get_new_wall(polygons, types, buffer_range)
    time_bqh = time.time()
    ts9 = int((time_bqh - time_bqq))
    print('inersect complete:%d' % ts9)

    #展示墙角点及热图等中间信息
    # process_display(prediction,wall_points,wall_part_points,n_rooms,pil_img,save_path)

    # shps = write_shp_file(height, polygons, types, room_classes, icon_classes, save_path,connect_wall_line,transform_scale)
    shps = write_geojson_file(height, polygons, types, room_classes, icon_classes, save_path,connect_wall_line,transform_scale)
    #post_prosessing_display(pol_room_seg, pol_icon_seg, n_rooms, n_icons, room_classes, icon_classes, save_path,wall_points)
    time_pp = time.time()
    ts7 = int((time_pp - time_bqh))
    print('shpwrite complete:%d' % ts7)
    ts4 = int((time_pp - time_pd))
    print('post processed complete:%d' % ts4)
    ts6 = int((time_pp - time_start))
    # print('total time-cost:%d' % ts6)
    logging.info('toal time-cost:%d' % ts6)
    return shps

def predict(gpu_flag, model_file_dir, classes_path, data_dict, transform_scale,save_dir,shm_obj=None, buffer_range=[1, 15]):
    logging.basicConfig(level=logging.INFO)
    time_start = time.time()
    discrete_cmap()

    model, n_classes, split = load_model(gpu_flag,model_file_dir)
    time_lm = time.time()
    ts1 = int((time_lm - time_start))
    # print("Model loaded:%d" % ts1)
    logging.info("Model loaded:%d" % ts1)

    room_classes, icon_classes, n_rooms, n_icons = get_classes(classes_path)
    datas_result = {}
    for count, data_key in enumerate(data_dict.keys()):
        data_file = data_dict[data_key]
        img_name = data_key #获取文件名称

        save_path = os.path.join(save_dir,img_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        time_lm = time.time()
        image, np_img, pil_img = load_image(data_file, save_path)
        time_ld = time.time()
        ts2 = int((time_ld - time_lm))
        print('data loaded:%d' % ts2)

        prediction, height, width, img_size = predicting(gpu_flag,image, model, n_classes)
        prediction_display(prediction, n_rooms, n_icons, room_classes, icon_classes, save_path)
        time_pd = time.time()
        ts3 = int((time_pd - time_ld))
        print('predict complete:%d' % ts3)

        polygons, types, wall_lines_from_polygon = post_prosess(prediction, img_size, split,pil_img,save_path)
        time_bqq = time.time()
        ts3 = int((time_bqq - time_pd))
        print('getpolygon complete:%d' % ts3)

        #补齐单线墙缺口
        connect_wall_line = get_connect_wall_line(wall_lines_from_polygon)

        #补齐墙缺口，形成封闭区间
        polygons, types, wall_polylist_new = get_new_wall(polygons, types, buffer_range)
        time_bqh = time.time()
        ts9 = int((time_bqh - time_bqq))
        print('inersect complete:%d' % ts9)

        #展示墙角点及热图等中间信息
        # process_display(prediction,wall_points,wall_part_points,n_rooms,pil_img,save_path)

        # shps = write_shp_file(height, polygons, types, room_classes, icon_classes, save_path,connect_wall_line,transform_scale)
        shps = write_geojson_file(height, polygons, types, room_classes, icon_classes, save_path,connect_wall_line,transform_scale)
        #post_prosessing_display(pol_room_seg, pol_icon_seg, n_rooms, n_icons, room_classes, icon_classes, save_path,wall_points)
        time_pp = time.time()
        ts7 = int((time_pp - time_bqh))
        print('shpwrite complete:%d' % ts7)
        ts4 = int((time_pp - time_pd))
        print('post processed complete:%d' % ts4)

        datas_result[data_key] = json.dumps(shps)

        if shm_obj:
            print('count:{}'.format(count+1))
            shm_obj.value = count + 1
            print('shm_obj.value count:{}'.format(shm_obj.value))


    ts6 = int((time_pp - time_start))
    # print('total time-cost:%d' % ts6)
    logging.info('toal time-cost:%d' % ts6)

    return datas_result


#将线段两端按照给定gap延长，包含斜线线段
def extend_wall_line(wall_line,gap):
    [[x1,y1],[x2,y2]] = wall_line
    dy = abs(y1 - y2)
    dx = abs(x1 - x2)
    length = (dx**2 + dy**2)**0.5
    if x1 < x2:
        x1 -= int(dx*gap/length)
        x2 += int(dx*gap/length)
    else:
        x1 += int(dx*gap/length)
        x2 -= int(dx*gap/length)        
    if y1 < y2:
        y1 -= int(dy*gap/length)
        y2 += int(dy*gap/length)
    else:
        y1 += int(dy*gap/length)
        y2 -= int(dy*gap/length)
    extend_wall_line_b = [[x1,y1],[x2,y2]]
    return extend_wall_line_b

#获取线段两个端点及两端延长后与其他线段的交点，取这些点组成的线段中最长的线段作为该扩展的线段
def get_extend_wall(wall_line_original,wall_lines,insects,gap):
    original_points = [point for point in wall_line_original]
    cross_points_o = []
    cross_points_e = []
    wall_candidate = []
    wall_candidate_length = []
    
    wall_line_original_b = extend_wall_line(wall_line_original,gap)
    wall_line_original_b_o = LineString(wall_line_original_b)

    for i in insects[0]:
        wall_line_insect = wall_lines[i]
        wall_line_insect_b = extend_wall_line(wall_line_insect,gap)
        wall_line_insect_b_o = LineString(wall_line_insect_b)
        cross_point = wall_line_original_b_o.intersection(wall_line_insect_b_o).coords[:][0]
        # print('cross_point:{}'.format(cross_point))
        cross_points_o.append([cross_point[0],cross_point[1]])

    for i in insects[1]:
        wall_line_insect = wall_lines[i]
        wall_line_insect_b = extend_wall_line(wall_line_insect,gap)
        wall_line_insect_b_o = LineString(wall_line_insect_b)
        cross_point = wall_line_original_b_o.intersection(wall_line_insect_b_o).coords[:][0]
        # print('cross_point:{}'.format(cross_point))
        cross_points_e.append([cross_point[0],cross_point[1]])
    candidate_points = original_points + cross_points_e
    cross_points = cross_points_o + cross_points_e

    for wall_line in list(itertools.combinations(candidate_points,2)):
        ([x1,y1],[x2,y2]) = wall_line
        if (x1 > x2) or (x1 == x2 and y1 > y2):
            x1,y1,x2,y2 = x2,y2,x1,y1
        wall_candidate.append([[x1,y1],[x2,y2]])
        wall_candidate_length.append(LineString([[x1,y1],[x2,y2]]).length)
    #获取最长的墙
    wall_line_extend = wall_candidate[wall_candidate_length.index(max(wall_candidate_length))]
    #删除小线段
    wall_line_new = delete_overstep_short_line(wall_line_extend,cross_points,gap)

    return wall_line_new

#删除因墙厚而超出交点的小线段
def delete_overstep_short_line(wall_line,cross_points,gap):
    original_extend_points = [point for point in wall_line]
    start_point_candidate_lines = [[original_extend_points[0],point] for point in cross_points]
    end_point_candidate_lines = [[point,original_extend_points[1]] for point in cross_points]

    start_point_candidate_lines_length = [LineString(line).length for line in start_point_candidate_lines]
    end_point_candidate_lines_length = [LineString(line).length for line in end_point_candidate_lines]

    start_point_candidate_line = start_point_candidate_lines[start_point_candidate_lines_length.index(min(start_point_candidate_lines_length))]
    end_point_candidate_line = end_point_candidate_lines[end_point_candidate_lines_length.index(min(end_point_candidate_lines_length))]

    if LineString(start_point_candidate_line).length >= gap:
        start_point = start_point_candidate_line[0]
    else:
        start_point = start_point_candidate_line[1]
    if LineString(end_point_candidate_line).length >= gap:
        end_point = end_point_candidate_line[1]
    else:
        end_point = end_point_candidate_line[0]
    wall_line_new = [start_point,end_point]

    return wall_line_new

#删除本身相交的两个墙超出的小线段
def get_shorten_wall(wall_line_original,wall_lines,insects,gap):
    cross_points = []
    
    wall_line_original_o = LineString(wall_line_original)

    for i in insects[0]:
        wall_line_insect = wall_lines[i]
        wall_line_insect_o = LineString(wall_line_insect)
        cross_point = wall_line_original_o.intersection(wall_line_insect_o).coords[:][0]
        # print('cross_point:{}'.format(cross_point))
        cross_points.append([cross_point[0],cross_point[1]])

    wall_line_new = delete_overstep_short_line(wall_line_original,cross_points,gap)

    return wall_line_new    
    
#找到符合条件的平行墙，并进行合并处理(递归)
def get_translation_wall(wall_lines,gap):
    translation_wall_lines = wall_lines.copy()
    # candidate_indexes = []
    finded = False
    for i,wall_line_i in enumerate(wall_lines):
        if finded:
            break
        # line_i = LineString(extend_wall_line(wall_line_i,gap))
        line_i = LineString(wall_line_i)

        for j,wall_line_j in enumerate(wall_lines):
            if i < j:
                dxi = abs(wall_line_i[0][0] - wall_line_i[1][0])
                dyi = abs(wall_line_i[0][1] - wall_line_i[1][1])
                dxj = abs(wall_line_j[0][0] - wall_line_j[1][0])
                dyj = abs(wall_line_j[0][1] - wall_line_j[1][1])
                if (dxi == 0 and dxj == 0) or (dxi != 0 and dxj != 0 and dyi*dxj == dyj*dxi):#平行的两条线
                    # line_j = LineString(extend_wall_line(wall_line_j,gap))
                    line_j = LineString(wall_line_j)
                    if not line_i.intersects(line_j) and line_i.distance(line_j) <= gap: #不相交在gap距离内
                        length_i = line_i.length
                        length_j = line_j.length
                        if length_i > length_j:
                            long_line = wall_line_i
                            trans_line = wall_line_j
                            # dargs = [dxi*2*gap/length_i,dyi*2*gap/(length_i-2*gap),dxj*2*gap/(length_j-2*gap),dyj*2*gap/(length_j-2*gap)]
                            dargs = [dxi*2*gap/length_i,dyi*2*gap/length_i,dxj*2*gap/length_j,dyj*2*gap/length_j]

                        else:
                            long_line = wall_line_j
                            trans_line = wall_line_i
                            dargs = [dxj*2*gap/length_j,dyj*2*gap/length_j,dxi*2*gap/length_i,dyi*2*gap/length_i]
                        new_get_translation_wall = get_new_translation_wall(long_line,trans_line,dargs,gap)
                        translation_wall_lines.remove(wall_line_i)
                        translation_wall_lines.remove(wall_line_j)
                        translation_wall_lines += new_get_translation_wall
                        finded = True
                        break
    if finded == False:
        return translation_wall_lines
    else:
        return get_translation_wall(translation_wall_lines,gap)

#平移短墙并与长墙合并
def get_new_translation_wall(long_line,trans_line,dargs,gap):
    [dxl,dyl,dxt,dyt] = dargs
    [[x1,y1],[x2,y2]] = trans_line
    long_line_e = LineString(extend_wall_line(long_line,LineString(long_line).length+2*gap))
    long_line_b = LineString(long_line)

    if (dxl == 0 and dxt == 0) or trans_line[0][1] < trans_line[1][1]:
        vertical_line1 = [[x1 - dyl, y1 + dxl], [x1 + dyl, y1 - dxl]]
        vertical_line2 = [[x2 - dyl, y2 + dxl], [x2 + dyl, y2 - dxl]]

    elif (dyl == 0 and dyt == 0) or trans_line[0][1] >= trans_line[1][1]:
        vertical_line1 = [[x1 - dyl, y1 - dxl], [x1 + dyl, y1 + dxl]]
        vertical_line2 = [[x2 - dyl, y2 - dxl], [x2 + dyl, y2 + dxl]]        
        
    vertical_line1_b = LineString(vertical_line1)
    vertical_line2_b = LineString(vertical_line2)
    cross_point1 = long_line_e.intersection(vertical_line1_b).coords[:]
    cross_point2 = long_line_e.intersection(vertical_line2_b).coords[:]
    transed_wall = [cross_point1[0],cross_point2[0]]
    transed_wall_b = LineString(transed_wall)

    all_endpoint = [long_line[0],long_line[1],transed_wall[0],transed_wall[1]]
    wall_candidate = []
    wall_candidate_length = []
    for wall_line in list(itertools.combinations(all_endpoint,2)):
        ([x1,y1],[x2,y2]) = wall_line
        if (x1 > x2) or (x1 == x2 and y1 > y2):
            x1,y1,x2,y2 = x2,y2,x1,y1
        wall_candidate.append([[x1,y1],[x2,y2]])
        wall_candidate_length.append(LineString([[x1,y1],[x2,y2]]).length)

    new_get_translation_wall = [wall_candidate[wall_candidate_length.index(max(wall_candidate_length))]]

    return new_get_translation_wall

#平移平行墙，补上应墙厚导致的小缺口并删除应墙厚导致的超出线段
def get_connect_wall_line(wall_lines):
    gap = 15
    #在平行墙之间的距离小于gap，则平移短墙至与长墙合并
    wall_lines = get_translation_wall(wall_lines,gap=int(gap/2))

    #判断延长墙线有哪些是相交的(不包含原本相交)，并按下标存成列表
    extend_intersect_line_new = []
    for i,line_current in enumerate(wall_lines):
        insect_line_e = []
        insect_line_o = []
        line_current_o = LineString(line_current)
        extend_wall_line_b = extend_wall_line(line_current,gap)
        line_current_e = LineString(extend_wall_line_b)
        for j,line_trave in enumerate(wall_lines):
            if i == j:
                continue
            line_trave_o = LineString(line_trave)
            extend_wall_line_b = extend_wall_line(line_trave,gap)
            line_trave_e = LineString(extend_wall_line_b)
            if line_current_o.intersects(line_trave_o):
                insect_line_o.append(j)
            elif line_current_e.intersects(line_trave_e):
                insect_line_e.append(j)
        insect_line = [insect_line_o,insect_line_e]
        extend_intersect_line_new.append(insect_line)

    #找到与其他线段相交的最小延长距离的延长线段,并删除因墙厚超出的小线段
    wall_lines_extend = []
    for i,insects in enumerate(extend_intersect_line_new):
        wall_line_original = wall_lines[i]
        if len(insects[0]) == 0 and len(insects[1]) == 0:
            wall_line_new = wall_line_original
        elif len(insects[0]) != 0 and len(insects[1]) == 0:
            wall_line_new = get_shorten_wall(wall_line_original,wall_lines,insects,gap)
        else:
            wall_line_new = get_extend_wall(wall_line_original,wall_lines,insects,gap)
        wall_lines_extend.append(wall_line_new)

    return wall_lines_extend


def predict_server(gpu_flag, model_file_dir, classes_path, data_dict, transform_scale, save_path, shm_obj, err_queue,result_queue):
    try:
        result = predict(gpu_flag, model_file_dir, classes_path, data_dict, transform_scale,save_path,shm_obj)
        print('result:{}'.format(result))
        result_queue.put(json.dumps(result))
    except Exception as e:
        err_queue.put(repr(e))


if __name__ == '__main__':
    ###load data
    data_file_name = 'x-input.png'
    model_file_name = 'xining_first_test'
    #model_file_name = '3_shouguang_firstbatch_modelparam.pkl'



    basedir = os.getcwd()
    model_file_dir = basedir + '/support_files/trained_weights/' + model_file_name
    classes_path = basedir + '/support_files/class_map/'
    #data_file = basedir + '/data_datasets/datas/test_samples/010280008/' + data_file_name
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")

    folders = genfromtxt(basedir + '/data_datasets/datas' + '/test.txt', dtype='str') #获取数据目录(数据最后一层文件夹名称)，形成字符串列表
    for fd in folders:
        data_file = basedir + '/data_datas/datas' + fd + '/' + data_file_name
        save_path = basedir + '/run_logs/inference_logs/' + data_file_name[:-4] + '_' + time_stamp + fd + '/'
        #save_path = os.path.join(basedir,'TR_master','run_logs','inference_logs',data_file_name[:-4])

        if not os.path.exists(save_path):
            # os.makedirs(save_path)
            os.makedirs(save_path)
        # buffer_range=[1,15] 
        transform_scale = [0,0,1,0]
        gpu_flag = True if torch.cuda.is_available() else False
        predict_o(gpu_flag,model_file_dir, classes_path, data_file, transform_scale,save_path)
        

