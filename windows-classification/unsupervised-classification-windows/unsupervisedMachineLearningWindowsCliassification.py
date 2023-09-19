#简单演示k-means算法
# %matplotlib inline
import matplotlib.pyplot as plt
# import seaborn as sns;sns.set()
import numpy as np
# from sklearn.datasets.samples_generator import make_blobs
# x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
import cv2
import os  
import pandas as pd
  
# 指定要读取的文件夹路径  
folder_path = r"C:\Users\yiper\Desktop\20230904_142329_test\windows\data\all" 

  
# 获取文件夹下所有文件的列表  
files = os.listdir(folder_path)  
  
# 遍历文件列表，筛选出图片文件  
image_files = []  
for file in files:  
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):  
        file = folder_path + os.sep + file
        image_files.append(file)  
  
# # 输出所有图片文件的文件名  
# for image_file in image_files:  
#     print(image_file)

# 训练集
XX_train = []
for i in image_files:
    # 读取图像
    # print i
    # i = "C:\Users\yiper\Desktop\20230904_142329_test\windows\data\all\" + i
    image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 图像像素大小一致
    img = cv2.resize(image, (256, 256),
                     interpolation=cv2.INTER_CUBIC)

    # # 计算图像直方图并存储至X数组
    # hist = cv2.calcHist([img], [0, 1], None,
    #                     [256, 256], [0.0, 255.0, 0.0, 255.0])

    XX_train.append(((img / 255).flatten()).tolist())



x = XX_train

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)

kmeans.fit(x)
y_kmeans = kmeans.predict(x)



# print(y_kmeans)
# [0 1 2 2 1 1 1 2 2 2 2 2 2 2 2 0 0 2 0 1 1 1]
# y_predict_label = 
#[0 1 2 2 1 1 1 2 2 2 2 2 2 2 2 0 0 2 0 1 1 1]
# y_true_label = 



#可视化
plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='k', s=200, alpha=0.5)
print()