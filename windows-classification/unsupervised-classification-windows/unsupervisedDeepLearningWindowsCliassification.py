from transformers import CLIPProcessor, CLIPModel
import torch
import os
# device = "cuda"
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
# .to(device)

def embed_images(img_list):
  with torch.no_grad():
    images = processor(
        # text=None, images=img_list, return_tensors='pt')['pixel_values'].to(device)
        text=None, images=img_list, return_tensors='pt')['pixel_values']
    return model.get_image_features(images)

from PIL import Image

# 指定要读取的文件夹路径  
folder_path = r"C:\Users\yiper\Desktop\20230904_142329_test\windows\data\all" 

  
# 获取文件夹下所有文件的列表  
files = os.listdir(folder_path)  
  
# 遍历文件列表，筛选出图片文件  
imgs = []  
for file in files:  
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):  
        file = folder_path + os.sep + file
        imgs.append(Image.open(file))  

# imgs = [Image.open(r'data\all\1.png'), Image.open('2.png'), Image.open('3.png'), Image.open('4.png'), Image.open('5.png'),
# Image.open('6.png'), Image.open('7.png'), Image.open('8.png'), Image.open('9.png'), Image.open('10.png'),
# Image.open('11.png'), Image.open('12.png'), Image.open('13.png'), Image.open('14.png'), Image.open('15.png'),
# Image.open('16.png'), Image.open('17.png'), Image.open('18.png'), Image.open('19.png'), Image.open('20.png')]

all_emb = embed_images(imgs)
all_emb = all_emb.numpy()







from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)

kmeans.fit(all_emb)
y_kmeans = kmeans.predict(all_emb)
print(y_kmeans)






print(all_emb[0])
print(len(all_emb[0]))
# print(embedded)



# def get_single_embedding_tensor_of(concepts):
#   tensor_list = []

#   for filename in [Path.cwd() / f'{concept}.pt' for concept in concepts]:
#       tensor = torch.load(filename)
#       tensor_list.append(tensor)

#   # Normalize by vector l2_norm to have them comparable
#   embedded = torch.cat(tensor_list, dim=0)
#   l2_norm = torch.norm(embedded, dim=1, keepdim=True)
#   return embedded / l2_norm

# all_emb = get_single_embedding_tensor_of(concepts)

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# import numpy as np
# concepts = ['cat', 'dog', 'dinosaur', 'elephant']
# tsne = TSNE(n_components=2)
# tsne_embeddings = tsne.fit_transform(all_emb.cpu().numpy())
import csv

# data = [['nameservers','panel'], ['nameservers','panel']]

# with open('tmp_file2dim2.txt', 'w') as f:
#     csv.writer(f, delimiter='\t').writerows(tsne_embeddings)

with open('tmp_file2dim2.txt', 'w') as f:
    csv.writer(f, delimiter='\t').writerows(all_emb)    




# num_classes = 4
# colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

# class_labels = [0,1,2,3]
# color_list = [colors[label] for label in class_labels]

# patches = [Patch(color=colors[i], label=cls) for i, cls in enumerate(concepts)]

# fig = plt.figure(figsize=(10, 8))
# plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=color_list)
# plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.80))
# plt.show()
