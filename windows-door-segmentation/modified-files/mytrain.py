import os,sys
print(f'os.getcwd:{os.getcwd()}')
sys.path.append(os.getcwd())

import time
import numpy as np
import torch
from torch.utils import data
from torch.nn.functional import threshold, normalize
import matplotlib.pyplot as plt

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.modeling.semseg_decoder import build_SemSegDecoder
from segment_anything.utils.data_loader import Load_dataset,Compose,RandomCropToSizeTorch,RandomPaddingToSizeTorch




def train(dataset_path,check_point_path,epoch,batch_size,lr,device):
    st_time = time.time()
    # load data
    input_size=(1024,1024)
    # aug = RandomCropToSizeTorch(input_size)
    aug = Compose([RandomCropToSizeTorch(input_size),RandomPaddingToSizeTorch(input_size)])
    
    data_set = Load_dataset(dataset_path,'train',augmentations=aug,format='txt',input_size=input_size)
    data_loader = data.DataLoader(data_set,batch_size=batch_size,num_workers=8,shuffle=True,pin_memory=True)

    val_set = Load_dataset(dataset_path,'val',augmentations=aug,format='txt',input_size=input_size)
    val_loader = data.DataLoader(val_set,batch_size=1,num_workers=8,pin_memory=True)

    # load model(model head rebuild????)
    sam_model = sam_model_registry['vit_h'](checkpoint=check_point_path)
    sam_model.to(device)



    ssd_model = build_SemSegDecoder(
        image_embeddings_dim=256,
        n_classes=3,)
    # ssd_model.load_state_dict(torch.load(r'D:\VC\SAM\checkpoint\ssd_model_min.pth'))
    ssd_model.to(device)

    # create optimizer for mask decoder
    optimizer = torch.optim.Adam(ssd_model.parameters(),lr=lr)

    # difine loss
    # loss_fn = torch.nn.MSELoss
    loss_fn = torch.nn.functional.cross_entropy
    ld_time = time.time()
    print(f'load data and model time:{ld_time-st_time}')
    epoch_time_last = ld_time
    # train loop
    loss_min = np.inf
    for i in range(epoch):
        # train
        batch_time_last = time.time()
        ssd_model.train()
        for j,datas in enumerate(data_loader):
            # get batch image and label 
            input_image = datas['image'].to(device)
            gt_mask = datas['label'].to(device)
            # input_size = datas['input_size']
            # original_image_size = datas['original_image_size']

            nt_time = time.time()
            # image encoder 
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)
            sam_time = time.time()

            # promt encoder
            with torch.no_grad():
                sparse_embeddings,dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )

            # mask decoder
            low_res_masks,iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # # mask decoder
            masks = ssd_model(image_embedding)
            ssd_time = time.time()
            
            # loss caculate
            upscaled_masks = ssd_model.postprocess_masks(masks,input_size)
            # print(f'upscaled_masks.shape:{upscaled_masks.shape},upscaled_masks.dtype:{upscaled_masks.dtype}')
            # print(f'gt_mask.shape:{gt_mask.shape},gt_mask.dtype:{gt_mask.dtype}')
            loss = loss_fn(upscaled_masks,gt_mask)
            # print(f'epoch:{i},batch:{j},loss:{loss}')

            # loss backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step() 

            batch_time = time.time()
            print(f'epoch:{i},batch:{j},loss:{loss},epoch time:{batch_time-batch_time_last},bld_time:{nt_time - batch_time_last},sam time:{sam_time-nt_time},ssd time:{ssd_time-sam_time},optimizer time:{batch_time-ssd_time}')
            batch_time_last = batch_time

        # val
        ssd_model.eval()
        loss_eval = 0
        for j,datas in enumerate(val_loader):
            # get batch image and label 
            input_image = datas['image'].to(device)
            gt_mask = datas['label'].to(device)
            # input_size = datas['input_size']
            # original_image_size = datas['original_image_size']

            # image encoder 
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)

            # # promt encoder
            # with torch.no_grad():
            #     sparse_embeddings,dense_embeddings = sam_model.prompt_encoder(
            #         points=None,
            #         boxes=None,
            #         masks=None,
            #     )

            # # mask decoder
            # low_res_masks,iou_predictions = sam_model.mask_decoder(
            #     image_embeddings=image_embedding,
            #     image_pe=sam_model.prompt_encoder.get_dense_pe(),
            #     sparse_prompt_embeddings=sparse_embeddings,
            #     dense_prompt_embeddings=dense_embeddings,
            #     multimask_output=False,
            # )

            # # mask decoder
            with torch.no_grad():
                masks = ssd_model(image_embedding)
            
            # loss caculate
            upscaled_masks = ssd_model.postprocess_masks(masks,input_size)
            # print(f'upscaled_masks.shape:{upscaled_masks.shape},upscaled_masks.dtype:{upscaled_masks.dtype}')
            # print(f'gt_mask.shape:{gt_mask.shape},gt_mask.dtype:{gt_mask.dtype}')
            loss = loss_fn(upscaled_masks,gt_mask)
            loss_eval += loss
        print(f'epoch:{i},loss_eval:{loss_eval}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        if loss_eval < loss_min:
            loss_min = loss_eval
            torch.save(ssd_model.state_dict(),r'D:\VC\SAM\checkpoint\ssd_model_min.pth')
        loss_eval = 0
        epoch_end_time = time.time()
        print(f'epoch:{i},epoch time:{epoch_end_time-epoch_time_last}')
        epoch_time_last = epoch_end_time

    # save model
    # torch.save(ssd_model.state_dict(),'checkpoint/ssd_model.pth')

    # test model accuracy
    

if __name__=='__main__':
    dataset_path = r'D:\VC\SAM\TIETU5'
    check_point_path = r'D:\VC\SAM\checkpoint\sam_vit_h_4b8939.pth'
    epoch = 100

    batch_size = 8
    lr = 0.0001

    device = 'cuda:0'
    # device = 'cpu'

    train(dataset_path,check_point_path,epoch,batch_size,lr,device)
