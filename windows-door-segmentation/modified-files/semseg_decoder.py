# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
import torchvision.models as models

# def resnet18_classifier():
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         num_layers: int,
#         sigmoid_output: bool = False,
#     ) -> None:
#         super().__init__()


        
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )
#         self.sigmoid_output = sigmoid_output


#         self.resnet18 = models.resnet18(pretrained=True)
        
        

#         return resnet18
#         # resnet18_classifier = resnet18_classifiy()
#         # resnet18_classifier.to(device)

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

#         x = nn.Linear(512,3)
#         resnet18.fc = torch.nn.Linear(512,3)


#         if self.sigmoid_output:
#             x = F.sigmoid(x)
#         return x
        

def build_SemSegDecoder(
    image_embeddings_dim: int, #256
    activation: Type[nn.Module] = nn.GELU,
    n_classes: int = 2, #2
    classify_layer_num: int = 3, #3
    checkpoint=None,
    is_train: bool = True,
):
    ssd = SemSegDecoder_conv(
        image_embeddings_dim=image_embeddings_dim,
        activation=activation,
        n_classes=n_classes,
        classify_layer_num=classify_layer_num,
    )

    # ssd = resnet18_classifiy()

    if is_train:
        ssd.train()
    else:
        ssd.eval()
        
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        ssd.load_state_dict(state_dict)

    return ssd


class SemSegDecoder(nn.Module):
    def __init__(
        self,
        image_embeddings_dim: int, #256
        activation: Type[nn.Module] = nn.GELU,
        n_classes: int = 2, #2
        classify_layer_num: int = 3, #3
    ) -> None:
        """
        Predicts masks given an image, using a
        MLP behand ConvTranspose2d architecture.

        Arguments:
          image_embeddings_dim (int): the channel dimension of the transformer
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()

        #输出比例缩放，图像扩大，通道减少：image_embeddings_dim：256 ==> image_embeddings_dim//16：16,[64,256,64,64]->[64,16,1024,1024]
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(image_embeddings_dim, image_embeddings_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(image_embeddings_dim // 2),
            activation(),
            nn.ConvTranspose2d(image_embeddings_dim//2, image_embeddings_dim // 4, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(image_embeddings_dim // 4, image_embeddings_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(image_embeddings_dim // 8),
            activation(),
            nn.ConvTranspose2d(image_embeddings_dim // 8, image_embeddings_dim // 16, kernel_size=2, stride=2),
            activation(),
        )

        #3层全连接网络，输入256通道，隐藏层256通道，输出n_classes通道
        self.classify_prediction_head = MLP(
            image_embeddings_dim*65536, image_embeddings_dim*65536, n_classes, classify_layer_num
        )
        # self.classify_prediction_head = resnet18_classifier(
        #     image_embeddings_dim*65536, image_embeddings_dim*65536, n_classes, classify_layer_num
        #     )

        for m in self.modules():
            if isinstance(m,nn.ConvTranspose2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(
        self,
        image_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """
        Predict masks given image.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
        Returns:
          torch.Tensor: batched predicted masks
        """

        # vit feature to classify feature [1,256,64,64] -> [1,16,1024,1024]
        upscaled_embedding = self.output_upscaling(image_embeddings) #torch.Size([1, 16, 1024, 1024])

        # classify prediction head [1,16,1024,1024] -> [1, n_classes, 1024, 1024]
        b, c, h, w = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.view(b, c* h * w)

        masks = self.classify_prediction_head(upscaled_embedding) #torch.Size([1, n_classes, 1024, 1024])
        masks = masks.view(b, -1, h, w)

        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            input_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks




class SemSegDecoder_conv(nn.Module):
    def __init__(
        self,
        image_embeddings_dim: int, #256
        activation: Type[nn.Module] = nn.GELU,
        n_classes: int = 2, #2
        classify_layer_num: int = 3, #3
    ) -> None:
        """
        Predicts masks given an image, using a ConvTranspose2d architecture.

        Arguments:
          image_embeddings_dim (int): the channel dimension of the transformer
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()

        #输出比例缩放，图像扩大，通道减少：image_embeddings_dim：256 ==> n_classes：2
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(image_embeddings_dim, image_embeddings_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(image_embeddings_dim // 2),
            activation(),
            nn.ConvTranspose2d(image_embeddings_dim//2, image_embeddings_dim // 4, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(image_embeddings_dim // 4, image_embeddings_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(image_embeddings_dim // 8),
            activation(),
            nn.ConvTranspose2d(image_embeddings_dim // 8, n_classes, kernel_size=2, stride=2),
        )

        for m in self.modules():
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(
        self,
        image_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """
        Predict masks given image.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
        Returns:
          torch.Tensor: batched predicted masks
        """

        # vit feature to classify feature [1,256,64,64] -> [1,n_classes,1024,1024]
        masks = self.output_upscaling(image_embeddings) #torch.Size([1, n_classes, 1024, 1024])

        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            input_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks


class MaskDecoder_reference(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int, #256
        transformer: nn.Module, #TwoWayTransformer
        num_multimask_outputs: int = 3, #3
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3, #3
        iou_head_hidden_dim: int = 256, #256
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim #256
        self.transformer = transformer #TwoWayTransformer

        self.num_multimask_outputs = num_multimask_outputs #3

        self.iou_token = nn.Embedding(1, transformer_dim) #shape:[1,256]
        self.num_mask_tokens = num_multimask_outputs + 1 #4，with background
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim) #shape:[4,256]

        #输出比例缩放，图像扩大，通道减少：transformer_dim：256 ==> transformer_dim//8：32
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        #4个3层网络，每个网络全连接输入256通道，隐藏层256通道，输出32通道，
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        #3层全连接网络，输入256通道，隐藏层256通道，输出4通道
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor, #torch.Size([1, 256, 64, 64])
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor, #torch.Size([64, 2, 256])
        dense_prompt_embeddings: torch.Tensor, #torch.Size([64, 256, 64, 64])
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) #shape:[5,256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) #[64,5,256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) #[64,7,256]
        print(f'tokens.shape:{tokens.shape}') #torch.Size([64, 7, 256])

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) #torch.Size([64, 256, 64, 64])
        src = src + dense_prompt_embeddings #torch.Size([64, 256, 64, 64])
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) #torch.Size([64, 256, 64, 64])
        b, c, h, w = src.shape
        print(f'src.shape:{src.shape}') #torch.Size([64, 256, 64, 64])
        print(f'pos_src.shape:{pos_src.shape}') #torch.Size([64, 256, 64, 64])

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        print(f'src.shape 2:{src.shape}') #torch.Size([64, 4096, 256])
        print(f'hs.shape:{hs.shape}') #torch.Size([64, 7, 256])

        iou_token_out = hs[:, 0, :] #torch.Size([64, 256])
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] #torch.Size([64, 4, 256])
        print(f'iou_token_out.shape 2:{iou_token_out.shape}') #torch.Size([64, 256])
        print(f'mask_tokens_out.shape:{mask_tokens_out.shape}') #torch.Size([64, 4, 256])

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        print(f'src.shape 3:{src.shape}') #torch.Size([64, 256, 64, 64])
        print(f'upscaled_embedding.shape:{upscaled_embedding.shape}') #torch.Size([64, 32, 256, 256])

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        print(f'self.num_mask_tokens:{self.num_mask_tokens}') #4
        print(f'hyper_in.shape:{hyper_in.shape}') #torch.Size([64, 4, 32])

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        print(f'masks.shape:{masks.shape}') #torch.Size([64, 4, 256, 256])

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out) #torch.Size([64, 4])
        print(f'iou_pred.shape:{iou_pred.shape}') #torch.Size([64, 4])

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        print("==============START MLP forward==============")
        print("self.layers:", self.layers)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
