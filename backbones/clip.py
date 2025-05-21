import torch
import torch.nn.functional as F
from typing import Sequence, Tuple, Union
from functools import partial
import clip
import numpy as np
from svf import * 

class Clip_visual(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        device = "cuda"
        clip_model, preprocess = clip.load("ViT-B/16", device=device, download_root= model_path)
        clip_model.to(device = device)
        self.encoder = clip_model.visual
        self.patch_size = 16

    def _get_intermediate_layers_not_chunked(self, x, n=1):
            x = self.encoder.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            pos_token = self.encoder.positional_embedding[0,:].unsqueeze(0)
            sq_positial_embedding = self.encoder.positional_embedding[1:,:].reshape(14, 14, -1).unsqueeze(0).permute(0,3,1,2)
            dim = int(np.sqrt(x.shape[1]-1))
            sq_positial_embedding = F.interpolate(sq_positial_embedding, size=[dim,dim], mode='bilinear', align_corners=False).squeeze().permute(1,2,0).reshape(dim*dim,-1)
            sq_positial_embedding = torch.cat([pos_token,sq_positial_embedding])
            x = x + sq_positial_embedding.to(x.dtype)
            #x = x + self.encoder.positional_embedding.to(x.dtype)
            x = self.encoder.ln_pre(x)
            x = x.permute(1, 0, 2)

            output, total_block_len = [], len(self.encoder.transformer.resblocks)
            blocks_to_take = n
            for i, blk in enumerate(self.encoder.transformer.resblocks):
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x.permute(1, 0, 2))
            assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
            return output
    
    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = [11],  # Layers or n last layers to take
            reshape: bool = True,
            norm=True,
        ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
            
            outputs = self._get_intermediate_layers_not_chunked(x, n)
            if norm:
                outputs = [self.encoder.ln_post(out) for out in outputs]
            class_tokens = [out[:, 0] for out in outputs]
            outputs = [out[:, 1:] for out in outputs]
            if reshape:
                B, _, w, h = x.shape
                outputs = [
                    out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                    for out in outputs
                ]
            return tuple(outputs)


class CLIP_linear(nn.Module):
    def __init__(self, method, num_classes, model_path):
        super().__init__()
        model = Clip_visual(model_path)
        self.method = method  
        if method == "multilayer" :
            n = [8,9,10,11]
        else : 
            n = [11]
        model.encoder.forward = partial(
            model.get_intermediate_layers,
            n=n,
            reshape=True,
        )
        self.encoder = model.encoder
        self.encoder.attnpool = nn.Identity()
        print(self.encoder)
        if method == "svf" : 
            self.encoder = self.encoder.to(torch.float32)
            self.encoder = resolver(self.encoder)
        if method == "finetune" : 
            self.encoder = self.encoder.to(torch.float32)
        if method == "lora" : 
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["out_proj"],
                #target_modules=["conv1", "conv3"],
                lora_dropout=0.1,
                bias="none",
            )
            self.encoder = self.encoder.to(torch.float32)
            self.encoder = get_peft_model(self.encoder, config)
        if method == "multilayer" : 
            self.in_channels = 768*4
        else : 
            self.in_channels = 768 #2048
        self.bn = nn.SyncBatchNorm(self.in_channels) 
        self.decoder = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)

    def forward(self, x): 
        if self.method in ["linear", "multilayer"] : 
            with torch.no_grad():
                x = F.interpolate(x, size=[224,224], mode='bilinear', align_corners=False)
                x = x.to(torch.float16)
                x = self.encoder(x)
        else : 
            x = F.interpolate(x, size=[224,224], mode='bilinear', align_corners=False)
            x = self.encoder(x)        
        x = torch.cat(x,dim=1)
        x = self.bn(x)
        x = x.to(torch.float32)
        return self.decoder(x)
