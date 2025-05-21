import os
import sys
import torch.nn as nn 
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from svf import * 


class SAM_linear(nn.Module):
    def __init__(self, method, num_classes, model_repo_path, model_path):
        super().__init__()
        sys.path.insert(0,os.path.join(model_repo_path, "segment-anything"))
        from segment_anything import sam_model_registry

        sam_checkpoint = os.path.join(model_path,"sam_vit_b_01ec64.pth")
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.encoder = sam.image_encoder
        self.method = method
        if method == "svf" : 
            self.encoder = resolver(self.encoder)
        if method == "lora" : 
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["qkv"],
                lora_dropout=0.1,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, config)

        if method == "multilayer" : 
            n = [9,10,11,12]
            self.in_channels = 768*3 + 256
            self.encoder.forward = partial(self.get_intermediate_layers , n = n)
        else : 
            self.in_channels = 256
        self.decoder = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.bn = nn.SyncBatchNorm(self.in_channels)

    def forward(self, x): 
        x = F.interpolate(x, size=[1024,1024], mode='bilinear', align_corners=False)
        if self.method in ["linear", "multilayer"] : 
            with torch.no_grad():
                x = self.encoder(x)
                if self.method == "multilayer" : 
                    x = torch.cat(x,dim=1)
        else : 
            x = self.encoder(x)
        x = self.bn(x)
        return self.decoder(x)
    
    def get_intermediate_layers(self, x, n) :         
        output = []
        x = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        for blk_index, blk in enumerate(self.encoder.blocks) : 
            x = blk(x)
            if blk_index in n : 
                output.append(x.permute(0, 3, 1, 2))
        x = self.encoder.neck(x.permute(0, 3, 1, 2))
        if len(self.encoder.blocks) in n :
            output.append(x)
        return output