from typing import Any, Optional
import torch.nn.functional as F

from torchvision.models._utils import _ovewrite_value_param, IntermediateLayerGetter
from torchvision.models.resnet import ResNet, resnet50, ResNet50_Weights
from torchvision.models.segmentation.fcn import FCNHead, FCN, FCN_ResNet50_Weights
from peft import LoraConfig, get_peft_model
from svf import * 

def _fcn_resnet(
    backbone: ResNet,
    num_classes: int,
    aux: Optional[bool],
) -> FCN:
    return_layers = {"layer1" : "layer1", "layer2" : "layer2", "layer3" : "layer3", "layer4" : "layer4"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = FCNHead(2048, num_classes)
    return FCN(backbone, classifier, aux_classifier)
    
def fcn_resnet50(
    *,
    weights: Optional[FCN_ResNet50_Weights] = None,
    model_path = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> FCN:

    weights = FCN_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, model_dir = model_path))

    return model

class ResNet_linear(nn.Module):
    def __init__(self, method, num_classes, model_path):
        super().__init__()
        model = fcn_resnet50( weights='COCO_WITH_VOC_LABELS_V1', model_path = model_path)
        self.encoder = model.backbone
        self.method = method
        if method == "svf" : 
            self.encoder = resolver(self.encoder)
        if method == "lora" : 
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["conv1", "conv3"],
                lora_dropout=0.1,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, config)
        if method == "multilayer" : 
            self.in_channels = 2048 + 1024 + 512 + 256
        else : 
            self.in_channels = 2048 
        self.decoder = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.bn = nn.SyncBatchNorm(self.in_channels)

    def forward(self, x): 
        x = F.interpolate(x, size=[1024,1024], mode='bilinear', align_corners=False)
        if self.method in ["linear", "multilayer"] : 
            with torch.no_grad():
                if self.method == "linear" : 
                    x = self.encoder(x)["layer4"] 
                else : 
                    x =[F.interpolate(self.encoder(x)["layer1"], size=[128,128], mode='bilinear', align_corners=False),self.encoder(x)["layer2"],self.encoder(x)["layer3"],self.encoder(x)["layer4"]]
                    x = torch.cat(x,dim = 1)
        else : 
            x = self.encoder(x)["layer4"] #["out"]  
        x = self.bn(x)
        return self.decoder(x)
