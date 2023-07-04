import torchvision.models as models
import torch
import torch.onnx
from torch import Tensor, nn
from typing import List, Tuple

# load the pretrained model
weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
fasterrcnn_resnet50_fpn_v2 = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

print(fasterrcnn_resnet50_fpn_v2)
print("------------------")
print(fasterrcnn_resnet50_fpn_v2.roi_heads.box_predictor)

class Adapter(torch.nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()

    def forward(self, x: List[Tensor]):
        return (x[0]["boxes"], x[0]["labels"], x[0]["scores"])
        

fasterrcnn_hat = torch.nn.Sequential(fasterrcnn_resnet50_fpn_v2, Adapter())

# set the model to inference mode
fasterrcnn_resnet50_fpn_v2.eval()

# create dummy input
x = torch.randn(1, 3, 640, 480, requires_grad=True)

out = fasterrcnn_hat(x)
print(out)
print(type(out))


traced_model = torch.jit.trace(fasterrcnn_hat, x)   
print("model traced")
torch.onnx.export(traced_model, 
                    x,  # model input (or a tuple for multiple inputs)
                    "./models/fasterrcnn_resnet50_fpn_v2.onnx",  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=13,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model's input names
                    output_names=['output0', 'output1', 'output2'])  # the model's output names])

