
from torchvision.io.image import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import time
import torch
from torch import Tensor, nn
from typing import List, Tuple

def parse_to_onnx(model, input_size=False):
    if not input_size:
        input = [torch.randn((1,3,300,300)).to("cuda")]
    else:
        input = [torch.randn((1,3,input_size[0],input_size[1])).to("cuda")]
    model = model.eval().to("cuda")
    traced_model = torch.jit.trace(model, input)    
    torch.onnx.export(traced_model,  # model being run
                        input,  # model input (or a tuple for multiple inputs)
                        "./models/ssd_lite.onnx",  # where to save the model (can be a file or file-like object)
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=13,  # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=['input'],  # the model's input names
                        output_names=['output0', 'output1', 'output2'])  # the model's output names])
    
class Adapter(torch.nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()

    def forward(self, x: List[Tensor]):
        return (x[0]["boxes"], x[0]["labels"], x[0]["scores"])

# Step 1: Initialize model with the best available weights
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.9)
model.cuda()
model.eval()


model_hat = torch.nn.Sequential(model, Adapter())
print("----- saving model -----")
print(model_hat)
parse_to_onnx(model_hat)
print("----- done -----")