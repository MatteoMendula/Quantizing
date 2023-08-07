import os
import torch
import argparse
from utils.apex_utils import network_to_half
from utils.inference_utils import *
from model import SSD300, ResNet
import torch.nn as nn

class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduction='none')

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float()*sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        #print(con.shape, mask.shape, neg_mask.shape)
        closs = (con*((mask + neg_mask).float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        return ret


class SSD_creator():
    def __init__(self, precision, pretrained_weigths):
        self.precision = precision
        self.pretrained_weigths = pretrained_weigths
        self.ssd300 = None

        self.load_model()

    def load_weights(self):
        if not os.path.exists("./models/ssd300_weights_{}_{}.pth".format(self.pretrained_weigths, self.precision)):
            print("Loading weights for {} precision".format(self.precision))
            # torch.cuda.empty_cache() 
            ssd300 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=self.precision)
            torch.save(ssd300.state_dict(),"./models/ssd300_weights_{}_{}.pth".format(self.pretrained_weigths, self.precision))
            print("Weights saved")
        else:
            print("Weights already exist for {} precision".format(self.precision))

    def load_model(self):
        self.load_weights()
        ssd300 = SSD300(backbone=ResNet())
        ssd300.load_state_dict(torch.load("./models/ssd300_weights_{}_{}.pth".format(self.pretrained_weigths, self.precision)))
        if self.precision == "fp16":
            self.ssd300 = network_to_half(ssd300.cuda())
            print("network converted to fp16")
        else:
            self.ssd300 = ssd300#.cuda()
            print("network converted to fp32")

    def parse_to_onnx(self):
        input = [torch.randn((1,3,300,300)).to("cuda")]
        if self.precision == 'fp16':
            input = [torch.randn((1,3,300,300)).to("cuda").half()]
        model = self.ssd300.eval().to("cuda")
        traced_model = torch.jit.trace(model, input)    
        torch.onnx.export(traced_model,  # model being run
                            input,  # model input (or a tuple for multiple inputs)
                            "./models/ssd_{}.onnx".format(self.precision),  # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=13,  # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names=['input'],  # the model's input names
                            output_names=['output'])
        
    def parse_to_trt(self):
        import subprocess
        cmd = '/home/matteo/TensorRT-8.6.1.6/bin/trtexec --onnx=./models/ssd_{}.onnx --saveEngine=./models/ssd_{}.trt'.format(self.precision, self.precision)
        output = subprocess.check_call(cmd.split(' '))
        print(output)

def get_parameters():
    parser = argparse.ArgumentParser(description='Parser for SSD300')
    parser.add_argument('-p', '--precision', default="fp32", type=str)
    parser.add_argument('-w', '--pretrained_weigths', default="IMAGENET1K_V2", type=str)
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    args = get_parameters()
    ssd_creator = SSD_creator(args['precision'], args['pretrained_weigths'])
    ssd_creator.parse_to_onnx()
    ssd_creator.parse_to_trt()
