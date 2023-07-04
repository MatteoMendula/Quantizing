import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import quantization_utils as quantization_utils 
from pytorch_quantization import tensor_quant
import torchvision.datasets as datasets
from tqdm import tqdm
# £import torchvision.transforms.PILToTensor as PILToTensor
import torchvision.transforms as transforms

class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None, weights="IMAGENET1K_V2"):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(weights=None if backbone_path else weights)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(weights=None if backbone_path else weights)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, backbone=ResNet('resnet50')):
        super().__init__()

        self.feature_extractor = backbone

        self.label_num = 81  # number of COCO classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).reshape(s.size(0), 4, -1), c(s).reshape(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs

class SSD300Head(nn.Module):
    def __init__(self, original_ssd):
        super().__init__()
        self.original_ssd = original_ssd
        model_iterable = model.children()
        inside_resnet = next(model_iterable).children()
        sequential = next(inside_resnet).children()
        head_list = []
        for index, module in enumerate(sequential):
            if index < 5:
                head_list.append(module)
        self.head_model = nn.Sequential(*head_list)

    def forward(self, x):
        head_output = self.head_model(x)
        head_outpu_q, scale, zero_point = quantization_utils.quantize_tensor_torch(head_output)
        return head_outpu_q, scale, zero_point
    
    def parse_to_onnx(self):
        input = [torch.randn((1,3,300,300)).to("cuda")]
        model = self.eval().to("cuda")
        traced_model = torch.jit.trace(model, input)    
        torch.onnx.export(traced_model,  # model being run
                            input,  # model input (or a tuple for multiple inputs)
                            "./models/head.onnx",  # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=13,  # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names=['input'],  # the model's input names
                            output_names=['output0', 'output1', 'output2'],  # the model's output names]
                        )
        
    def parse_to_trt(self, precision = 'fp32'):
        import subprocess
        print("[{}] parse_to_trt".format(precision))
        cmd = '/home/matteo/TensorRT-8.6.1.6/bin/trtexec --onnx=./models/head.onnx --saveEngine=./models/head_fp32.trt'
        if precision == 'fp16':
            cmd = '/home/matteo/TensorRT-8.6.1.6/bin/trtexec --onnx=./models/head.onnx --saveEngine=./models/head_fp16.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16'
        output = subprocess.check_call(cmd.split(' '))
        print(output)

class SSD300Tail(nn.Module):
    def __init__(self, original_ssd):
        super().__init__()
        self.original_ssd = original_ssd
        model_iterable = self.original_ssd.children()
        inside_resnet = next(model_iterable).children()
        sequential = next(inside_resnet).children()
        tail_list = []
        for index, module in enumerate(sequential):
            if index >= 5:
                tail_list.append(module)
        self.tail_model = nn.Sequential(*tail_list)

    def quantize(self):
        pass

    def forward(self, x):
        x = self.tail_model(x)
        detection_feed = [x]
        for l in self.original_ssd.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.original_ssd.bbox_view(detection_feed, self.original_ssd.loc, self.original_ssd.conf)
        return locs, confs

if __name__ == "__main__":
    import time
    import inference_utils as utils
    from SSD_creator import SSD_creator
    import argparse

    parser = argparse.ArgumentParser(description='Parser for SSD300')
    parser.add_argument('-p', '--precision', default="fp32", type=str)
    args = vars(parser.parse_args())

    precision = args['precision']

    ssd_creator = SSD_creator("fp32", "IMAGENET1K_V2")
    model = ssd_creator.ssd300
    # model.eval()
    model_head = model.feature_extractor.feature_extractor[:6]
    model_layer3 = model.feature_extractor.feature_extractor[6]
    head = SSD300Head(model)
    tail = SSD300Tail(model)

    # save head to onnx and trt
    # head.parse_to_onnx()
    # head.parse_to_trt(precision = precision)

    head = head.eval().cuda()
    tail = tail.eval().cuda()
    model = model.eval().cuda()

    inputs = [utils.prepare_input("./kitti_1.png")]
    inputs = [utils.prepare_input("./kitchen.jpg")]
    print("inputs shape:", inputs[0].shape)
    x = utils.prepare_tensor(inputs, should_half = False)

    jpeg_compression_size = utils.jpeg_size_compression("./kitchen.jpg", 90)
    print("JPEG compression size:", jpeg_compression_size)
    # detemine size of x in bytes
    image_size = x.element_size() * x.nelement()
    print("image tensor size:", image_size)

    x = x.cuda()
    start = time.time()
    original_model_output = model(x)
    print("original model time:", time.time() - start)

    start = time.time()    
    head_output, scale, zero_point = head(x)
    print("head_time", time.time() - start)
    print("head_output shape:", head_output.shape)
    print("head_output n_times jpeg:", head_output.element_size() * head_output.nelement()/jpeg_compression_size)
    x_test = torch.randn(1, 12, 75, 75).to(torch.int8)
    print("bottleneck target output sizen_times jpeg: ", x_test.element_size() * x_test.nelement()/jpeg_compression_size)

    head_output_q = head_output.to(torch.int8)
    print("head_output int8 n_times jpeg:", head_output_q.element_size() * head_output_q.nelement()/jpeg_compression_size)
    
    head_output = quantization_utils.dequantize_tensor(head_output, scale.item(), zero_point.item())
    splitted_model_output = tail(head_output)

    results_per_input = utils.decode_results((splitted_model_output[0], splitted_model_output[1]))
    best_results_per_input_trt = [utils.pick_best(results, 0.40) for results in results_per_input]
    # Visualize results bare TensorRT
    classes_to_labels= utils.get_coco_object_dictionary()
    print("best_results_per_input_trt", best_results_per_input_trt)
    utils.plot_results(best_results_per_input_trt, inputs, classes_to_labels)

