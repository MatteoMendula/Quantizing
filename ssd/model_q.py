import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import quantization_utils as quantization_utils 
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

    def save_to_onnx(self, path):
        dummy_input = torch.randn(1, 3, 300, 300)
        torch.onnx.export(self, dummy_input, path)

    def forward(self, x):
        return self.head_model(x)
    
class SSD300Tail(nn.Module):
    def __init__(self, original_ssd):
        super().__init__()
        self.original_ssd = original_ssd
        model_iterable = model.children()
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
        locs, confs = model.bbox_view(detection_feed, model.loc, model.conf)
        return locs, confs

if __name__ == "__main__":
    import time
    import inference_utils as utils
    from SSD_creator import SSD_creator
    precision = 32

    ssd_creator = SSD_creator("fp{}".format(precision), "IMAGENET1K_V2")
    model = ssd_creator.ssd300
    model.eval()
    head = SSD300Head(model)
    tail = SSD300Tail(model)
    head.eval()
    tail.eval()

    head = head.cuda()
    tail = tail.cuda()
    model = model.cuda()

    inputs = [utils.prepare_input("./kitchen.jpg")]
    inputs = [utils.prepare_input("http://images.cocodataset.org/val2017/000000037777.jpg")]
    inputs = [utils.prepare_input("http://images.cocodataset.org/val2017/000000252219.jpg")]
    x = utils.prepare_tensor(inputs, False)
    x = x.cuda()

    start = time.time()
    original_model_output = model(x)
    print("original model time:", time.time() - start)

    start = time.time()    
    head_output = head(x)
    print("head_time", time.time() - start)

    head_outpu_q = quantization_utils.quantize_tensor(head_output)
    head_output_dq = quantization_utils.dequantize_tensor(head_outpu_q)

    splitted_model_output = tail(head_output_dq)

    results_per_input = utils.decode_results((splitted_model_output[0], splitted_model_output[1]))
    best_results_per_input_trt = [utils.pick_best(results, 0.40) for results in results_per_input]
    # Visualize results bare TensorRT
    classes_to_labels= utils.get_coco_object_dictionary()
    utils.plot_results(best_results_per_input_trt, inputs, classes_to_labels)
    
