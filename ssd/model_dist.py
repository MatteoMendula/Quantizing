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
    def __init__(self, precision="fp32"):
        super().__init__()
        # self.original_ssd = original_ssd
        self.precision = precision
        # model_iterable = model.children()

        self.input_layer = nn.Conv2d(3, 12, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder = nn.Sequential(
            self.input_layer,
            self.bn1,
            self.relu,
            self.max,
            self.bn1,
            self.relu
        )
       
    def forward(self, x):
        # x = self.input_layer(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.max(x)
        # x = self.bn1(x)
        head_output = self.encoder(x)
        head_outpu_q, scale, zero_point = quantization_utils.quantize_tensor_torch(x=head_output)
        # head_output = self.head_model(x)
        # head_outpu_q, scale = tensor_quant.tensor_quant(head_output, head_output.abs().max())
        # head_outpu_q, scale, zero_point = quantization_utils.quantize_tensor_torch(head_output)
        return (head_outpu_q, scale, zero_point)
    
    def parse_to_onnx(self):
        input = [torch.randn((1,3,300,300)).to("cuda")]
        if self.precision == 'fp16':
            input = [torch.randn((1,3,300,300)).to("cuda").half()]
        model = self.eval().to("cuda")
        traced_model = torch.jit.trace(model, input)    
        torch.onnx.export(traced_model,  # model being run
                            input,  # model input (or a tuple for multiple inputs)
                            "./models/head_{}.onnx".format(self.precision),  # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=13,  # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names=['input'],  # the model's input names
                            output_names=['output0', 'output1', 'output2'],  # the model's output names]
                        )
        
    def parse_to_trt(self):
        import subprocess
        print("self.precision", self.precision)
        cmd = '/home/matteo/TensorRT-8.6.1.6/bin/trtexec --onnx=./models/head_{}.onnx --saveEngine=./models/head_{}.trt'.format(self.precision, self.precision)
        # if self.precision == 'fp16':
        #     cmd = '/home/matteo/TensorRT-8.6.1.6/bin/trtexec --onnx=./models/head_{}.onnx --saveEngine=./models/head_{}.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16'.format(self.precision, self.precision)
        output = subprocess.check_call(cmd.split(' '))
        print(output)

class SSD300Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=2, bias=False)
        self.bn64 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 512, kernel_size=2, stride=1, padding=1, bias=False)
        self.bn512 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)

        self.decoder = nn.Sequential(
            self.conv1,
            self.bn64,
            self.relu,
            self.conv2,
            self.bn512,
            self.relu,
            self.conv3,
            self.bn512,
            self.relu,
            self.conv4,
            self.bn512,
            self.relu,
            self.conv4,
            self.avgpool
        )

        self.decode_out = None

    def forward(self, x):
        x = quantization_utils.dequantize_tensor_pytorch(quantized_tensor=x[0], scale=x[1], zero_point=x[2]) 
        x = self.decoder(x)
        self.decode_out = x
        return x


class SSD300Tail(nn.Module):
    def __init__(self, original_ssd):
        super().__init__()
        model_iterable = original_ssd.children()
        inside_resnet = next(model_iterable).children()
        sequential = next(inside_resnet).children()
        tail_list = []
        for index, module in enumerate(sequential):
            if index > 5:
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
    teacher_head = model.feature_extractor.feature_extractor[:6]
    teacher_decoder = model.feature_extractor.feature_extractor[6]
    head = SSD300Head(precision)
    decoder = SSD300Decoder()
    tail = SSD300Tail(model)

    teacher_head = teacher_head.eval().cuda()
    teacher_decoder = teacher_decoder.eval().cuda()
    head = head.cuda()
    decoder = decoder.cuda()
    tail = tail.cuda()

    teacher_head = teacher_head.requires_grad_(False)
    teacher_decoder = teacher_decoder.requires_grad_(False)
    head = head.requires_grad_(True)
    decoder = decoder.requires_grad_(True)
    tail = tail.requires_grad_(False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(
        [
            {'params': head.parameters()},
            {'params': decoder.parameters()}
        ],
        lr=0.001,
        momentum=0.9
    )
    # optimizer = torch.optim.SGD(list(head.parameters())  + list(decoder.parameters()), lr=0.001, momentum=0.9)

    path2data="../coco/train2017"
    path2json="../coco/annotations/instances_train2017.json"

    transforms_thing = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])

    coco_train = datasets.CocoDetection(root = path2data,
                                annFile = path2json,
                                transform = transforms_thing)

    def collate_fn(batch):
        images = []
        targets = []
        for image, target in batch:
            images.append(image)
            targets.append(target)
        return torch.stack(images, 0), targets

    train_loader = torch.utils.data.DataLoader(coco_train, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # inputs = [utils.prepare_input("./kitchen.jpg")]
    # inputs = utils.prepare_tensor(inputs, should_half=False)
    # print("it didnt work: ", inputs)

    epochs = 1
    interms = [0, 0, 0, 0, 0, 0]
    tail_outs = [0,0,0,0,0]
    loss = torch.zeros(1,1)

    for epoch in range(epochs):
        step = 0
        for inputs, targets in tqdm(train_loader):
            loss_first_part = 0
            loss_second_part = 0
            inputs = inputs.cuda()
            
            teacher_head = teacher_head.eval().cuda()
            interms[0] = teacher_head(inputs)
            # interms[0] = interms[0][:, :, :interms[0].shape[2]-1, :].cuda()
            interms[1] = teacher_decoder[0](interms[0])
            interms[2] = teacher_decoder[1](interms[1])
            interms[3] = teacher_decoder[2](interms[2])
            interms[4] = teacher_decoder[3](interms[3])
            interms[5] = teacher_decoder[4](interms[4])

            head = head.eval().cuda()
            decoder = decoder.eval().cuda()
            head_out = head(inputs)
            decoded_out = decoder(head_out)
            
            head = head.train().cuda()
            decoder = decoder.train().cuda()

            tail = tail.eval().cuda()

            tail_outs[0] = tail.tail_model[0][0](decoded_out)
            tail_outs[1] = tail.tail_model[0][1](tail_outs[0])
            tail_outs[2] = tail.tail_model[0][2](tail_outs[1])
            tail_outs[3] = tail.tail_model[0][3](tail_outs[2])
            tail_outs[4] = tail.tail_model[0][4](tail_outs[3])

            loss_first_part = loss_fn(decoded_out, interms[0])

            for i in range(5):
                if i == 0:
                    loss_second_part = loss_fn(interms[i+1], tail_outs[i])
                else:
                    loss_second_part += loss_fn(interms[i+1], tail_outs[i])

            loss = loss_first_part + loss_second_part
            print("loss: ", loss.item(), loss)
            optimizer.zero_grad()
            loss.backward()
            # torch.cuda.synchronize(device='cuda:0')
            optimizer.step()

            if step % 20 == 0:
                print(interms[0][0]) 
                print(decoded_out[0])
                print('step: ', step, " -- loss: ", loss.item())
            if step % 200 == 0:
                print("saving model")
                torch.save(head.state_dict(), "./models/dist_checks/head.pth")
                torch.save(decoder.state_dict(), "./models/dist_checks/decoder.pth")
                torch.save(tail.state_dict(), "./models/dist_checks/tail.pth")
            step += 1
