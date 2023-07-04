import os
from coco_pipeline import COCOPipeline, DALICOCOIterator
from inference_utils import *
from torch.autograd import Variable

def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

def get_train_loader(args, local_seed):
    train_annotate = os.path.join(args["data"], "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args["data"], "train2017")
    train_pipe = COCOPipeline(batch_size=args["batch_size"],
        file_root=train_coco_root,
        annotations_file=train_annotate,
        default_boxes=dboxes300_coco(),
        device_id= os.getenv('LOCAL_RANK',0),
        num_shards= args["N_gpu"],
        output_fp16= args["amp"],
        output_nhwc=False,
        pad_output=False,
        num_threads=args["num_workers"], seed=local_seed)
    train_pipe.build()
    test_run = train_pipe.schedule_run(), train_pipe.share_outputs(), train_pipe.release_outputs()
    train_loader = DALICOCOIterator(train_pipe, 118287 / args["N_gpu"])
    return train_loader

def train_loop(model, loss_func, scaler, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    for nbatch, data in enumerate(train_dataloader):
        img = data[0][0][0]
        bbox = data[0][1][0]
        label = data[0][2][0]
        label = label.type(torch.cuda.LongTensor)
        bbox_offsets = data[0][3][0]
        bbox_offsets = bbox_offsets.cuda()
        img.sub_(mean).div_(std)
        img = img.cuda()
        bbox = bbox.cuda()
        label = label.cuda()
        bbox_offsets = bbox_offsets.cuda()

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue

        # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)

        with torch.cuda.amp.autocast(enabled=args.amp):
            if args["data_layout"] == 'channels_last':
                img = img.to(memory_format=torch.channels_last)
            ploc, plabel = model(img)

            ploc, plabel = ploc.float(), plabel.float()
            trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
            gloc = Variable(trans_bbox, requires_grad=False)
            glabel = Variable(label, requires_grad=False)

            loss = loss_func(ploc, plabel, gloc, glabel)

        if args["warmup"]:
            warmup(optim, args.warmup, iteration, args["learning_rate"])

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        if args.local_rank == 0:
            logger.update_iter(epoch, iteration, loss.item())
        iteration += 1

    return iteration

def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]

def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)

def generate_mean_std(args):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean_val).cuda()
    std = torch.tensor(std_val).cuda()

    view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    return mean, std