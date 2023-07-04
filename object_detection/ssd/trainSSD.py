import numpy as np
import torch
from inference_utils import *
from evaluation_utils import *
from ssd.utils.train_utils import *
import argparse
from SSD_creator import SSD_creator, Loss
from logger_utils import Logger
import dllogger as DLLogger
from torch.optim.lr_scheduler import MultiStepLR

def train(ssd300, train_loop_func, logger, args):
    use_cuda = True
    N_gpu = 1
    seed = np.random.randint(1e4)

    print("Using seed = {}".format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed=seed)


    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    train_loader = get_train_loader(args, seed - 2**31)

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    learning_rate = args["learning_rate"] * N_gpu * (args["batch_size"] / 32)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)

    if use_cuda:
        ssd300.cuda()
        loss_func.cuda()

    optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=learning_rate,
                                momentum=args["momentum"], weight_decay=args["weight_decay"])
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300.module if args.distributed else ssd300, args.checkpoint)
            checkpoint = torch.load(args.checkpoint,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    inv_map = {v: k for k, v in val_dataset.label_map.items()}

    total_time = 0

    if args.mode == 'evaluation':
        acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
        if args.local_rank == 0:
            print('Model precision {} mAP'.format(acc))
        return

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    mean, std = generate_mean_std(args)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        iteration = train_loop_func(ssd300, loss_func, scaler,
                                    epoch, optimizer, train_loader, val_dataloader, encoder, iteration,
                                    logger, args, mean, std)
        if args.mode in ["training", "benchmark-training"]:
            scheduler.step()
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            logger.update_epoch_time(epoch, end_epoch_time)

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)

            if args.local_rank == 0:
                logger.update_epoch(epoch, acc)

        if args.save and args.local_rank == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'label_map': val_dataset.label_info}
            if args.distributed:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            os.makedirs(args.save, exist_ok=True)
            save_path = os.path.join(args.save, f'epoch_{epoch}.pt')
            torch.save(obj, save_path)
            logger.log('model path', save_path)
        train_loader.reset()
    DLLogger.log((), { 'total time': total_time })
    logger.log_summary()

def get_parameters():
    parser = argparse.ArgumentParser(description='Parser for SSD300')
    parser.add_argument('-p', '--precision', default="fp32", type=str)
    parser.add_argument('-d', '--data', default="../coco", type=str)
    parser.add_argument('-a', '--amp_autocast', default=False, type=bool)
    parser.add_argument('-e', '--eval_batch_size', default=32, type=int)
    parser.add_argument('-n', '--num_workers', default=8, type=int)
    parser.add_argument('-w', '--pretrained_weigths', default="IMAGENET1K_V2", type=str)
    parser.add_argument('-s', '--save', type=str, default="./models/original_ssd", help='save model checkpoints in the specified directory')
    parser.add_argument('-l', '--log_interval', type=int, default=20, help='path to model checkpoint file')
    parser.add_argument('-j', '--json_summary', default='training_summary', help='if specified, output json summary')
    parser.add_argument('-r', '--learning_rate', '--lr', type=float, default=2.6e-3, help='learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='momentum argument for SGD optimizer')
    parser.add_argument('-nw', '--num-workers', type=int, default=8)
    parser.add_argument('-wd' ,'--weight_decay', type=float, default=0.0005, help='momentum argument for SGD optimizer')    
    parser.add_argument('-g', '--N_gpu', type=int, default=1, help='number of GPUs to use for training')
    parser.add_argument('-am', '--amp', type=bool, default=False, help='use mixed precision')
    parser.add_argument('-wa', '--warmup', type=bool, default=True, help= 'use warmup scheduler')
    parser.add_argument('--data_layout', default="channels_last", choices=['channels_first', 'channels_last'], help="Model data layout. It's recommended to use channels_first with --no-amp")
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='number of examples for each iteration')
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = get_parameters()
    train_loop_func = train_loop
    logger = Logger('Training logger', log_interval=args['log_interval'], json_output=args['json_summary'])
    ssd_creator = SSD_creator(args['precision'], args['pretrained_weigths'])
    train(ssd_creator.ssd300, train_loop_func, logger, args)