import torch
from torch import nn
from dataset import Cost2100DataLoader
import argparse
from models import dcrnet
import os
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import _LRScheduler
import time
import thop
import sys
from colorama import Fore
import logging
import random
import numpy as np
# arg
parser = argparse.ArgumentParser(description='PyTorch CSI feeback')

parser.add_argument('--data', default='./COST2100', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--scenario', type=str, default="in", choices=["in", "out"],
                    help="the channel scenario")
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')
parser.add_argument('--outputs', default='./outputs',
                    help='folder to output model checkpoints')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=200, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--expansion', default=1, type=int,
                   help='expansion rate of dcrnet')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--val-freq', '-v', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()
#random seed
#seed=0
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#random.seed(seed)
#np.random.seed(seed)





def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger
class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]

def load_model(args):
    # load model
    model = dcrnet(reduction=args.cr, expansion=args.expansion)
    image = torch.randn([1, 2, 32, 32])
    flops, params = thop.profile(model, inputs=(image,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    return model,flops, params

def main():


    try:
        os.makedirs(args.outputs+"/log")
        os.makedirs(args.outputs + "/checkpoints")

    except OSError:
        pass
    t=time.strftime("%m%d%H%M", time.localtime())
    logger = get_logger(args.outputs + f"/log/{t}-{args.expansion}X-{args.cr}-{args.scenario}.log")
    # init device
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device="cuda"
        pin_memory=True
        torch.backends.cudnn.benchmark = True
        print(f"Use gpu {args.gpu}")
    else:
        device="cpu"
        pin_memory = False

    model,flops,params=load_model(args)
    logger.info(f'Model Name: CRNet [pretrained: {args.pretrained}]')
    logger.info(f'Model Config: compression ratio=1/{args.cr}')
    logger.info(f'Model Flops: {flops}')
    logger.info(f'Model Params Num: {params}\n')
    model.to(device)
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device(device))['state_dict']
        print("pretrained model loaded from {}".format(args.pretrained))
        model.load_state_dict(state_dict,strict=False)

    # Create the data loader
    train_loader, val_loader, test_loader = Cost2100DataLoader(
        root="./COST2100",
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario)()
    criterion=nn.MSELoss()
    parameters = model.parameters()
    # optimizer = torch.optim.AdamW(parameters, lr=args.lr,
    #                         betas=(0.9, 0.999),
    #                         weight_decay=0.01)
    optimizer=torch.optim.Adam(parameters,lr=args.lr)
    num_steps = len(train_loader) * args.epochs
    scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                             T_max=num_steps,
                                             T_warmup=30 * len(train_loader),
                                             eta_min=5e-5)
    #warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=10)
    #scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2500,eta_min=5e-5)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu:
                checkpoint = torch.load(args.resume)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_nmse = checkpoint['best_nmse']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("no checkpoint found at '{}'".format(args.resume))
    else:
        best_nmse = 10
        best_epoch = -1

    if args.evaluate:
        print("=> evaluating...")
        test(test_loader, device,model,criterion)
        return
    for epoch in range(args.start_epoch,args.epochs):

        training_loss=train(train_loader,device, model, criterion, optimizer,scheduler,epoch)
        logger.info(f"Epoch: {epoch + 1} traning loss={training_loss:3e}")
        if (epoch+1) % args.val_freq==0:
            print("=> val:")
            nmse=test(test_loader,device,model,criterion,epoch)
            if nmse<=best_nmse:
                best_nmse=nmse
                best_epoch=epoch
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_nmse': best_nmse,
                    'best_epoch': best_epoch
                }
                logger.info(f"Test Epoch: [{epoch+1}/{args.epochs}] nmese={nmse} (dB), [best nmse={best_nmse} best epoch {best_epoch+1}]")
                torch.save(state, os.path.join(args.outputs+'/checkpoints',f"{args.expansion}X-cr{args.cr}-{args.scenario}" ))
            print(f"=> best nmse:{best_nmse:3e} (db), epoch:{best_epoch+1}\n\n")






def train(train_loader,device, model, criterion, optimizer, scheduler,epoch):
    iter_loss = AverageMeter('Iter loss')
    iter_time = AverageMeter('Iter time')
    time_tmp = time.time()
    model.train()
    t = tqdm(train_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET),ncols=150)
    for batch_idx, (sparse_gt,) in enumerate(t):
        sparse_gt = sparse_gt.to(device)

        optimizer.zero_grad()
        sparse_pred = model(sparse_gt)
        loss = criterion(sparse_pred, sparse_gt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Log and visdom update
        iter_loss.update(loss)
        iter_time.update(time.time() - time_tmp)
        time_tmp = time.time()

        t.set_description(f"Epoch:[{epoch+1}/{args.epochs}]")
        t.set_postfix({"lr":f"{scheduler.get_last_lr()[0]:.2e}",
            "MSE loss":f"{iter_loss.avg:.3e}",
        })
        # if (batch_idx + 1) % args.print_freq == 0:
        #     print(f'Epoch: [{epoch}/{args.epochs}]'
        #                 f'[{batch_idx + 1}/{len(train_loader)}] '
        #                 f'lr: {optimizer.param_groups[0]["lr"]:.2e} | '
        #                 f'MSE loss: {iter_loss.avg:.3e} | '
        #                 f'time: {iter_time.avg:.3f}')

    return iter_loss.avg

def test(data_loader,device,model,criterion,epoch=1):
    iter_rho = AverageMeter('Iter rho')
    iter_nmse = AverageMeter('Iter nmse')
    iter_loss = AverageMeter('Iter loss')
    t = tqdm(data_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET), ncols=150)
    model.eval()
    with torch.no_grad():
        for batch_idx, (sparse_gt, raw_gt) in enumerate(t):
            sparse_gt = sparse_gt.to(device)
            sparse_pred = model(sparse_gt)
            loss = criterion(sparse_pred, sparse_gt)
            rho, nmse = evaluator(sparse_pred, sparse_gt, raw_gt)
            # Log and visdom update
            iter_loss.update(loss)
            iter_rho.update(rho)
            iter_nmse.update(nmse)
            t.set_description(f"Testing: Epoch:[{epoch + 1}]")
            t.set_postfix({"MSE loss": f"{iter_loss.avg:.3e}",
                           "NMSE(db)": f"{iter_nmse.avg:.3e}",
                           "rho":f"{iter_rho.avg:.3e}"
                           })
    return iter_nmse.avg
class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"

def evaluator(sparse_pred, sparse_gt, raw_gt):
    r""" Evaluation of decoding implemented in PyTorch Tensor
         Computes normalized mean square error (NMSE) and rho.
    """

    with torch.no_grad():
        # Basic params
        nt = 32
        nc = 32
        nc_expand = 257

        # De-centralize
        sparse_gt = sparse_gt - 0.5
        sparse_pred = sparse_pred - 0.5

        # Calculate the NMSE
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())

        # Calculate the Rho
        n = sparse_pred.size(0)
        sparse_pred = sparse_pred.permute(0, 2, 3, 1)  # Move the real/imaginary dim to the last
        zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))
        sparse_pred = torch.cat((sparse_pred, zeros), dim=2)
        raw_pred = torch.fft(sparse_pred,signal_ndim=1)[:, :, :125, :]

        norm_pred = raw_pred[..., 0] ** 2 + raw_pred[..., 1] ** 2
        norm_pred = torch.sqrt(norm_pred.sum(dim=1))

        norm_gt = raw_gt[..., 0] ** 2 + raw_gt[..., 1] ** 2
        norm_gt = torch.sqrt(norm_gt.sum(dim=1))

        real_cross = raw_pred[..., 0] * raw_gt[..., 0] + raw_pred[..., 1] * raw_gt[..., 1]
        real_cross = real_cross.sum(dim=1)
        imag_cross = raw_pred[..., 0] * raw_gt[..., 1] - raw_pred[..., 1] * raw_gt[..., 0]
        imag_cross = imag_cross.sum(dim=1)
        norm_cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)

        rho = (norm_cross / (norm_pred * norm_gt)).mean()

        return rho, nmse

if __name__ == '__main__':
    main()