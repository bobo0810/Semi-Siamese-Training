"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import sys
import shutil
import argparse
import logging as logger

import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset_SST
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def moving_average(probe, gallery, alpha):
    """Update the gallery-set network in the momentum way.(MoCo)
    用probe参数更新gallery网络
    """
    for param_probe, param_gallery in zip(probe.parameters(), gallery.parameters()):
        param_gallery.data =  \
            alpha* param_gallery.data + (1 - alpha) * param_probe.detach().data

def train_BN(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        
def shuffle_BN(batch_size):
    """ShuffleBN for batch, the same as MoCo https://arxiv.org/abs/1911.05722 #######
    打乱BN内部图像顺序，以打乱BN
    """
    # 对于batch，打乱BN  两个网络输入的batch数据相同,顺序不同
    shuffle_ids = torch.randperm(batch_size).long().cuda()# 返回指定大小的随机排列方式
    reshuffle_ids = torch.zeros(batch_size).long().cuda()
    # index_copy_(维度，位置，待替换元素)
    reshuffle_ids.index_copy_(0, shuffle_ids, torch.arange(batch_size).long().cuda())
    return shuffle_ids, reshuffle_ids
    
def train_one_epoch(data_loader, probe_net, gallery_net, prototype, optimizer, 
                    criterion, cur_epoch, conf, loss_meter):
    """Tain one epoch by semi-siamese training. 
    """
    for batch_idx, (images1, images2, id_indexes) in enumerate(data_loader):
        batch_size = images1.size(0)
        # 当前总共迭代的batch总数
        global_batch_idx = cur_epoch * len(data_loader) + batch_idx
        images1 = images1.cuda()
        images2 = images2.cuda()

        # 打乱BN内部图像顺序，以打乱BN
        # set inputs as probe or gallery 
        shuffle_ids, reshuffle_ids = shuffle_BN(batch_size)

        # images1输入探测网络   images2打乱batch内部图像顺序后输入底库网络
        images1_probe = probe_net(images1)
        with torch.no_grad():
            images2 = images2[shuffle_ids]
            images2_gallery = gallery_net(images2)[reshuffle_ids]
            images2 = images2[reshuffle_ids]


        # images1和images2对调，在重复一次
        shuffle_ids, reshuffle_ids = shuffle_BN(batch_size)
        images2_probe = probe_net(images2)
        with torch.no_grad():
            images1 = images1[shuffle_ids]
            images1_gallery = gallery_net(images1)[reshuffle_ids]
            images1 = images1[reshuffle_ids]

        # 输入SST原型，得到 output1, output2加间隔并扩展到超球面得分  label队列内部所属的类别标签   id_set当前队列内部包含的图像类别
        output1, output2, label, id_set  = prototype(
            images1_probe, images2_gallery, images2_probe, images1_gallery, id_indexes)
        # 计算交叉熵损失
        loss = (criterion(output1, label) + criterion(output2, label))/2
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 滑动平均更新 底库网络参数
        moving_average(probe_net, gallery_net, conf.alpha)
        loss_meter.update(loss.item(), batch_size)
        if batch_idx % conf.print_freq == 0:
            loss_val = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d, lr %f, loss %f'  % 
                        (cur_epoch, batch_idx, lr, loss_val))
            conf.writer.add_scalar('Train_loss', loss_val, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
    if cur_epoch % conf.save_freq == 0 or cur_epoch == conf.epoches - 1:
        saved_name = ('Epoch_{}.pt'.format(cur_epoch))
        torch.save(probe_net.state_dict(), os.path.join(conf.out_dir, saved_name))
        logger.info('save checkpoint %s to disk...' % saved_name)
    return id_set

def train(conf):
    """Total training procedure. 
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    conf.device = torch.device('cuda:0')
    criterion = torch.nn.CrossEntropyLoss().cuda(conf.device) # 交叉熵损失
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file) # 定义主干网络
    probe_net = backbone_factory.get_backbone() # 初始化探测网络
    gallery_net = backbone_factory.get_backbone() # 初始化底库网络
    head_factory = HeadFactory(conf.head_type, conf.head_conf_file) # 定义SST原型
    prototype = head_factory.get_head().cuda(conf.device) # 初始化SST原型

    # 主干网络转为GPU
    probe_net = torch.nn.DataParallel(probe_net).cuda()
    gallery_net = torch.nn.DataParallel(gallery_net).cuda()

    # 仅优化探测网络参数
    optimizer = optim.SGD(probe_net.parameters(), lr=conf.lr, momentum=conf.momentum, weight_decay=5e-4)
    # 学习率调度器
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.milestones, gamma=0.1)

    # 探测网络 加载预训练权重
    if conf.resume:
        probe_net.load_state_dict(torch.load(args.pretrain_model))

    # 将探测网络参数 赋值给 底库网络
    moving_average(probe_net, gallery_net, 0)

    # 探测网络 设置为训练模式
    probe_net.train()
    # 底库网络 设置为评估模式，但BN层为训练模式
    gallery_net.eval().apply(train_BN)    

    # 该轮次训练时需排除的类别
    exclude_id_set = set()
    loss_meter = AverageMeter()
    for epoch in range(conf.epoches):
        # 每轮次 均初始化训练集加载器，以排除 上一轮次最后一次更新时队列内部类别
        data_loader = DataLoader(
            ImageDataset_SST(conf.data_root, conf.train_file, exclude_id_set), 
            conf.batch_size, True, num_workers = 4, drop_last = True)
        # 一轮训练完成，得到该最后一次更新时队列内部类别
        exclude_id_set = train_one_epoch(data_loader, probe_net, gallery_net, 
            prototype, optimizer, criterion, epoch, conf, loss_meter)
        lr_schedule.step()

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='semi-siamese_training for face recognition.')
    conf.add_argument("--data_root", type = str,
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str,   default='pass',
                      help = "The train file path.")
    conf.add_argument('--backbone_type', type=str, default='MobileFaceNet',
                      help='Mobilefacenets, Resnet.')
    conf.add_argument('--backbone_conf_file', type=str, default='../backbone_conf.yaml',
                      help='the path of backbone_conf.yaml.')
    conf.add_argument("--head_type", type = str, default='SST_Prototype',
                      help = "mv-softmax, arcface, npc-face ...")
    conf.add_argument("--head_conf_file", type = str, default='../classifier_conf.yaml',
                      help = "the path of classifier_conf.yaml..")
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type=str, default='out_dir', 
                      help=" The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 250,
                      help = 'The training epoches.') 
    conf.add_argument('--step', type = str, default = '150,200,230',
                      help = 'Step for lr.')
    conf.add_argument('--print_freq', type = int, default = 100,
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type=int, default=10,
                      help='The save frequency for training state.')
    conf.add_argument('--batch_size', type=int, default=128,
                      help='batch size over all gpus.')
    conf.add_argument('--momentum', type=float, default=0.9, 
                      help='The momentum for sgd.')
    conf.add_argument('--alpha', type=float, default=0.999, 
                      help='weight of moving_average')
    conf.add_argument('--log_dir', type = str, default = 'log', 
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str,default='sst_mobileface',
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str,
                      help = 'The path of pretrained model')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False, 
                      help = 'Resume from checkpoint or not.')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
