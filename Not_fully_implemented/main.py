from networks import MobileFaceNet
from tensorboardX import SummaryWriter
from prototype import Prototype
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch import optim


import os
import argparse
import numpy as np
import torch
import random
import logging as logger
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')



def train_BN(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def moving_average(probe, gallery, alpha):
    '''
    用probe参数更新gallery网络
    :param probe:
    :param gallery:
    :param alpha:
    :return:
    '''
    for param_probe, param_gallery in zip(probe.parameters(), gallery.parameters()):
        param_gallery.data =  alpha* param_gallery.data + (1 - alpha) * param_probe.detach().data

        
####### shuffleBN for batch, the same as MoCo https://arxiv.org/abs/1911.05722 #######
def shuffle_BN(batch_size):
    '''
    对于batch，打乱BN  两个网络输入的batch数据相同,顺序不同
    :param batch_size:
    :return:
    '''
    shuffle_ids = torch.randperm(batch_size).long().cuda() # 返回指定大小的随机排列方式
    reshuffle_ids = torch.zeros(batch_size).long().cuda()
    # index_copy_(维度，位置，待替换元素)
    reshuffle_ids.index_copy_(0, shuffle_ids, torch.arange(batch_size).long().cuda())
    return shuffle_ids, reshuffle_ids
  
  
def trainlist_to_dict(source_file):
    '''
    读取数据集
    :param source_file:
    :return:
    '''
    trainfile_dict = {}
    with open(source_file, 'r') as infile:
        for line in infile:
            l = line.rstrip().lstrip()
            if len(l) > 0:
                lmdb_key, label = l.split(' ')
                label = int(label)
                if label not in trainfile_dict:
                    trainfile_dict[label] = {'lmdb_key':[],'num_images':0}
                trainfile_dict[label]['lmdb_key'].append(lmdb_key)
                trainfile_dict[label]['num_images'] += 1
    return trainfile_dict


def train_sample(train_dict, class_num, queue_size, last_id_list=False):
    '''
    数据集采样：每个ID随机采样两张  组成浅层人脸学习
    :param train_dict:
    :param class_num:
    :param queue_size:
    :param last_id_list:
    :return:
    '''
    all_id = range(0, class_num)
    # Make sure there is no overlap ids bewteen queue and curr batch.
    # 确保 队列与当前batch没有重合的ids    队列视为mini版的分类器权重
    if last_id_list:
        last_tail_id= last_id_list[-queue_size:] # 尾部队列内的id
        non_overlap_id = list(set(all_id) - set(last_tail_id)) # 不重合的id= 全部id - 尾部队列id
        assert len(non_overlap_id) >= queue_size
        curr_head_id = random.sample(non_overlap_id, queue_size) # 从不重合的id内随机采样 队列长度范围的id,作为头部id
        curr_remain_id = list(set(all_id) - set(curr_head_id)) # 剩余的id
        random.shuffle(curr_remain_id) # 打乱剩余id
        curr_head_id.extend(curr_remain_id) # 全部id= 头部id+剩余id
        curr_id_list = curr_head_id
    # 第一轮训练  此时队列为空
    else:
        random.shuffle(all_id)
        curr_id_list = all_id
    
    # For each ID, two images are randomly sampled
    # 对于每个类别，随机采样两张照片
    curr_train_list =[]
    for index in curr_id_list:
        lmdb_key_list =  train_dict[index]['lmdb_key']
        # 深层人脸数据 （每轮均随机采样，导致 网络训练用到所有图像）
        if int(train_dict[index]['num_images']) >= 2:
            training_samples = random.sample(lmdb_key_list, 2)
            line = training_samples[0] + ' ' + training_samples[1]
        # 浅层人脸数据
        else:
            line = lmdb_key_list[0] + ' '+ lmdb_key_list[0]
        curr_train_list.append(line+ ' '+ str(index) +'\n')
    # curr_train_list ['img_path  label']    curr_id_list 0~类别总数的标签顺序打乱
    return curr_train_list,curr_id_list


def train_one_epoch(data_loader, probe_net, gallery_net, prototype, optimizer, 
    criterion, cur_epoch, conf):
    db_size = len(data_loader) # 数据集一轮batch_size总数
    check_point_size = (db_size // 2)
    batch_idx = 0
    initial_lr = get_lr(optimizer)

    probe_net.train() # 转为训练模式    参数正常更新
    gallery_net.eval().apply(train_BN) # 转评估模式（固定BN和Dropout）,参数参考probe_net滑动均值更新   train_BN:BN开启训练

    for batch_idx, (images, _ ) in enumerate(data_loader):
        # images[batch_size,6,W,H]  6:代表 生活照和证件照两张图像
        batch_size = images.size(0)
        global_batch_idx = (cur_epoch - 1) * db_size + batch_idx  #当前训练的总batch_size

        # the label of current batch in prototype queue
        # 队列原型中对应的当前batch的真值标签 [1，batch_size]    队列内部不断变化，所以对应类别也在不断变化
        label = (torch.LongTensor([range(batch_size)]) + global_batch_idx * batch_size) % conf.queue_size # 取余是 返回队列开头
        label = label.squeeze().cuda() # [batch_size]
        images = images.cuda() # [batch,6,H,W]
        x1, x2 = torch.split(images, [3, 3], dim=1)  # 分成两张 生活照和证件照[batch,3,H,W]

        # set inputs as probe or gallery
        # 打乱一个batch内部图像的排列顺序
        shuffle_ids, reshuffle_ids = shuffle_BN(batch_size)


        x1_probe = probe_net(x1)
        with torch.no_grad():
            x2 = x2[shuffle_ids] # 打乱batch内部图像顺序
            x2_gallery = gallery_net(x2)[reshuffle_ids] # 提取特征后再恢复batch内部顺序。  gallery_net内部学习BN
            x2 = x2[reshuffle_ids] #x2 顺序恢复

        # 再次随机
        shuffle_ids, reshuffle_ids = shuffle_BN(batch_size)
        x2_probe = probe_net(x2)
        with torch.no_grad():
            x1 = x1[shuffle_ids]
            x1_gallery = gallery_net(x1)[reshuffle_ids]
            x1 = x1[reshuffle_ids]

        # 计算得分 并扩展到半径为s的超球面上
        output1, output2  = prototype(x1_probe,x2_gallery,x2_probe,x1_gallery,label)
        # 计算损失
        loss = criterion(output1, label) + criterion(output2, label)
        
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # gallery_net参考probe_net滑动均值更新
        moving_average(probe_net, gallery_net, conf.alpha)

        # 可视化
        if batch_idx % conf.print_freq == 0:
            loss_val = loss.item()
            lr = get_lr(optimizer)
            logger.info('epoch %d, iter %d, lr %f, loss %f'  % (cur_epoch, batch_idx, lr, loss_val))
            conf.writer.add_scalar('Train_loss', loss_val, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
    # 保存
    if cur_epoch % conf.save_freq == 0 :
        saved_name = ('{}_epoch_{}.pt'.format(conf.model_type,cur_epoch))
        torch.save(probe_net.state_dict(), os.path.join(conf.saved_dir, saved_name))
        logger.info('save checkpoint %s to disk...' % saved_name)

def train_sst(conf):
    # 孪生的两个网络
    probe_net = MobileFaceNet(conf.feat_dim)
    gallery_net = MobileFaceNet(conf.feat_dim) 

    # 滑动均值   两者初始化相同参数
    moving_average(probe_net, gallery_net, 0)

    # 队列原型
    prototype = Prototype(conf.feat_dim, conf.queue_size, conf.scale,conf.margin, conf.loss_type).cuda()     
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # 优化器
    optimizer = optim.SGD(probe_net.parameters(), lr=conf.lr, momentum=conf.momentum, weight_decay=5e-4)
    # 学习率更新
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.lr_decay_epochs, gamma=0.1)

    # 多GPU  进行数据并行
    probe_net = torch.nn.DataParallel(probe_net).cuda()
    gallery_net = torch.nn.DataParallel(gallery_net).cuda()
    # 读取数据集
    train_dict = trainlist_to_dict(conf.source_file)

    # 开始训练
    for epoch in range(1, conf.epochs + 1):
        # 数据集随机采样每个ID两张  组成浅层人脸学习
        # curr_train_list ['img_path  label']    curr_id_list 0~N标签顺序打乱
        if epoch == 1:
            curr_train_list, curr_id_list = train_sample(train_dict, conf.class_num, conf.queue_size)
        else:
            curr_train_list, curr_id_list = train_sample(train_dict, conf.class_num, conf.queue_size, curr_id_list)

        # 数据集加载器
        data_loader = DataLoader(Dataset(conf.source_lmdb, curr_train_list, conf.key),
                                 conf.batch_size, shuffle = False, num_workers=4, drop_last = True)
        # 开始训练
        train_one_epoch(data_loader, probe_net, gallery_net, prototype, optimizer,
            criterion, epoch, conf)
        # 更新学习率
        lr_schedule.step()



if __name__ == '__main__':
    '''
    生活照对应探测集probe 
    证件照对应底库gallery
    '''
    conf = argparse.ArgumentParser(description='train arcface on face database.')
    conf.add_argument('--key', type=int, default=None, help='you must give a key before training.')
    conf.add_argument("--train_db_dir", type=str, default='/export/home/data', help="input database name")
    conf.add_argument("--train_db_name", type=str, default='deepglint_unoverlap_part40', help="comma separated list of training database.")
    conf.add_argument("--train_file_dir", type=str, default='/export/home/data/deepglint_unoverlap_part40', help="input train file dir.")
    conf.add_argument("--train_file_name", type=str, default='deepglint_train_list.txt', help="input train file name.")
    conf.add_argument("--output_model_dir", type=str, default='./snapshot', help=" save model paths")
    conf.add_argument('--model_type',type=str, default='mobilefacenet',choices=['mobilefacenet'], help='choose model_type')  # 网络
    conf.add_argument('--feat_dim', type=int, default=512, help='feature dimension.') # 特征维度
    conf.add_argument('--queue_size', type=int, default=16384, help='size of prototype queue') # 队列长度 论文推荐值，适用深层和浅层人脸
    conf.add_argument('--class_num', type=int, default=72778, help='number of categories') # 类别总数
    conf.add_argument('--loss_type', type=str, default='softmax',choices=['softmax','am_softmax','arc_softmax'], help="loss type, can be softmax, am or arc") # 损失函数
    conf.add_argument('--margin', type=float, default=0.0, help='loss margin ') # 损失间隔
    conf.add_argument('--scale', type=float, default=30.0, help='scaling parameter ') # 损失的缩放尺度
    conf.add_argument('--lr', type=float, default=0.05, help='initial learning rate.')
    conf.add_argument('--epochs', type=int, default=100, help='training epochs')
    conf.add_argument('--lr_decay_epochs', type=str, default='48,72,90', help='training epochs') # 第X个epoch后更新学习率
    conf.add_argument('--momentum', type=float, default=0.9, help='momentum')
    conf.add_argument('--alpha', type=float, default=0.999, help='weight of moving_average') # 滑动均值权重
    conf.add_argument('--batch_size', type=int, default=128, help='batch size over all gpus.')
    conf.add_argument('--print_freq', type=int, default=100, help='frequency of displaying current training state.') # 打印
    conf.add_argument('--save_freq', type=int, default=1, help='frequency of saving current training state.')
    args = conf.parse_args()
    args.lr_decay_epochs = [int(p) for p in args.lr_decay_epochs.split(',')]
    args.source_file = os.path.join(args.train_file_dir, args.train_file_name) # 数据集
    args.source_lmdb = os.path.join(args.train_db_dir, args.train_db_name)

    subdir =datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S')
    loss_type=args.loss_type
    args.saved_dir = os.path.join(args.output_model_dir,loss_type,subdir)
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    writer = SummaryWriter(log_dir=args.saved_dir)
    args.writer = writer
    logger.info('Start optimization.')
    logger.info(args)
    train_sst(args)
    logger.info('Optimization done!')
