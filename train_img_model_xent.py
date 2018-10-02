from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
from PIL import Image
import pdb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset, ImageDatasetLazy
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from optimizers import init_optim

import enum

class CameraCheck(enum.Enum):
    primary = 1
    skipped = 2
    all = 3

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=20, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    gallery = ImageDatasetLazy(dataset.gallery, transform=transform_test)
    galleryloader = DataLoader(gallery,
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, gallery, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        
        if args.stepsize > 0: scheduler.step()
        
        if (epoch+1) > args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, gallery, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)
        
        outputs = model(imgs)
        if isinstance(outputs, tuple):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform_test(img)
    return img

def check_exit(s_upper_b, exit_time):
    if s_upper_b >= exit_time:
        print("could not find person, giving up!")
        return True
    return False

def handle_retry(f_rate, s_lower_b, s_upper_b, fallback_time, cam_check):
    # revert to historical search
    if cam_check == CameraCheck.primary and s_upper_b >= fallback_time:
        print("now checking OTHER cameras!")
        cam_check = CameraCheck.skipped
        s_lower_b = 0.
        s_upper_b = f_rate * 2.
    # search next frame
    else:
        if cam_check == CameraCheck.skipped and s_upper_b >= fallback_time:
            print("now checking ALL cameras!")
            cam_check = CameraCheck.all
        s_lower_b = s_upper_b
        s_upper_b += (f_rate * 2.0)

    return s_lower_b, s_upper_b, cam_check

def test(model, queryloader, gallery, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()
    
    model.eval()

    f_rate = 60.
    dist_thresh = 160.
    fallback_time = f_rate * 30
    exit_time = f_rate * 67.75

    cam_offsets = [5542, 3606, 27243, 31181, 0, 22401, 18967, 46765]
    corr_matrix = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7]
    ]

    # process query images
    with torch.no_grad():
        qf, q_pids, q_camids, q_fids, q_names = [], [], [], [], []
        for batch_idx, (names, imgs, pids, camids, fids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            # adjust frame ids
            fids += torch.LongTensor([cam_offsets[cid] for cid in camids])

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            q_names.extend(names)
            q_fids.extend(fids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_fids = np.asarray(q_fids)
        q_names = np.asarray(q_names)
        print("query imgs", q_names)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    tot_img_seen = 0
    tot_img_elim = 0
    tot_match_found = 0
    tot_match_pres = 0
    tot_delay = 0.

    tot_t_pos = 0
    tot_f_pos = 0
    tot_t_neg = 0
    tot_f_neg = 0

    # execute queries
    for q_idx, (q_pid, q_camid, q_fid, q_name) in enumerate(zip(q_pids, q_camids, q_fids, q_names)[:100]):

        print("\nnew query person ------------------------------------ ")
        print("query id: ", q_idx, "pid: ", q_pid, "camid: ", q_camid,
            "frameid: ", q_fid, "name: ", q_name)

        # query vars
        q_iter = 0
        s_lower_b = 0.
        s_upper_b = f_rate * 2.
        cam_check = CameraCheck.primary

        # query features
        qf_orig = qf[q_idx].unsqueeze(0)
        qf_i = qf_orig

        # query stats
        q_img_seen = 0
        q_img_elim = 0
        q_match_found = 0
        q_match_pres = 0
        q_delay = 0.

        q_img_seen_arr = []
        q_img_elim_arr = []
        q_delay_arr = []

        t_pos = 0.
        f_pos = 0.
        t_neg = 0.
        f_neg = 0.

        num_inst = 0.

        # count total num. of pos. examples
        for idx in range(0, len(gallery)):
            img_name, pid, camid, fid = gallery[idx]
            fid += cam_offsets[camid]

            if pid == q_pid and fid > q_fid:
                num_inst += 1

        while q_iter >= 0:
            print("\nquery: (", q_idx, ",", q_iter, ")",
                "pid: ", q_pid, "camid: ", q_camid, "frameid: ", q_fid, "name: ", q_name,
                "\twin: [", s_lower_b / f_rate, ",", s_upper_b / f_rate, "]")
            print("search mode: ", cam_check)

            img_elim = 0

            gf, g_pids, g_camids, g_fids, g_names = [], [], [], [], []
            g_a_pids, g_a_camids = [], []

            # load gallery
            for idx in range(0, len(gallery)):
                img_name, pid, camid, fid = gallery[idx]

                # adjust frame id
                fid += cam_offsets[camid]

                if fid > (q_fid + s_lower_b) and fid <= (q_fid + s_upper_b):
                    check_frame = False

                    if cam_check == CameraCheck.all:
                        # baseline: check all
                        check_frame = True
                    elif cam_check == CameraCheck.skipped:
                        # special case: hist. search on skipped cameras
                        if camid not in corr_matrix[q_camid]:
                            check_frame = True
                    elif cam_check == CameraCheck.primary:
                        # pruned search
                        if camid in corr_matrix[q_camid]:
                            check_frame = True
                        else:
                            img_elim += 1

                    if check_frame:
                        g_names.append(img_name)
                        g_pids.append(pid)
                        g_camids.append(camid)
                        g_fids.append(fid)

                    g_a_pids.append(pid)
                    g_a_camids.append(camid)

            # load images
            imgs = []
            for img_name in g_names:
                path = osp.normpath("data/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/" + img_name)
                imgs.append(read_image(path))

            # update delay
            if len(q_delay_arr) <= q_iter:
                q_delay_arr.append(0)

            if cam_check == CameraCheck.skipped:
                q_delay += 2.
                q_delay_arr[q_iter] += 2.

            # handle no candidate case
            if len(imgs) == 0:
                print("no candidates detected, skipping")
                # check for exit
                if check_exit(s_upper_b=s_upper_b, exit_time=exit_time):
                    print("\nframes tracked: ", q_fids[q_idx], "-", q_fid)
                    break
                # handle retry
                s_lower_b, s_upper_b, cam_check = handle_retry(f_rate=f_rate,
                    s_lower_b=s_lower_b, s_upper_b=s_upper_b, fallback_time=fallback_time, cam_check=cam_check)
                continue

            with torch.no_grad():
                imgs = torch.stack(imgs, dim=0)
                if use_gpu: imgs = imgs.cuda()

                # extract features
                end = time.time()
                features = model(imgs)
                batch_time.update(time.time() - end)

                gf.append(features.data.cpu())
                gf = torch.cat(gf, 0)

            g_a_pids = np.asarray(g_a_pids)
            g_a_camids = np.asarray(g_a_camids)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)
            g_names = np.asarray(g_names)
            g_fids = np.asarray(g_fids)

            # gallery pruning stats
            print("eliminated: ", img_elim)
            print("new gallery size: ", len(gf))
            q_img_seen += len(gf)
            q_img_elim += img_elim
            if len(q_img_seen_arr) <= q_iter:
                q_img_seen_arr.append(0)
                q_img_elim_arr.append(0)
            q_img_seen_arr[q_iter] += len(gf)
            q_img_elim_arr[q_iter] += img_elim

            print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
            print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, len(gf)))

            # print("q_fids", q_fids)
            # print("g_fids", g_fids)

            # compute dist matrix
            m, n = qf_i.size(0), gf.size(0)
            distmat = torch.pow(qf_i, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf_i, gf.t())
            distmat = distmat.numpy()

            print("Computing CMC and mAP")
            cmc, AP, valid, f, p = evaluate(distmat, np.expand_dims(q_pid, axis=0), g_pids, np.expand_dims(q_camid, axis=0), g_camids,
                use_metric_cuhk03=args.use_metric_cuhk03, img_names=g_names, g_a_pids=g_a_pids, g_a_camids=g_a_camids)

            if valid == 1:
                all_cmc.append(cmc[0])
                all_AP.append(AP[0])
                num_valid_q += valid
                q_match_found += f
                q_match_pres  += p

            print("mAP (so far): {:.1%}".format(np.mean(all_AP)))
            print("img seen (so far): {}".format(q_img_seen))
            print("img tot. (so far): {}".format(q_img_seen + q_img_elim))
            print("matches found (so far): {}".format(q_match_found))
            print("matches pres. (so far): {}".format(q_match_pres))
            print("delay (so far): {}".format(q_delay))
            print("t_pos {}, f_neg {}".format(t_pos, f_neg))
            print("t_pos {}, f_pos {}".format(t_pos, f_pos))

            # check for match
            indices = np.argsort(distmat, axis=1)
            if distmat[0][indices[0][0]] > dist_thresh:
                print("not close enough, waiting...", distmat[0][indices[0][0]])
                # set accuracy stats
                if q_pids[q_idx] in g_pids[indices][0]:
                    f_neg += 1.
                else:
                    t_neg += 1.

                # check for exit
                if check_exit(s_upper_b=s_upper_b, exit_time=exit_time):
                    print("\nframes tracked: ", q_fids[q_idx], "-", q_fid)
                    break
                # handle retry
                s_lower_b, s_upper_b, cam_check = handle_retry(f_rate=f_rate,
                    s_lower_b=s_lower_b, s_upper_b=s_upper_b, fallback_time=fallback_time, cam_check=cam_check)
                continue

            else:
                print("match declared:", distmat[0][indices[0][0]])
                # set accuracy stats
                if q_pids[q_idx] == g_pids[indices][0][0]:
                    t_pos += 1.
                else:
                    f_pos += 1.

                # reset window, flag
                s_lower_b = 0.
                s_upper_b = f_rate * 2.
                cam_check = CameraCheck.primary

                # find next query img
                q_iter += 1
                q_pid = g_pids[indices][0][0]
                q_camid = g_camids[indices][0][0]
                q_fid = g_fids[indices][0][0]
                q_name = g_names[indices][0][0]
                print("Next query (name, pid, cid, fid): ", q_name, q_pid, q_camid, q_fid)

                # extract next img features
                ori_w = 0.5
                run_w = 0.0
                new_w = 0.5
                with torch.no_grad():
                    next_path = osp.normpath("data/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/" + q_name)
                    next_img = read_image(next_path)
                    if use_gpu: next_img = next_img.cuda()
                    features = model(next_img.unsqueeze(0))
                    qf_i = (ori_w * qf_orig) + (run_w * qf_i) + (new_w * features.data.cpu())

        print("\nFinal query {} stats ----------".format(q_idx))
        print("img seen: {}".format(sum(q_img_seen_arr[:-1])))
        print("img tot.: {}".format(sum(q_img_seen_arr[:-1] + q_img_elim_arr[:-1])))
        print("matches found: {}".format(q_match_found))
        print("matches pres.: {}".format(q_match_pres))
        print("delay: {}".format(sum(q_delay_arr[:-1])))
        print("num inst: {}".format(num_inst))
        print("acc. (recall) {:1.3f}".format(t_pos / (1e-8 + num_inst)))
        print("acc. (precis) {:1.3f}".format(t_pos / (1e-8 + t_pos + f_pos)))

        # update aggregate stats
        tot_img_seen += sum(q_img_seen_arr[:-1])
        tot_img_elim += sum(q_img_elim_arr[:-1])
        tot_match_found += q_match_found
        tot_match_pres  += q_match_pres
        tot_delay += sum(q_delay_arr[:-1])
        tot_t_pos += t_pos
        tot_f_pos += f_pos
        tot_t_neg += t_neg
        tot_f_neg += (num_inst - t_pos)

        print("\nAggregate results ----------")
        print("img seen: {}".format(tot_img_seen))
        print("img tot.: {}".format(tot_img_seen + tot_img_elim))
        print("matches found: {}".format(tot_match_found))
        print("matches pres.: {}".format(tot_match_pres))
        print("delay (avg.): {}".format(tot_delay / (q_idx + 1)))
        print("mAP: {:.1%}".format(np.mean(all_AP)))
        print("acc. (recall) {}".format(tot_t_pos / (tot_t_pos + tot_f_neg)))
        print("acc. (precis) {}".format(tot_t_pos / (tot_t_pos + tot_f_pos)))

    min_len = min(map(len, all_cmc))
    all_cmc = [cmc[:min_len] for cmc in all_cmc]
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / num_valid_q

    print("CMC curve")
    for r in ranks:
        if r-1 < len(cmc):
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]

if __name__ == '__main__':
    main()