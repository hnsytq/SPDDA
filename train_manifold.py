import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from utils_HSI import sample_gt, metrics, seed_worker, test_hsi, hsi_metrics
from datasets import get_dataset, HyperX
import random
import os
import time
import numpy as np
import pandas as pd
import argparse
from con_losses import SupConLoss
from network import discrim_hyperG
from datetime import datetime
from hsi_loss import SelfHSILoss
from network.SPDDA import SPDDA

parser = argparse.ArgumentParser(description='PyTorch SPDDA')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='D:/HSI_Cross-scene/Pavia/')

parser.add_argument('--source_name', type=str, default='paviaU',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='paviaC',
                    help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-3,
                         help="Learning rate, set by the model if not specified.")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=256,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--pro_dim', type=int, default=128)
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=1233,
                    help='random seed ')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('--num_epoch', type=int, default=400,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=0,
                    help='multiple of of data augmentation')
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--d_se', type=int, default=64)
parser.add_argument('--weight_1', type=float, default=0.1)
parser.add_argument('--weight_2', type=float, default=0.1)
parser.add_argument('--lr_scheduler', type=str, default='none')

group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")

args = parser.parse_args()


def inter_domain_func(rich_hsi, hsi):
    rand = torch.nn.init.uniform_(torch.empty(len(rich_hsi), 1, 1, 1)).cuda()

    rand_spec = random.sample(range(rich_hsi.shape[1]), hsi.shape[1])
    rand_spec = sorted(rand_spec)
    min_x = rich_hsi[:, rand_spec, :, :]
    x_ID = rand * hsi + (1 - rand) * min_x
    return x_ID


def evaluate(net, val_loader, hyperparameter, gpu, tgt=False):
    ps = []
    ys = []
    for i, (x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(gpu)
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max() + 1)
        print(results['Confusion_matrix'], '\n', 'AA:', results['class_acc'], '\n', 'OA:', results['Accuracy'], '\n',
              'Kappa:', results['Kappa'])
        probility = test_hsi(net, val_loader.dataset.data, hyperparameter)
        np.save(args.source_name + 'tsne_pred.npy', probility)
        prediction = np.argmax(probility, axis=-1)

        run_results = hsi_metrics(prediction, val_loader.dataset.label - 1, [-1],
                                  n_classes=hyperparameter['n_classes'])
        print(run_results)
        return acc, results, prediction
    else:
        return acc


def evaluate_tgt(cls_net, gpu, loader, hyperparameter, modelpath):
    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['Discriminator'])
    cls_net.eval()
    teacc, best_results, pred = evaluate(cls_net, loader, hyperparameter, gpu, tgt=True)
    return teacc, best_results, pred


def experiment():
    settings = locals().copy()
    print(settings)
    hyperparams = vars(args)
    print(hyperparams)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name + 'to' + args.target_name)
    log_dir = os.path.join(root, str(args.lr) + '_dim' + str(args.pro_dim) +
                           '_pt' + str(args.patch_size) + '_bs' + str(args.batch_size) + '_' + time_str)
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # writer = SummaryWriter(log_dir)
    df = pd.DataFrame([args])
    df.to_csv(os.path.join(log_dir, 'params.txt'))

    seed_worker(args.seed)
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                        args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                        args.data_path)
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    n_classes = np.max(gt_src)
    print(n_classes)
    for i in range(n_classes):
        count_class = np.copy(gt_src)
        test_count = np.copy(gt_tar)
        # sparse_class=np.copy(sparse_ground_truth)

        count_class[(gt_src != i + 1)] = 0
        # sparse_class[(sparse_ground_truth != i + 1)[:H_SD, :W_SD]] = 0
        class_num = np.count_nonzero(count_class)

        test_count[gt_tar != i + 1] = 0

        print([i + 1], ':', class_num, np.count_nonzero(test_count))

    print("Total", np.count_nonzero(gt_src), np.count_nonzero(gt_tar))
    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    if 'whu' in args.source_name:
        train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
        val_gt_src, _, _, _ = sample_gt(train_gt_src, 0.1, mode='random')
    else:
        train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')

    # val_gt_src, _,_, _ = sample_gt(train_gt_src, 0.1, mode='random')
    print("All training number is,", np.count_nonzero(train_gt_src), np.count_nonzero(val_gt_src))
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    if tmp < 1:
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True, )
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 batch_size=hyperparams['batch_size'])
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  batch_size=hyperparams['batch_size'])
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    if 'whu' in args.source_name:
        pad = True
    else:
        pad = False

    model = discrim_hyperG.Discriminator(inchannel=N_BANDS, outchannel=args.pro_dim, num_classes=num_classes,
                                         patch_size=hyperparams['patch_size'], pad=pad).to(args.gpu)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    cls_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(device=args.gpu)
    recon_criterion = SelfHSILoss(args.weight_1, args.weight_2)

    best_acc = 0
    best_acc_tgt = 0
    taracc, taracc_list = 0, []

    model_G = SPDDA(N_BANDS).to(args.gpu)
    optim_G = optim.Adam(model_G.parameters(), lr=args.lr)
    for epoch in range(1, args.max_epoch + 1):

        t1 = time.time()
        loss_list = []
        model.train()
        for i, (x, y) in enumerate(train_loader):
            b, _, h, w = x.shape
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1

            optim_G.zero_grad()
            optimizer.zero_grad()
            rand_hsi_mix = model_G(x)
            p_mix, z_mix = model(rand_hsi_mix, mode='train')
            recon_loss = recon_criterion(x, rand_hsi_mix)
            g_loss = recon_loss + 0.025 * cls_criterion(p_mix, y.long())
            g_loss.backward(retain_graph=True)
            hsi_mix = model_G(x)

            if hsi_mix.shape[1] >= x.shape[1]:
                x_ID = inter_domain_func(hsi_mix, x)
            else:
                x_ID = inter_domain_func(x, hsi_mix)

            p_src_mix, z_src_mix = model(hsi_mix.detach(), mode='train')
            p_ID, z_ID = model(x_ID.detach(), mode='train')
            p_src, z_src = model(x, mode='train')
            z_aug = torch.cat([z_ID.unsqueeze(1), z_src_mix.unsqueeze(1)], dim=1)
            z_all = torch.cat([z_src.unsqueeze(1), z_aug], dim=1)
            p_all = torch.cat([p_src, p_src_mix, p_ID], dim=0)
            label_all = torch.cat([y.long(), y.long(), y.long()], dim=0)
            cls_loss = cls_criterion(p_all, label_all)
            con_loss = con_criterion(z_all, y, adv=False) + con_criterion(z_aug, adv=True)
            loss = cls_loss + con_loss
            loss.backward()
            optim_G.step()
            optimizer.step()
            loss_list.append([cls_loss.item(), con_loss.item(), recon_loss.item()])

        cls_loss_mean, con_loss_mean, recon_loss_mean = np.mean(loss_list, 0)

        model.eval()
        teacc = evaluate(model, val_loader, hyperparams, args.gpu)
        if best_acc < teacc:
            best_acc = teacc
            torch.save({'Discriminator': model.state_dict(),
                        'Generation': model_G.state_dict()},
                       os.path.join(log_dir, f'best.pkl'))
        t2 = time.time()

        print(
            f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f},'
            f' src_cls {cls_loss_mean:.4f}, con_loss {con_loss_mean:.4f},'
            f' recon_loss {recon_loss_mean:.4f}, teacc {teacc:.4f}')

        if epoch % args.log_interval == 0:
            pklpath = f'{log_dir}/best.pkl'
            taracc, result_tgt, pred = evaluate_tgt(model, args.gpu, test_loader, hyperparams, pklpath)
            if best_acc_tgt < taracc:
                best_acc_tgt = taracc
                best_results = result_tgt
                np.save(args.source_name + '_pred_OURS.npy', pred)
            taracc_list.append(round(taracc, 2))
            print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')
    with open('out_' + args.source_name + '_ablation.log', 'a') as f:
        f.write("\n")
        f.write('max:' + str(max(taracc_list)) + "\n")
        f.write('weight_1:' + str(args.weight_1) + "\n")
        f.write('weight_2:' + str(args.weight_2) + "\n")
        f.write("\n")
        f.write('OA:' + str(best_results['Accuracy']) + "\n")
        f.write('AA:' + str(best_results['class_acc']) + "\n")
        f.write('Kappa:' + str(best_results['Kappa']) + "\n")
        f.write("\n")

    f.close()


if __name__ == '__main__':
    experiment()
