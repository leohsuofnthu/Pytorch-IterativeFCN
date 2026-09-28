import os
import time
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import CSIDataset
from utils.metrics import DiceCoeff, Segloss
from iterativeFCN import IterativeFCN

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)


def train_single(model, device, img_patch, ins_patch, gt_patch, weight, c_label, optimizer, lamda=0.1):
    model.train()
    correct = 0

    img_patch = img_patch.float()
    ins_patch = ins_patch.float()
    gt_patch = gt_patch.float()
    weight = weight.float()
    c_label = c_label.float()

    optimizer.zero_grad()

    input_patch = torch.cat((img_patch, ins_patch), dim=1)
    input_patch = input_patch.to(device)
    gt_patch = gt_patch.to(device)
    weight = weight.to(device)
    c_label = c_label.to(device)

    S, C = model(input_patch)

    pred = torch.round(S).detach()
    train_dice_coef = DiceCoeff(pred, gt_patch.detach())

    # Paper: L_seg = λ·FP_soft + FN_soft (distance-weighted); λ anneals 0.1 → 1
    FP, FN = Segloss(S, gt_patch, weight)
    s_loss = lamda * FP + FN
    c_loss = F.binary_cross_entropy(C.view_as(c_label.view(-1)), c_label.view(-1))
    train_loss = s_loss + c_loss

    logging.info(
        'train_dice_coef: %s, S Loss: %s, C Loss: %s',
        train_dice_coef, s_loss.item(), c_loss.item(),
    )

    if torch.equal(C.view(-1).round(), c_label.view(-1).round()):
        correct = 1

    train_loss.backward()
    optimizer.step()
    return train_loss.item(), correct, train_dice_coef


def test_single(model, device, img_patch, ins_patch, gt_patch, weight, c_label, lamda=0.1):
    model.eval()
    correct = 0

    img_patch = img_patch.float()
    ins_patch = ins_patch.float()
    gt_patch = gt_patch.float()
    weight = weight.float()
    c_label = c_label.float()

    input_patch = torch.cat((img_patch, ins_patch), dim=1)
    input_patch = input_patch.to(device)
    gt_patch = gt_patch.to(device)
    weight = weight.to(device)
    c_label = c_label.to(device)

    with torch.no_grad():
        S, C = model(input_patch)

    pred = torch.round(S).detach()
    test_dice_coef = DiceCoeff(pred, gt_patch.detach())

    FP, FN = Segloss(S, gt_patch, weight)
    s_loss = lamda * FP + FN
    c_loss = F.binary_cross_entropy(C.view_as(c_label.view(-1)), c_label.view(-1))

    logging.info(
        'val_dice_coef: %s, S Loss: %s, C Loss: %s',
        test_dice_coef, s_loss.item(), c_loss.item(),
    )

    if torch.equal(C.view(-1).round(), c_label.view(-1).round()):
        correct = 1

    return (s_loss + c_loss).item(), correct, test_dice_coef


def learning_rate_for_iteration(iteration):
    """Repo schedule (differs from paper's constant 1e-3): 1e-3 / 1e-4 / 1e-5."""
    if iteration <= 10000:
        return 1e-3
    if iteration <= 20000:
        return 1e-4
    return 1e-5


def lambda_fp(iteration, n_max):
    """
    Paper λ schedule: start at λ_min=0.1, anneal toward 1 so FP and FN
    become equally weighted around n_max/2.
    λ(n) = λ_min + (1-λ_min) / (1 + exp(-ϑ(n))), ϑ=(n - n_max/2)/(n_max/10)
    """
    lam_min = 0.1
    theta = (iteration - n_max / 2.0) / (n_max / 10.0)
    return lam_min + (1.0 - lam_min) / (1.0 + np.exp(-theta))


def main():
    logging.info('Pytorch Version: %s', torch.__version__)

    parser = argparse.ArgumentParser(description='Iterative Fully Convolutional Network')
    parser.add_argument('--dataset', type=str, default='./crop_isotropic_dataset',
                        help='path of processed dataset (crop_isotropic_dataset)')
    parser.add_argument('--weight', type=str, default='./weights',
                        help='directory to save best model weights')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='directory for training snapshots')
    parser.add_argument('--resume', action='store_true',
                        help='resume from latest checkpoint')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='training batch size (paper uses 1)')
    parser.add_argument('--iterations', type=int, default=80000,
                        help='number of training iterations (paper-scale default)')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='iterations between logging / validation / snapshot')
    parser.add_argument('--eval_iters', type=int, default=20,
                        help='number of validation patches per log interval')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial Adam learning rate')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_channels', type=int, default=64,
                        help='feature width (must be 64 to match released pretrained weights)')
    parser.add_argument('--device', type=str, default=None,
                        help="cuda / cpu (default: auto)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s', device)

    os.makedirs(args.weight, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)

    # num_channels=64 matches weights/IterativeFCN_pretrained.pth
    model = IterativeFCN(num_channels=args.num_channels).to(device)

    train_dataset = CSIDataset(args.dataset, subset='train')
    test_dataset = CSIDataset(args.dataset, subset='test')
    if len(train_dataset) == 0:
        raise FileNotFoundError(
            f'No training images under {args.dataset}/train/img. '
            'Run: python -m data.preprocessing --dataset <CSI_root>'
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Paper uses Adam with elevated momentum 0.99 (mapped to β1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.99, 0.999))

    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    train_dice, test_dice = [], []
    best_train_loss = float('inf')
    best_test_dice = 0.0
    iteration = 1

    if args.resume:
        ckpt_path = os.path.join(args.checkpoints, 'latest_checkpoints.pth')
        logging.info('Resume Training: loading %s', ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration'] + 1
        train_loss = checkpoint.get('train_loss', [])
        test_loss = checkpoint.get('test_loss', [])
        train_acc = checkpoint.get('train_acc', [])
        test_acc = checkpoint.get('test_acc', [])
        train_dice = checkpoint.get('train_dice', [])
        test_dice = checkpoint.get('test_dice', [])
        best_train_loss = checkpoint.get('best_train_loss', float('inf'))
        best_test_dice = checkpoint.get('best_test_dice', 0.0)

    train_iter = iter(train_loader)

    while iteration <= args.iterations:
        start_time = time.time()
        for pg in optimizer.param_groups:
            pg['lr'] = learning_rate_for_iteration(iteration)

        try:
            img_patch, ins_patch, gt_patch, weight, c_label = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            img_patch, ins_patch, gt_patch, weight, c_label = next(train_iter)

        lam = float(lambda_fp(iteration, args.iterations))
        t_loss, t_c, t_dice = train_single(
            model, device, img_patch, ins_patch, gt_patch, weight, c_label, optimizer, lamda=lam
        )

        # accumulate within the log window
        if not hasattr(main, '_win') or iteration % args.log_interval == 1:
            main._win = {
                'train_loss': [], 'train_dice': [], 'train_correct': 0,
            }
        main._win['train_loss'].append(t_loss)
        main._win['train_dice'].append(t_dice)
        main._win['train_correct'] += t_c

        if iteration % args.log_interval == 0:
            avg_train_loss = sum(main._win['train_loss']) / len(main._win['train_loss'])
            avg_train_dice = (sum(main._win['train_dice']) / len(main._win['train_dice'])) * 100
            epoch_train_accuracy = (main._win['train_correct'] / args.log_interval) * 100

            logging.info(
                'Iter {}-{}: Loss: {:.6f}\t acc: {:.6f}%\t dice: {:.6f}%'.format(
                    iteration - args.log_interval + 1,
                    iteration,
                    avg_train_loss,
                    epoch_train_accuracy,
                    avg_train_dice,
                )
            )

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                path = os.path.join(args.weight, 'IterativeFCN_best_train.pth')
                logging.info('--- Saving best-train model (loss={:.6f}) to %s ---', best_train_loss, path)
                torch.save(model.state_dict(), path)

            epoch_test_loss, epoch_test_dice = [], []
            correct_test_count = 0
            test_iter = iter(test_loader)
            n_eval = min(args.eval_iters, max(1, len(test_dataset)))
            for _ in range(n_eval):
                try:
                    batch = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    batch = next(test_iter)
                v_loss, v_c, v_dice = test_single(model, device, *batch)
                epoch_test_loss.append(v_loss)
                epoch_test_dice.append(v_dice)
                correct_test_count += v_c

            avg_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
            avg_test_dice = (sum(epoch_test_dice) / len(epoch_test_dice)) * 100
            epoch_test_accuracy = (correct_test_count / n_eval) * 100

            logging.info(
                'Iter {}-{} eval: Loss: {:.6f}\t acc: {:.6f}%\t dice: {:.6f}%'.format(
                    iteration - args.log_interval + 1,
                    iteration,
                    avg_test_loss,
                    epoch_test_accuracy,
                    avg_test_dice,
                )
            )

            if avg_test_dice > best_test_dice:
                best_test_dice = avg_test_dice
                path = os.path.join(args.weight, 'IterativeFCN_best_valid.pth')
                logging.info('--- Saving best-valid model (dice={:.2f}%) to %s ---', best_test_dice, path)
                torch.save(model.state_dict(), path)

            train_loss.append(avg_train_loss)
            test_loss.append(avg_test_loss)
            train_acc.append(epoch_train_accuracy)
            test_acc.append(epoch_test_accuracy)
            train_dice.append(avg_train_dice)
            test_dice.append(avg_test_dice)

            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_dice': train_dice,
                'test_dice': test_dice,
                'best_train_loss': best_train_loss,
                'best_test_dice': best_test_dice,
            }, os.path.join(args.checkpoints, 'latest_checkpoints.pth'))
            logging.info('--- Snapshot saved (%.1fs) ---', time.time() - start_time)

        iteration += 1


if __name__ == '__main__':
    main()
