import os
import time
import argparse
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import CSIDataset
from utils.metrics import DiceCoeff
from iterativeFCN import IterativeFCN


def seg_loss(pred, target, weight):
    FP = torch.sum(weight * (1 - target) * pred)
    FN = torch.sum(weight * (1 - pred) * target)
    return FP, FN


def train_single(model, device, img_patch, ins_patch, gt_patch, weight, c_label, optimizer):
    torch.cuda.empty_cache()

    model.train()
    correct = 0

    # convert data to float, just in case
    img_patch = img_patch.float()
    ins_patch = ins_patch.float()
    gt_patch = gt_patch.float()
    weight = weight.float()
    c_label = c_label.float()

    # pick a random scan
    optimizer.zero_grad()

    # concatenate the img_patch and ins_patch
    input_patch = torch.cat((img_patch, ins_patch), dim=1)
    input_patch, gt_patch, weight, c_label = input_patch.to(device), gt_patch.to(device), weight.to(device), c_label.to(
        device)

    S, C = model(input_patch.float())

    # calculate dice coefficient
    pred = torch.round(S).detach()
    train_dice_coef = DiceCoeff(pred, gt_patch.detach())

    print(train_dice_coef * 100, '%')

    # calculate total loss
    lamda = 0.1
    FP, FN = seg_loss(S, gt_patch, weight)
    s_loss = lamda * FP + FN
    c_loss = F.binary_cross_entropy(torch.unsqueeze(C, dim=0), c_label)

    print(s_loss.item(), c_loss.item())
    train_loss = s_loss + c_loss

    if C.round() == c_label:
        correct = 1

    # optimize the parameters
    train_loss.backward()
    optimizer.step()

    return train_loss.item(), correct, train_dice_coef


def test_single(device, img_patch, ins_patch, gt_patch, weight, c_label):
    torch.cuda.empty_cache()

    model.eval()
    correct = 0

    img_patch = img_patch.float()
    ins_patch = ins_patch.float()
    gt_patch = gt_patch.float()
    weight = weight.float()
    c_label = c_label.float()

    input_patch = torch.cat((img_patch, ins_patch), dim=1)
    input_patch, gt_patch, weight, c_label = input_patch.to(device), gt_patch.to(device), weight.to(device), c_label.to(
        device)

    with torch.no_grad():
        S, C = model(input_patch.float())

    # calculate dice coefficient
    pred = torch.round(S).detach()
    test_dice_coef = DiceCoeff(pred, gt_patch.detach())

    print(test_dice_coef * 100, '%')

    # calculate total loss
    lamda = 0.1
    FP, FN = seg_loss(S, gt_patch, weight)
    s_loss = lamda * FP + FN
    c_loss = F.binary_cross_entropy(torch.unsqueeze(C, dim=0), c_label)
    print(s_loss.item(), c_loss.item())

    if C.round() == c_label:
        correct = 1

    test_loss = s_loss + c_loss

    return test_loss.item(), correct, test_dice_coef


if __name__ == "__main__":
    # Version of Pytorch
    print("Pytorch Version:", torch.__version__)

    # Training args
    parser = argparse.ArgumentParser(description='Iterative Fully Convolutional Network')
    parser.add_argument('--dataset', type=str, default='./crop_isotropic_dataset',
                        help='path of processed dataset')
    parser.add_argument('--weight', type=str, default='./weights',
                        help='path of processed dataset')
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume training by loading last snapshot')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--iterations', type=int, default=100000, metavar='N',
                        help='number of iterations to train (default: 5000)')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='number of iterations to train (default: 5000)')
    parser.add_argument('--eval_iters', type=int, default=20, metavar='N',
                        help='number of iterations to train (default: 5000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and check if we want to resume training
    model = IterativeFCN(num_channels=8).to('cuda')
    if args.resume:
        """
            load state from last training snapshot
        """
        pass
    # model.load_state_dict(torch.load('./IterativeFCN_best_train_lamda.pth'))

    batch_size = args.batch_size
    batch_size_valid = batch_size

    train_dataset = CSIDataset(args.dataset, subset='train')
    test_dataset = CSIDataset(args.dataset, subset='test')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    train_dice = []
    test_dice = []
    best_train_loss = 0
    best_test_dice = 0

    total_iteration = args.iterations
    train_interval = args.log_interval
    eval_interval = args.eval_iters

    # Start Training
    for epoch in range(int(total_iteration / train_interval)):

        start_time = time.time()
        epoch_train_dice = []
        epoch_test_dice = []
        epoch_train_loss = []
        epoch_test_loss = []
        epoch_train_accuracy = 0.
        epoch_test_accuracy = 0.
        correct_train_count = 0
        correct_test_count = 0

        # training process
        for i in range(train_interval):
            img_patch, ins_patch, gt_patch, weight, c_label = next(iter(train_loader))
            t_loss, t_c, t_dice = train_single(model, device, img_patch, ins_patch, gt_patch, weight, c_label,
                                               optimizer)
            epoch_train_loss.append(t_loss)
            epoch_train_dice.append(t_dice)
            correct_train_count += t_c

        epoch_train_accuracy = correct_train_count / train_interval
        avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        avg_train_dice = sum(epoch_train_dice) / len(epoch_train_dice)

        print('Train Epoch: {} \t Loss: {:.6f}\t acc: {:.6f}%\t dice: {:.6f}%'.format(epoch,
                                                                                      avg_train_loss,
                                                                                      epoch_train_accuracy * 100,
                                                                                      avg_train_dice * 100))

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            print('--- Saving model at Avg Train Dice:{:.2f}%  ---'.format(avg_train_dice * 100))
            torch.save(model.state_dict(), os.path.join(args.weight, '.IterativeFCN_best_train.pth'))

        # validation process
        for i in range(eval_interval):
            img_patch, ins_patch, gt_patch, weight, c_label = next(iter(test_loader))
            v_loss, v_c, v_dice = test_single(model, device, img_patch, ins_patch, gt_patch, weight, c_label)
            epoch_test_loss.append(v_loss)
            epoch_test_dice.append(v_dice)
            correct_test_count += v_c

        epoch_test_accuracy = correct_test_count / eval_interval
        avg_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
        avg_test_dice = sum(epoch_test_dice) / len(epoch_test_dice)

        print('Validation Epoch: {} \t Loss: {:.6f}\t acc: {:.6f}%\t dice: {:.6f}%'.format(epoch,
                                                                                           avg_test_loss,
                                                                                           epoch_test_accuracy * 100,
                                                                                           avg_test_dice * 100))

        if avg_test_dice > best_test_dice:
            best_test_dice = avg_test_dice
            print('--- Saving model at Avg Train Dice:{:.2f}%  ---'.format(avg_test_dice * 100))
            torch.save(model.state_dict(), os.path.join(args.weight, './IterativeFCN_best_valid.pth'))

        print('-------------------------------------------------------')

        train_loss.append(epoch_train_loss)
        test_loss.append(epoch_test_loss)
        train_acc.append(epoch_train_accuracy)
        test_acc.append(epoch_test_accuracy)

        print("--- %s seconds ---" % (time.time() - start_time))
