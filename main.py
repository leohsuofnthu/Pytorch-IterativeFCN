# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:21:22 2019

@author: Gabriel Hsu
"""
from __future__ import print_function, division
import argparse
import time


import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from model import iterativeFCN
from dataset import CSI_Dataset
from metrics import DiceCoeff, ASSD

def seg_loss(pred, target, weight):
    size = pred.shape[0]
    FP = torch.sum(weight*(1-target)*pred)
    FN = torch.sum(weight*(1-pred)*target)
    return FP/size, FN/size
    
#%%
def train(args, model, device, train_loader, optimizer, epoch, max_epoch):
    model.train()
    train_loss = 0
    total_dice = 0
    
    for batch_idx, (img_patch, ins_patch, gt_patch, weight, c_label) in enumerate(train_loader):
        #pick a random scan
        optimizer.zero_grad()
        
        #concatenate the img_patch and ins_patch
        input_patch = torch.cat((img_patch, ins_patch.double()), dim=1)
        input_patch, gt_patch, weight, c_label = input_patch.to(device), gt_patch.to(device), weight.to(device), c_label.to(device)
        S, C = model(input_patch.float())        
        
        pred = S.detach().cpu().numpy()
        pred = np.squeeze(pred)
        
        #compute the loss
        phi_n = (epoch - (max_epoch/2))/(max_epoch/10)
        lamda = 0.1 + (1-0.1)/(1+np.exp(-1*phi_n))
#        print(lamda)
        
        #segloss 
        FP, FN = seg_loss(S, gt_patch, weight) 
        
        s_loss = lamda*FP + FN
        c_loss = -1*c_label*torch.log(C)-(1-c_label)*torch.log(1-C)


        total_loss = s_loss + c_loss
        
        d_score = DiceCoeff(S, gt_patch)
        #optimize the parameters
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
        total_dice += d_score
        
    return train_loss/len(train_loader), total_dice/len(train_loader)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    total_dice = 0
 
    #save model per 10 samples
    for batch_idx, (img_patch, ins_patch, gt_patch, weight, c_label) in enumerate(train_loader):
        
        input_patch = torch.cat((img_patch, ins_patch.double()), dim=1)
        input_patch, gt_patch, weight, c_label = input_patch.to(device), gt_patch.to(device), weight.to(device), c_label.to(device)
        
        with torch.no_grad():
            S, C = model(input_patch.float())

        S = S.float()
        S = torch.round(S)
        
        pred = S.detach().cpu().numpy()
        pred = np.squeeze(pred)
    
        gt_patch = gt_patch.float()
        c_label = c_label.float()
        

        #closs
        c_loss = -1*c_label*torch.log(C)-(1-c_label)*torch.log(1-C)

        

        test_loss += c_loss.item()
        d_score = DiceCoeff(S, gt_patch)

        
        test_loss += test_loss
        total_dice += d_score
        
        
    return test_loss/len(test_loader), total_dice/len(test_loader)
    
#%%Main
if  __name__ == "__main__" :   
    # Version of Pytorch
    print("Pytorch Version:", torch.__version__)
    
    # Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.99, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #root directory of dataset
    data_root = './isotropic_dataset'
    
    
    # Create FCN
    model = iterativeFCN().to('cuda')


    batch_size = args.batch_size
    batch_size_valid = batch_size

    
    train_set = CSI_Dataset(data_root, subset='train')
    test_set = CSI_Dataset(data_root, subset='test')
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size_valid
    )
    
#%%    
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_loss = []
    train_dicescore = []
    test_loss = []
    test_dicescore = []
    best_test_dicescore = -1
    
    start_time = time.time()

    
    # Start Training
    for epoch in range(1, args.epochs+1):
        #train loss.
        t_loss, t_mean_dice = train(args, model, device, train_loader, optimizer, epoch, args.epochs)
        print('Train Epoch: {} \t Loss: {:.6f}\t Mean_Dice Score(%):{}%'.format(
            epoch, t_loss, t_mean_dice*100))
        
        # validation loss
        v_loss, v_mean_dice = test(args, model, device, test_loader)
        print('Validation Epoch: {} \t Loss: {:.6f}\t Mean_Dice_Score(%):{}%'.format(
            epoch, v_loss, v_mean_dice*100))


        torch.cuda.empty_cache()
        print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
        print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
        print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
             
            
        train_loss.append(t_loss)
        test_loss.append(v_loss)
        test_dicescore.append(v_mean_dice)
        if v_mean_dice > best_test_dicescore:
            best_test_dicescore = v_mean_dice
            print('--- Saving model at Dice Score:{:.2f}% ---'.format(100 *  best_test_dicescore))
            torch.save(model.state_dict(),'IterativeFCN_best.pth')
        
        print('-------------------------------------------------------')
        
    print("--- %s seconds ---" % (time.time() - start_time))

        
    print("training:", len(train_loader))
    print("validation:", len(test_loader))
    x = list(range(1, args.epochs+1))
    #plot train/validation loss versus epoch
    plt.figure()
    plt.title("Train/Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.plot(x, train_loss,label="train loss")
    plt.plot(x, test_loss, color='red', label="validation loss")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    #plot train/validation loss versus epoch
    plt.figure()
    plt.title("Train/Validation Dice Score")
    plt.xlabel("Epochs")
    plt.ylabel("Mean IOU")
#    plt.plot(x, train_dicescore,label="train iou")
    plt.plot(x, test_dicescore, color='red', label="validation iou")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    
    # test set
    print("Best Test Mean Dice Score:",  best_test_dicescore)