# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:43:48 2019

@author: Gabriel Hsu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:21:22 2019

@author: Gabriel Hsu
"""
from __future__ import print_function, division
import os
import argparse
import time


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from model import iterativeFCN
from dataset_simple import CSI_Dataset_Patched
from metrics import DiceCoeff, ASSD

def seg_loss(pred, target, weight):
    size = pred.shape[0]
    FP = torch.sum(weight*(1-target)*pred)
    FN = torch.sum(weight*(1-pred)*target)
    return FP/size, FN/size
    
#%%
def train_single(args, model, device, img_patch, ins_patch, gt_patch, weight, c_label, optimizer):

    model.train()
    correct = 0
    
    img_patch = img_patch.float()
    ins_patch = ins_patch.float()
    gt_patch = gt_patch.float()
    weight = weight.float()
    c_label = c_label.float()
    
    
    #pick a random scan
    optimizer.zero_grad()
    
    #concatenate the img_patch and ins_patch
    input_patch = torch.cat((img_patch, ins_patch), dim=1)
    input_patch, gt_patch, weight, c_label = input_patch.to(device), gt_patch.to(device), weight.to(device), c_label.to(device)
    S, C = model(input_patch.float())        
    
    pred = S.detach().cpu().numpy()
    pred = np.squeeze(pred)
    

    
    #compute the loss
    lamda = 0.1 
    
    #segloss 
    FP, FN = seg_loss(S, gt_patch, weight) 
    
    s_loss = lamda*FP + FN
    c_loss = -1*c_label*torch.log(C)-(1-c_label)*torch.log(1-C)


    train_loss = s_loss + c_loss
    
    print(C.round())
    
    if C.round() == c_label:
        correct = 1
    
    #optimize the parameters
    train_loss.backward()
    optimizer.step()
    
    
    

    return train_loss, correct

def test_single(args, model, device, img_patch, ins_patch, gt_patch, weight, c_label):
    
    model.eval()
    correct = 0
    
    img_patch = img_patch.float()
    ins_patch = ins_patch.float()
    gt_patch = gt_patch.float()
    weight = weight.float()
    c_label = c_label.float()
    
    input_patch = torch.cat((img_patch, ins_patch), dim=1)
    input_patch, gt_patch, weight, c_label = input_patch.to(device), gt_patch.to(device), weight.to(device), c_label.to(device)
    
    with torch.no_grad():
        S, C = model(input_patch.float())

    S = S.float()
    S = torch.round(S)
    
    pred = S.detach().cpu().numpy()
    pred = np.squeeze(pred)

    gt_patch = gt_patch.float()
    c_label = c_label.float()
    
    #compute the loss
    lamda = 0.1 
    
    #segloss 
    FP, FN = seg_loss(S, gt_patch, weight) 
    
    s_loss = lamda*FP + FN
    c_loss = -1*c_label*torch.log(C)-(1-c_label)*torch.log(1-C)
    
    if C.round() == c_label:
        correct = 1

    test_loss = s_loss + c_loss
        
    return test_loss, correct
    
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
    
    args = parser.parse_known_args()[0]
    #args = parser.parse_args()

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = './drive/My Drive/patches'
    
    
    # Create FCN
    model = iterativeFCN().to('cuda')


    batch_size = args.batch_size
    batch_size_valid = batch_size
    
    train_clabel = list(pd.read_excel(os.path.join(data_root, 'train/train_label.xlsx'))[0])   
    #test_clabel = list(pd.read_excel(os.path.join(data_root, 'test/test_label.xlsx'))[0])   

    
    train_set = CSI_Dataset_Patched(data_root, train_clabel, subset='train')
    #test_set = CSI_Dataset_Patched(data_root, test_clabel, subset='test')
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size
    )
    
    """

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size_valid
    )
    
    """
    
#%%    
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    best_test_loss = 1e+7
    
    start_time = time.time()

    train_interval = 50
    eval_interval =  10
    
    # Start Training
    for epoch in range(int(len(train_loader)/train_interval)):
        
        epoch_train_loss = []
        epoch_test_loss = []
        epoch_train_accuracy = 0.
        epoch_test_accuracy = 0.
        correct_train_count = 0
        correct_test_count = 0
        
        #training process
        for i in range(train_interval):
            img_patch, ins_patch, gt_patch, weight, c_label = next(iter(train_loader))
            t_loss, t_c = train_single(args, model, device, img_patch, ins_patch, gt_patch, weight, c_label, optimizer)
            epoch_train_loss.append(t_loss)
            correct_train_count+=t_c
            
        epoch_train_accuracy = correct_train_count/train_interval
        avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        
        print('Train Epoch: {} \t Loss: {:.6f}\t acc: {:.6f}'.format(epoch
              , avg_train_loss
              , epoch_train_accuracy))
            
        """
        #validation process
        for i in range(eval_interval):
            img_patch, ins_patch, gt_patch, weight, c_label = next(iter(test_loader))
            v_loss, v_c = test_single(args, model, device, img_patch, ins_patch, gt_patch, weight, c_label)
            epoch_test_loss.append(v_loss)
            correct_test_count+=v_c
            
        epoch_test_accuracy = correct_test_count/eval_interval
        avg_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
        
        print('Validation Epoch: {} \t Loss: {:.6f}\t acc: {:.6f}'.format(epoch
              , avg_test_loss
              , epoch_test_accuracy))
        
        if avg_test_loss < best_test_loss:
            best_test_dicescore = v_mean_dice
            print('--- Saving model at Avg Test Loss:{:.2f}% ---'.format(avg_test_loss))
            torch.save(model.state_dict(),'./drive/My Drive/IterativeFCN_best.pth')
        
        print('-------------------------------------------------------')
        
        train_loss.append(epoch_train_loss)
        test_loss.append(epoch_test_loss)
        train_acc.append(epoch_train_accuracy)
        test_acc.append(epoch_test_accuracy)
        
        """

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
    plt.title("Train/Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(x, train_acc,label="train acc")
    plt.plot(x, test_acc, color='red', label="validation acc")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    