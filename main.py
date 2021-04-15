# coding: utf-8
#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import math
import time
import torch
import torch.nn as nn
from models import tSTNN
import numpy as np
import sys
import os
from utils import *
import Optim
from easydict import EasyDict as edic

args = edic()
args.data = './data/us_hhs/data_example.txt'  
args.train = 0.7
args.valid = 0.1
args.model = 'tSTNN'                             
args.sim_mat = './data/us_hhs/ind_mat.txt'
args.hidRNN = 20
args.residual_window = 4    
args.ratio = 1              
args.output_fun = None
args.save_dir = './save/'
args.save_name = 'best_checkpoint'
args.optim = 'adam'
args.dropout = 0.2
args.epochs = 100
args.clip = 1.              
args.lr = 1e-3
args.weight_decay = 0
args.batch_size = 16
args.horizon = 1           
args.window = 20        
args.metric = 1             
args.normalize = 0          
args.seed = 666
args.gpu = 0
args.cuda = True

def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        X = X.cuda()
        
        output = model(X);
        
        if predict is None:
            predict = output.cpu();
            test = Y.cpu();
        else:
            predict = torch.cat((predict,output.cpu()));
            test = torch.cat((test, Y.cpu()));

        scale = loader.scale.expand(output.size(0), loader.m)
        total_loss += evaluateL2(output * scale , Y * scale ).item()
        total_loss_l1 += evaluateL1(output * scale , Y * scale ).item()
        n_samples += (output.size(0) * loader.m);

    rse = math.sqrt(total_loss / n_samples)/loader.rse
    rae = (total_loss_l1/n_samples)/loader.rae.cuda()
    correlation = 0;

    predict = predict.data.numpy();
    Ytest = test.data.numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);

    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();

    return predict, Ytest, rse, rae, correlation;


def train(loader, data, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;
    counter = 0
    for inputs in loader.get_batches(data, batch_size, True):
        counter += 1
        X, Y = inputs[0], inputs[1]
        X = X.cuda()
        model.zero_grad();
        output = model(X);
        scale = loader.scale.expand(output.size(0), loader.m)  
        loss = criterion(output * scale, Y * scale);
        loss.backward();
        optim.step();
        total_loss += loss.item();
        n_samples += (output.size(0) * loader.m);
    return total_loss / n_samples


args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args)

model = eval(args.model).Model(args, Data)
print('model:', model)
if args.cuda:
    model.cuda()

criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();


best_val = 10000000;
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip, weight_decay = args.weight_decay,
)

try:
    print('begin training');
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train, model, criterion, optim, args.batch_size)
        predict, Ytest, val_loss, val_rae, val_corr = evaluate(Data, Data.valid, model, evaluateL2, evaluateL1, args.batch_size);
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.8f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        if val_loss < best_val:
            best_val = val_loss
            model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('best validation');
            predict, Ytest, test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
            print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 100)
    print('Exiting from training early')

model_path = '%s/%s.pt' % (args.save_dir, args.save_name)

predict, Ytest, test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

