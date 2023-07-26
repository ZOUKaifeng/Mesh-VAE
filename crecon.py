"""
Created on Mon Oct 05 13:43:10 2020

@Author: Kaifeng

@Contact: kaifeng.zou@unistra.fr

main function 
"""



import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset, DataLoader

import pandas as pd
import mesh_operations
from config_parser import read_config
from data import MeshData
from model import get_model
from transform import Normalize
from utils import *
from psbody.mesh import Mesh
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import matplotlib.pyplot as plt

import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 

def save_obj(filename, vertices, faces):
    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))



def adjust_learning_rate(optimizer, lr_decay):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))


def classifier_(net, x):

    x = net.encoder(x)
    y_hat = net.classifier(x)
    index_pred = torch.argmax(y_hat,  dim = 1)
   #pred = torch.argmax(oppo_sex, dim = 1)
 
    return  index_pred

def euclidean_distances(gt, pred):
    return np.sqrt(((gt-pred)**2).sum(-1))


def train(model, dvae, train_loader, len_dataset, optimizer, device, criterion):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    for data in train_loader:
        x,x_gt, label, f, gt_mesh , R,m,s = data
        x_gt, label = x_gt.to(device).float(), label.to(device)


        diff, _ = estimate_diff(dvae, x_gt, label, "train")


        optimizer.zero_grad()
       
        pred = model(diff)

        
        loss = criterion(pred, label)



        loss.backward()
        optimizer.step()

        total_loss += loss.cpu().detach().numpy()
       # print(torch.nn.functional.softmax(pred))
        #predicted = torch.argmax(torch.nn.functional.softmax(pred), dim = -1)
        predicted = torch.argmax(F.softmax(pred), dim = -1)

      #  print(predicted)

        total += label.shape[0]
        correct += (predicted == label).sum().item()

    return total_loss / len_dataset, correct/total



def evaluate(model, dvae, test_loader, len_dataset, device, criterion, err_file = False):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    err = {}
    predicted_dist = np.empty((0))


    with torch.no_grad():
        for data in test_loader:
            x,x_gt, label, f, gt_mesh , R,m,s = data
            x_gt, label = x_gt.to(device).float(), label.to(device)


            diff, _ = estimate_diff(dvae, x_gt, label, "test")
            pred = model(diff)


            loss = criterion(pred, label)
    

            total_loss +=  loss.cpu().numpy()
            
            predicted =torch.argmax(F.softmax(pred), dim = -1)

            total += label.shape[0]
            correct += (predicted == label).sum().item()
           
            if err_file == True:
               
                predicted = predicted.cpu().numpy().squeeze(1)
                label = label.cpu().numpy().squeeze(1)
                pred = pred.cpu().numpy().squeeze(1)

                f = np.array(f)

                error_ind = np.where((predicted == label))


                for idx in range(label.shape[0]):
                    if predicted[idx] != label[idx]:
                        err.update({f[idx]: str(predicted[idx])})



    return total_loss / len_dataset, correct/total, err


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape



def estimate_diff(net, x, y,dtype):
    net = net.to(device)
    ori = copy.copy(x)
    if len(x.shape) == 2:
        x = x.reshape(1, -1, 3).to(device)
        y = torch.tensor(y).unsqueeze(0).to(device)

    with torch.no_grad():
        x = net.encoder(x)
        y_hat = net.classifier(x)    
        index_pred = torch.argmax(y_hat,  dim = 1)
    
        correct = torch.sum(index_pred == y).item()

  

        if dtype != "train":
            sex_hot = F.one_hot(index_pred, num_classes = 2)
            x = torch.cat([sex_hot, x], -1)
        else:
            
            sex_hot =  F.one_hot(y, num_classes = 2)
            x = torch.cat([sex_hot, x], -1)

        x_mean = net.z_mean(x)

        #sex_hot = #_m = F.one_hot(torch.ones_like(y).to(device), num_classes = 2)
        recon =  net.sample(sex_hot, x_mean)

        oppo = 1-sex_hot
        recon_oppo =  net.sample(oppo, x_mean)


        diff_1 = ori - recon_oppo
        diff_2 = ori - recon

        diff = torch.cat((diff_1, diff_2), dim=-1)#diff_1 + diff_2


    return diff, correct


def main(args):

    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)
    print(args.conf)
    config = read_config(args.conf)

    print('Initializing parameters')
    # template_mesh = pc2mesh(template)

 

    checkpoint_dir = config['checkpoint_dir']

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)



 
    print("Using device:",device)


    root_dir = config['root_dir']

    error_file = config['error_file']
    log_path = config['log_file']
    random_seeds = config['random_seeds']

    test_size = config['test_size']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    template_file_path = config['template']
    val_losses, accs, durations = [], [], []
    checkpoint_file = config['checkpoint_file']

    net = get_model(config, device)
    print('loading template...', config['template'])


    config_dvae = read_config(args.conf)
    dvae = get_model(config_dvae, device, model_type="cheb_VAE", save_init = False)
    print("loading checkpoint for DVAE from ", checkpoint_file)
  
    checkpoint = torch.load(checkpoint_file)
    dvae.load_state_dict(checkpoint['state_dict'])   


    template_mesh = Mesh(filename=config['template'])
    template = np.array(template_mesh.v)
    faces = np.array(template_mesh.f)
    num_points = template.shape[0]


    #criterion = BCEFocalLoss()

    

    my_log = open(log_path, 'w')

    print('model type:', config['type'], file = my_log)
    print('optimizer type', opt, file = my_log)
    print('learning rate:', lr, file = my_log)


    start_epoch = 1
    print(checkpoint_file)
    criterion = torch.nn.CrossEntropyLoss()

    labels = {}
    dataset_index = []
    files = os.listdir(root_dir)
    for name in files:
        if not name.endswith(".obj") : continue
        name_ = name.split("_")
        dataset_index.append(name)
        if name_[1] == "f":
            labels[name] = 0
        else:
            labels[name] = 1

    acc = []

    import time

    for i in range(1):


        # train_, test_index = train_test_split(dataset_index, test_size=test_size, random_state = random_seeds)


        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state = random_seeds)  # 5-folds repeated 10 times  

        n = 0

        y = np.ones(len(dataset_index))
        me = 0
        si = 0
        train_me = 0
        train_si = 0
        train_error_ = 0
        best_acc = 0
        max_error = []
        max_train_error = []


        for train_index, test_index in skf.split(dataset_index, y):
            train_, valid_index = train_test_split(np.array(dataset_index)[train_index], test_size=test_size, random_state = random_seeds)

            train_loss_history = []
            valid_loss_history = []
            train_kld_history = []
            valid_kld_history = []
            train_rec_loss_history = []
            valid_rec_loss_history = []

            error_history = []
            sigma_history = []

            train_error_history = []
  

            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

            n+=1
            train_dataset = MeshData(root_dir, train_, config, labels, dtype = 'train', template = template, pre_transform = Normalize())
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            valid_dataset = MeshData( root_dir, valid_index, config, labels, dtype = 'test', template = template, pre_transform = Normalize())
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


            best_val_acc = 0

            if args.train:

                for epoch in range(start_epoch, total_epochs + 1):

                # for epoch in range(10):
                    train_loss, train_acc = train(net, dvae, train_loader, len(train_loader), optimizer, device, criterion)
                    val_loss, valid_acc, _ = evaluate(net,dvae, valid_loader, len(valid_loader),device, criterion)

                    if valid_acc >= best_val_acc:
                        save_model(net, optimizer, n, train_loss, val_loss, checkpoint_dir)
                     #   torch.save(net, 'best_val_model'+str(n)+'.pth')
                        best_val_acc = valid_acc

                    # val_loss_history.append(val_loss.detach().cpu().numpy())
                    # train_loss_history.append(train_loss.detach().cpu().numpy())

                    
                    print('epoch ', epoch,' Train loss ', train_loss, 'train acc',train_acc, ' Val loss ', val_loss, 'acc ', valid_acc)
                    print('epoch ', epoch,' Train loss ', train_loss, 'train acc',train_acc, ' Val loss ', val_loss, 'acc ', valid_acc, file = my_log)
        



            if args.test:

                test_dataset = MeshData( root_dir, test_index, config, labels, dtype = 'test', template = template, pre_transform = Normalize())  
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                test_loss, test_acc, _ = evaluate(net, dvae, test_loader, len(test_loader), device, criterion, err_file = False)

                print( 'test loss ', test_loss, 'test acc',test_acc)







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-t', '--train',action='store_true')
    parser.add_argument('-s', '--test',action='store_true')
    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), './files/default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)
    acc = 0

    acc = main(args)
