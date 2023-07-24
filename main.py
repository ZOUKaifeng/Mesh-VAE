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
from torch_geometric.data import Dataset, DataLoader
import pandas as pd
import mesh_operations
from config_parser import read_config
from data import CTimageData
from model import get_model
from transform import Normalize
from utils import *
from psbody.mesh import Mesh
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
import random
import json
import time
 

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

def train(model, train_loader, len_dataset, optimizer, device,num_points, checkpoint_dir = None):
    model.train()
    total_loss = 0
    total_rec_loss = 0

    total_kld = 0

    norm_dict = np.load(os.path.join(checkpoint_dir, 'norm.npz'), allow_pickle = True)
    mean = torch.FloatTensor(norm_dict['mean'])
    std = torch.FloatTensor(norm_dict['std'])
    error_ = 0
    sigma = 0

    
    # lamda = 0.01

    max_error = 0
    total = 0
    total_correct = 0
    for data in train_loader:
        l1_reg = torch.tensor(0.0).to(device)

        x,x_gt, y, filename, gt_mesh , R,m,s = data

        x, x_gt = x.to(device), x_gt.to(device)
        sex_hot = F.one_hot(y, num_classes = 2).to(device)
        optimizer.zero_grad()
        loss, correct, out, z, y_hat = model(x, x_gt, sex_hot, m_type = "train")

        kld = z[0].mean()
        rec_loss = z[1].mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.cpu().detach().numpy()
        total_kld += kld.cpu().detach().numpy()
        total_rec_loss += rec_loss.cpu().detach().numpy()


        batch_size = x.num_graphs

        total += batch_size
        total_correct += correct
        recon_mesh = out.cpu() * std + mean
        s = s.unsqueeze(1)


        recon_mesh = torch.bmm(recon_mesh * s, R) + m  #procrust
        recon_mesh = recon_mesh.detach().cpu().numpy()
       
        gt_mesh = gt_mesh.detach().numpy()
        diff = euclidean_distances(recon_mesh, gt_mesh).mean()
        error_ += diff

        if np.max(diff) > max_error:
            max_error = np.max(diff)
            idx = np.where(diff==max_error)


            max_error_dict = {filename[idx[0][0]]:max_error}
    


    return total_loss / len_dataset, total_kld/len_dataset, total_rec_loss/len_dataset, error_/len_dataset, total_correct/total

def evaluate(n, model, test_loader, len_dataset, device,num_points, faces = None, checkpoint_dir = None, vis = False):
    model.eval()
    total_loss = 0
    total_rec_loss = 0
    count = 0
    total_kld = 0
    error = 0
    sigma = 0

    norm_dict = np.load(os.path.join(checkpoint_dir, 'norm.npz'), allow_pickle = True)
    mean = torch.FloatTensor(norm_dict['mean'])
    std = torch.FloatTensor(norm_dict['std'])
    error_ = np.empty((0, num_points))
    z_male = []
    z_female = []
    max_error = 0

    total = 0
    total_correct = 0

    acc = 0
    lamda = 0.01


    if vis:

        save_path = os.path.join(checkpoint_dir, "mesh"+str(n))    
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sucess_path = os.path.join(save_path, "sex_change_S")
        failed_path = os.path.join(save_path, "sex_change_F") 
        if not os.path.exists(sucess_path):
            os.makedirs(sucess_path)
        if not os.path.exists(failed_path):
            os.makedirs(failed_path)

    with torch.no_grad():
        for data in test_loader:
            x,x_gt, y, f, gt_mesh , R,m,s = data

            x, x_gt = x.to(device), x_gt.to(device)
            sex_hot = F.one_hot(y, num_classes = 2).to(device)
            loss, correct, out, z, y_hat = model(x, x_gt, sex_hot, m_type = "test")


            kld = z[0].mean()
            rec_loss = z[1].mean()
            #loss = kld + rec_loss 

            total_loss +=  loss.cpu().numpy()
            total_rec_loss += rec_loss.cpu().numpy()
            total_kld += kld.cpu().numpy()

            batch_size = x.num_graphs
            recon_mesh = out.cpu() * std + mean

            s = s.unsqueeze(1)

            recon_mesh = torch.bmm(recon_mesh * s, R) + m

            recon_mesh = recon_mesh.detach().cpu().numpy()



            gt_mesh = gt_mesh.detach().cpu().numpy()


            total += batch_size
            total_correct += correct.cpu().numpy()

            diff = euclidean_distances(recon_mesh, gt_mesh)
            #print(diff.shape)
            error_ = np.concatenate((error_, diff), axis = 0)
            oppo = 1 - sex_hot
            index_gt = torch.argmax(oppo,  dim = 1)

            z = z[2]

            oppo_x =  model.sample(oppo, z)

            index_pred = classifier_(model, oppo_x)

            acc += ((index_pred.squeeze() == index_gt.squeeze()).sum().item()/batch_size)

            oppo_mesh =  oppo_x.cpu() * std + mean

            oppo_mesh = torch.bmm(oppo_mesh * s, R) + m

            oppo_mesh = oppo_mesh.detach().cpu().numpy()

            if vis:
       
                for i in range(batch_size):
                    file = f[i].split('/')[-1]
                    file = file.split('.')[0]
                   # number = int(file[4:])

                    if index_pred[i] == index_gt[i]:
                
                        recon_path = os.path.join(sucess_path, file+'_recon'+'.obj')
                        save_obj(recon_path, recon_mesh[i], faces)
                        gt_path = os.path.join(sucess_path, file+'_gt'+'.obj')
                        save_obj(gt_path, gt_mesh[i], faces)

                        oppo_path = os.path.join(sucess_path, file+'.obj')
                        save_obj(oppo_path, oppo_mesh[i], faces)
                    else:
                    
                        recon_path = os.path.join(failed_path, file+'_recon'+'.obj')
                        save_obj(recon_path, recon_mesh[i], faces)
                        gt_path = os.path.join(failed_path, file+'_gt'+'.obj')
                        save_obj(gt_path, gt_mesh[i], faces)

                        oppo_path = os.path.join(failed_path, file+'.obj')
                        save_obj(oppo_path, oppo_mesh[i], faces)
                
             
    return total_loss/len_dataset, total_kld/len_dataset, total_rec_loss/len_dataset, total_correct/total, error_, acc/len_dataset, 

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape





def main(args):

    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)
    print(args.conf)
    config = read_config(args.conf)

    print('Initializing parameters')

    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cpu : device = 'cpu'
    print("Using device:",device)

    root_dir = config['root_dir']
    error_file = config['error_file']
    log_path = config['log_file']
    random_seeds = config['random_seeds']
    n_splits = config['folds']
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


    net = get_model(config, device)
    print('loading template...', config['template'])
    template_mesh = Mesh(filename=config['template'])
    template = np.array(template_mesh.v)
    faces = np.array(template_mesh.f)
    num_points = template.shape[0]


    #criterion = BCEFocalLoss()

    checkpoint_file = config['checkpoint_file']

    my_log = open(log_path, 'w')

    print('model type:', config['type'], file = my_log)
    print('optimizer type', opt, file = my_log)
    print('learning rate:', lr, file = my_log)

	
    torch.manual_seed(random_seeds)
    np.random.seed(random_seeds)
    random.seed(random_seeds)

    start_epoch = 1
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #To find if this is fixed in pytorch
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    labels = {}
    dataset_index = []
    files = sorted( os.listdir(root_dir) )
    for name in files:
        if not name.endswith(".obj") : continue
        name_ = name.split("_")
        dataset_index.append(name)
        if name_[1] == "f":
            labels[name] = 0
        else:
            labels[name] = 1

    for i in range(1):

        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state = random_seeds)

        n = 0
        y = np.ones(len(dataset_index))

        for train_index, test_index in skf.split(dataset_index, y):
            train_, valid_index = train_test_split(np.array(dataset_index)[train_index], test_size=test_size, random_state = random_seeds)
            history = []
            net.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'initial_weight.pt')))
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
            n+=1

            if args.train:
                train_dataset = CTimageData(root_dir, train_, config, labels, dtype = 'train', template = template, pre_transform = Normalize())
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

                valid_dataset = CTimageData(root_dir, valid_index, config, labels, dtype = 'test', template = template, pre_transform = Normalize())
                valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                best_loss = 10000000

                for epoch in range(start_epoch, total_epochs + 1):

                    begin = time.time()

                    if epoch > 500:
                        for p in optimizer.param_groups:
                            p['lr'] = 0.0001
                    elif epoch > 1000:
                        for p in optimizer.param_groups:
                            p['lr'] = 0.00005
                    train_loss, train_kld, train_rec_loss, train_error, train_acc = train(net, train_loader, len(train_loader), optimizer, device, num_points,checkpoint_dir = checkpoint_dir)

                    valid_loss, valid_kld, valid_rec_loss, valid_acc, error, acc  = evaluate(n, net, valid_loader, len(valid_loader),device, num_points,checkpoint_dir = checkpoint_dir)

                    duration = time.time() - begin

                    if valid_loss <= best_loss:
                        save_model(net, optimizer, n, train_loss, valid_loss, checkpoint_dir)
                        best_loss = valid_loss

                    history.append( {
                        "epoch" : epoch,
                        "begin" : begin,
                        "duration" : duration,
                        "training" : {
                            "loss" : train_loss,
                            "kld" : train_kld,
                            "reconstruction_loss" : train_rec_loss,
                            "accuracy" : train_acc.item(),
                            "error" : np.mean(train_error)
                        },
                        "validation" : {
                            "loss" : valid_loss,
                            "kld" : valid_kld,
                            "reconstruction_loss" : valid_rec_loss,
                            "accuracy" : float( str( acc ) ),
                            "error" : np.mean(error)
                        }
                    } )

                    if epoch%10 == 0:
                        toPrint = 'Epoch {}, train loss {}(kld {}, recon loss {}, train acc {}) || valid loss {}(error {}, rec_loss {}, valid acc {}, sex change acc {})'
                        print(toPrint.format(epoch, train_loss,train_kld, train_rec_loss, train_acc, valid_loss, np.mean(error), valid_rec_loss, valid_acc, acc))
                        print(toPrint.format(epoch, train_loss,train_kld, train_rec_loss, train_acc, valid_loss, np.mean(error), valid_rec_loss, valid_acc, acc), file = my_log)

                with open(os.path.join(checkpoint_dir, 'history' + str( n ) + '.json'), 'w') as fp:
                    json.dump(history, fp)

            if args.test:
                test_dataset = CTimageData(root_dir, np.array(dataset_index)[test_index], config, labels, dtype = 'test', template = template, pre_transform = Normalize())
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_'+ str(n)+'.pt')
                checkpoint = torch.load(checkpoint_file)
                net.load_state_dict(checkpoint['state_dict'])

                test_loss, test_kld, test_rec_loss, cls_acc, test_error, acc = evaluate(n, net, test_loader, len(test_loader),device, num_points, faces = faces, checkpoint_dir = checkpoint_dir, vis = args.vis)
                print(test_error.shape)
                print('round ', n,'test loss ', test_loss, 'mean error:', np.mean(test_error), "train sigma", np.std(test_error), "classification acc", cls_acc, "sex change rate", acc)
                print('round ', n,'test loss ', test_loss, 'mean error:', np.mean(test_error), "train sigma", np.std(test_error), "classification acc", cls_acc, "sex change rate", acc, file = my_log)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-t', '--train',action='store_true')
    parser.add_argument('-s', '--test',action='store_true')
    parser.add_argument('--cpu',action='store_true', help = "force cpu")
    parser.add_argument('-v', '--vis',action='store_true', help = "save transformed meshes")

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), './files/default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)
    main(args)
