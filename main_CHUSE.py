"""
Created on Mon Jan 04 09:43:10 2021

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
#from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config


from data import CTimageData

from transform import Normalize
from utils import *
from psbody.mesh import Mesh
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
import open3d as o3d
from models.cheb_VAE import cheb_VAE
from models.cheb_cls import cheb_GCN
import time

from sklearn.metrics.pairwise import euclidean_distances
EPS = 1e-9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:",device)
def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor
def save_obj(filename, vertices, faces):
    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))



def adjust_learning_rate(optimizer, lr_decay):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(coma, optimizer, n, train_loss, val_loss, checkpoint_dir, epoch):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(n)+'.pt'))


def train(model, train_loader, len_dataset, optimizer, label_mask, device, num_points, config, checkpoint_dir = None):
    model.train()
    total_loss = 0
    total_rec_loss = 0

    total_kld = 0
    euc_dist = 0
    norm_dict = np.load(os.path.join(checkpoint_dir, 'norm.npz'), allow_pickle = True)
    mean = torch.FloatTensor(norm_dict['mean'])
    std = torch.FloatTensor(norm_dict['std'])
    error_ = np.empty((0, num_points, 3))
    count = 0
    correct_count = 0
    lamda = 0.001
    for b, data in enumerate(train_loader):
        x, label, gt_mesh , R,m,s  = data
        batch_size = x.num_graphs

        count += batch_size

        x = x.to(device)

        labels_onehot = torch.zeros(label.shape[0], config["num_classes"])
        
        labels_onehot.scatter_(1, label.type(torch.LongTensor).unsqueeze(1), 1)
        
        labels_onehot = torch.clamp(labels_onehot, EPS, 1-EPS).to(device)
        optimizer.zero_grad()

        loss, correct, out, z, _ = model(x,  labels_onehot, m_type = "train")

        # for params in model.parameters():
        #     loss += lamda * torch.sum(abs(params))
        loss.backward()
        optimizer.step()

        total_loss += loss.cpu().detach().numpy()


        correct_count += correct.cpu().detach().numpy()

        recon_mesh = out.detach().cpu()
        s = s.unsqueeze(1)

        recon_mesh = recon_mesh * std + mean

        recon_mesh = torch.bmm(recon_mesh * s, R) + m

        euc_dist += euclidean_distances(gt_mesh, recon_mesh).item()
        
        recon_mesh = recon_mesh.numpy()

        gt_mesh = gt_mesh.detach().numpy()

        error_ = np.concatenate((error_, np.abs(recon_mesh - gt_mesh)), axis = 0)

    return total_loss / len_dataset, euc_dist/len_dataset, correct_count/count

def save_obj(filename, vertices, faces):
    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def euclidean_distances(gt, pred):
    return torch.sqrt(((gt-pred)**2).sum(-1)).mean()

def evaluate(model, test_loader, len_dataset, device, config, faces = None,checkpoint_dir = None, vis = False):
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


    total = 0
    correct_count = 0
    count = 0
    acc = 0
    euc_dist = 0
    with torch.no_grad():
        for data in test_loader:
            x, label,  gt_mesh, R,m,s = data
            batch_size = x.num_graphs

            total += batch_size

            x = x.to(device)

            labels_onehot = torch.zeros(label.shape[0], config["num_classes"])
            
            labels_onehot.scatter_(1, label.type(torch.LongTensor).unsqueeze(1), 1)
            
            labels_onehot = torch.clamp(labels_onehot, EPS, 1-EPS).to(device)


            loss, correct, out, z, _ = model(x, labels_onehot)


            total_loss += loss.cpu().detach().numpy()

        
            correct_count += correct.cpu().detach().numpy()


            recon_mesh = out.detach().cpu()

            s = s.unsqueeze(1) 

            recon_mesh = recon_mesh * std + mean

            recon_mesh = torch.bmm(recon_mesh * s, R) + m

            
            euc_dist += euclidean_distances(gt_mesh, recon_mesh).item()



    return total_loss/len_dataset,euc_dist/len_dataset, correct_count/total

def main(args):

    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)

    print('Initializing parameters')

 
    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    root_dir = config['root_dir']
    nb_patients = config['nb_patient']

    label_file = config['label_file']
    error_file = config['error_file']
    log_path = config['log_file']
    random_seeds = config['random_seeds']

    test_size = config['test_size']
    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    template_file_path = config['template']
    val_losses, accs, durations = [], [], []


    
    print('loading template...', config['template'])
    template_mesh = Mesh(filename=config['template'])#o3d.io.read_triangle_mesh(config['template'])

    template = np.array(template_mesh.v)
    faces = np.array(template_mesh.f)

   # template_mesh = Mesh(v=template_mesh.vertices, f=template_mesh.triangles)
    print(template.shape)
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])
    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]


    num_feature = template_mesh.v.shape[1]
    net = cheb_VAE(num_feature, config, D_t, U_t, A_t, num_nodes, model = config['model']).to(device)


    torch.save(net.state_dict(), os.path.join(config['checkpoint_dir'], 'initial_weight.pt')) #save initialization for cross validation


    criterion = nn.MSELoss()
    num_points = template.shape[0]

    checkpoint_file = config['checkpoint_file']

    my_log = open(log_path, 'w')

    print('model type:', config['type'], file = my_log)
    print('optimizer type', opt, file = my_log)
    print('learning rate:', lr, file = my_log)


    start_epoch = 1
    print(checkpoint_file)

    dataset_index = []
    labels = []
 

    import glob
    man_list = glob.glob("./fake_dataset/man/*.obj")
    woman_list = glob.glob("./fake_dataset/woman/*.obj")

    for name in man_list:
        dataset_index.append(name)
        labels.append(0)

    for name in woman_list:
        dataset_index.append(name)
        labels.append(1)


    best_error =np.zeros(5)
    best_acc =np.zeros(5)
    best_loss = np.ones(5) * 10000000
    


    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state = random_seeds)  # 5-folds repeated 10 times  

    n = 0

    y = np.array(labels)
    me = 0

    test_error_ = 0
    test_acc_ = 0
    train_error_ = 0

    for train_index, test_index in skf.split(dataset_index, y):
        train_, valid_ = train_test_split(np.array(dataset_index)[train_index], test_size=test_size, random_state = random_seeds)
        if args.train:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
            n+=1
            net.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'initial_weight.pt')))
            if checkpoint_file:
                print("loading weight from ", checkpoint_file[n-1])
                checkpoint = torch.load(checkpoint_file[n-1])

                print('start_epoch', start_epoch)
                net.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            

            train_dataset = CTimageData(train_, config, labels, dtype = 'train', template = template, pre_transform = Normalize())
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
            valid_dataset = CTimageData( valid_, config, labels,  dtype = 'test', template = template, pre_transform = Normalize())
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


            label_mask = {}

            
            for epoch in range(start_epoch, total_epochs+1):

                if epoch > 300:
                    for p in optimizer.param_groups:
                        p['lr'] = 0.0003
                elif epoch > 600:
                    for p in optimizer.param_groups:
                        p['lr'] = 0.0001
              
                train_loss, train_error, train_acc = train(net, train_loader, len(train_loader), optimizer,label_mask, device, num_points, config = config,checkpoint_dir = checkpoint_dir)
                valid_loss, valid_error, valid_acc = evaluate(net, valid_loader, len(valid_loader),device,faces = faces, config = config, checkpoint_dir = checkpoint_dir)

                if valid_loss < best_loss[n-1]:
                    best_loss[n-1] = valid_loss
                    best_error[n-1] = valid_error
                    best_acc[n-1] = valid_acc

                    save_model(net, optimizer, n, train_loss, valid_loss, checkpoint_dir, epoch)
                
            print('Epoch {}, train loss {} train_error {}, train_acc {},  valid loss {} valid_error {}, valid_acc {}'.format(epoch, train_loss,train_error, train_acc, valid_loss, valid_error, valid_acc))
            print('Epoch {}, train loss {} train_error {}, train_acc {},  valid loss {} valid_error {}, valid_acc {}'.format(epoch, train_loss,train_error, train_acc, valid_loss, valid_error, valid_acc), file = my_log)

        print("testing .....")

        test_dataset = CTimageData(np.array(dataset_index)[test_index], config, labels, dtype = 'test', template = template, pre_transform = Normalize())  
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        state_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint_'+ str(n)+'.pt'))
        net.load_state_dict(state_dict['state_dict'])

        test_loss, test_error, test_acc = evaluate(net, test_loader, len(test_loader),device,faces = faces, config = config, checkpoint_dir = checkpoint_dir)

        print('Round {}, test loss {} test_error {}, test_acc {} '.format(n, test_loss, test_error, test_acc))


def calculate_dist(gt, recon):
    return (torch.sqrt(((gt - recon)**2).sum(-1))).mean()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-t', '--train',type = bool, default= False, help='Is train?')
    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), './files/default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)
    acc = 0


    main(args)

