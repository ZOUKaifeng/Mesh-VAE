
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
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
from tqdm import tqdm
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:",device)

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



def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape


def inference(root_dir, net, output_path, mean, std, config, template, batch_size, faces):

    labels = {}
    dataset_index = []
    files = os.listdir(root_dir)
    for name in files:
        if not name.endswith(".obj") : continue
        print( name )
        name_ = name.split("_")
#        if name_[-1] != "box.json":
      #  number = int(name_[0])
         
        dataset_index.append(name)
        labels[name] = -1


    pred_sex = {}
    error_dict = {}

    n = 1
    dataset = CTimageData(root_dir, dataset_index, config, labels, dtype = 'test', template = template, pre_transform = Normalize())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    sucess_path = os.path.join(output_path, "sex_change")


    if not os.path.exists(sucess_path):
        os.makedirs(sucess_path)


    with torch.no_grad():
        for data in tqdm(data_loader):
            x,x_gt, y, f, gt_mesh , R,m,s = data

            x, x_gt = x.to(device), x_gt.to(device)
            batch_size = x.num_graphs
            x_gt = x_gt.reshape(batch_size, -1, 3).float()
            pred = classifier_(net, x_gt)


            for i in range(x_gt.shape[0]):

                pred_sex.update({ f[i]:str(pred[i].cpu().numpy())})

            sex_hot = F.one_hot(pred, num_classes = 2).to(device)
            loss, correct, out, z, y_hat = net(x, x_gt, sex_hot, m_type = "test")

            
            recon_mesh = out.cpu() * std + mean

            s = s.unsqueeze(1)

            recon_mesh = torch.bmm(recon_mesh * s, R) + m

            recon_mesh = recon_mesh.detach().cpu().numpy()

            gt_mesh = gt_mesh.detach().cpu().numpy()

            oppo = 1 - sex_hot

            z = z[2]

            oppo_x =  net.sample(oppo, z)



            oppo_mesh =  oppo_x.cpu() * std + mean

            oppo_mesh = torch.bmm(oppo_mesh * s, R) + m

            oppo_mesh = oppo_mesh.detach().cpu().numpy()

		   
			
            diff = euclidean_distances(recon_mesh, gt_mesh).mean(-1)

            for i in range(diff.shape[0]):
                error_dict.update({f[i]:format(diff[i], '.4f')})



            for i in range(batch_size):
                file = f[i].split('/')[-1]
                file = file.split('.')[0]

         
                recon_path = os.path.join(sucess_path, file+'_recon'+'.obj')
                save_obj(recon_path, recon_mesh[i], faces)
                gt_path = os.path.join(sucess_path, file+'_gt'+'.obj')
                save_obj(gt_path, gt_mesh[i], faces)

                oppo_path = os.path.join(sucess_path, file+'.obj')
                save_obj(oppo_path, oppo_mesh[i], faces)


    import json
    with open(os.path.join(output_path, 'pred_{}.json'.format(n)), 'w') as fp:
        json.dump(pred_sex, fp)

    with open(os.path.join(output_path, 'error_list_{}.json'.format(n)), 'w') as fp:
        json.dump(error_dict, fp)

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






    root_dir = args.data_dir
    output_path = args.output_path

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


    net = get_model(config, device)
    print('loading template...', config['template'])
    template_mesh = Mesh(filename=config['template'])
    template = np.array(template_mesh.v)
    faces = np.array(template_mesh.f)
    num_points = template.shape[0]


    #criterion = BCEFocalLoss()

    checkpoint_file = config['checkpoint_file']

    n = args.model
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_'+ str(n)+'.pt')
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['state_dict'])
    norm_dict = np.load(os.path.join(checkpoint_dir, 'norm.npz'), allow_pickle = True)
    mean = torch.FloatTensor(norm_dict['mean'])
    std = torch.FloatTensor(norm_dict['std'])
    with torch.no_grad():
        inference(root_dir, net, output_path, mean, std, config, template, batch_size, faces)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer')
    parser.add_argument('-u', '--conf', help='path of config file')
    parser.add_argument('-o', '--output_path',type = str, default= " ")
    parser.add_argument('-d', '--data_dir',type = str, default= " ")
    parser.add_argument('-n', '--model',type = int, default= 1)




    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), './files/default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)