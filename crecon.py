"""
Created on Mon Oct 05 13:43:10 2020

@Author: Kaifeng

@Contact: kaifeng.zou@unistra.fr

main function 
"""
import argparse
from config_parser import read_config
import json
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric
import mesh_operations
import plotLosses
from data import MeshData, listMeshes, save_obj
from model import get_model, classifier_, save_model
from transform import Normalize
from utils import *
from psbody.mesh import Mesh
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import copy
import time
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dvae, train_loader, optimizer, device, criterion):
    model.train()
    dvae.eval()
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
        batch_size = label.shape[0]
        total_loss += loss.cpu().detach().numpy() * batch_size
        predicted = torch.argmax(F.softmax(pred, 1), dim = -1)
        total += batch_size
        correct += (predicted == label).sum().item()

    return total_loss / total, correct / total

def evaluate(model, dvae, test_loader, device, criterion, err_file = False):
    model.eval()
    dvae.eval()
    total_loss = 0
    total = 0
    correct = 0
    err = {}

    with torch.no_grad():
        for data in test_loader:
            x,x_gt, label, f, gt_mesh , R,m,s = data
            x_gt, label = x_gt.to(device).float(), label.to(device)
            diff, _ = estimate_diff(dvae, x_gt, label, "test")
            pred = model(diff)
            batch_size = label.shape[0]
            loss = criterion(pred, label)
            total_loss +=  loss.cpu().numpy() * batch_size
            predicted =torch.argmax(F.softmax(pred, 1), dim = -1)
            total += batch_size
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

    return total_loss / total, correct/total, err

def inference(model, dvae, loader, output_path):
    model.eval()
    dvae.eval()
    results = {}
    if not os.path.exists(output_path): os.makedirs(output_path)

    with torch.no_grad():
        d = tqdm(loader)
        for data in d:
            x,x_gt, label, f, gt_mesh , R,m,s = data
            x_gt, label = x_gt.to(device).float(), label.to(device)
            diff, _ = estimate_diff(dvae, x_gt, label, "test")
            pred = model(diff)
            predicted =torch.argmax(F.softmax(pred, 1), dim = -1)

            for i in range(x_gt.shape[0]):
                predicted_sex = predicted[i].cpu().numpy()
                results[ f[ i ].split( "/" ).pop() ] = { "sex" : int( str( predicted_sex ) ) }

    with open(os.path.join(output_path, 'inference.json'), 'w') as fp:
        json.dump(results, fp)

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

    config[ 'checkpoint_dir' ] = os.path.join( os.path.dirname( args.conf ), config['checkpoint_dir'] )

    print('Initializing parameters')

    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cpu : device = 'cpu'
    print("Using device:",device)

    random_seeds = config['random_seeds']
    torch_geometric.seed_everything(random_seeds)
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    opt = config['optimizer']
    batch_size = config['batch_size']

    dvae, template_mesh = get_model(config, device, model_type="cheb_VAE", save_init = False)
    template = np.array(template_mesh.v)
    faces = np.array(template_mesh.f)

    checkpoint_file = config['checkpoint_file']
    print("loading checkpoint for DVAE from ", checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    dvae.load_state_dict(checkpoint['state_dict'])   


    if args.inferenceDir:
        config[ "root_dir" ] = args.inferenceDir
        dataset_index, labels = listMeshes( config )

        if args.all : models = range( 1, 1 + config[ "folds" ] )
        else : models = [ args.model ]
        net, _unused = get_model(config, device, model_type="cheb_GCN")
        inference_dataset = MeshData(dataset_index, config, labels, dtype = 'test', template = template, pre_transform = Normalize())
        inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

        for i in models:
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_'+ str(i)+'.pt')
            checkpoint = torch.load(checkpoint_file)
            net.load_state_dict(checkpoint['state_dict'])
            path = os.path.join( args.output_path, str( i ) )
            inference( net, dvae, inference_loader, path )
        exit( 0 )

    dataset_index, labels = listMeshes( config )
    my_log = open(config['log_file'], 'w')

    print('model type:', config['type'], file = my_log)
    print('optimizer type', opt, file = my_log)
    print('learning rate:', lr, file = my_log)

    print(checkpoint_file)
    #criterion = BCEFocalLoss()
    criterion = torch.nn.CrossEntropyLoss()
    skf = RepeatedStratifiedKFold(n_splits=config['folds'], n_repeats=1, random_state = random_seeds)
    n = 0
    y = np.ones(len(dataset_index))

    for train_index, test_index in skf.split(dataset_index, y):
        train_, valid_index = train_test_split(np.array(dataset_index)[train_index], test_size=config['test_size'], random_state = random_seeds)

        history = []
        net, _unused = get_model(config, device, model_type="cheb_GCN")
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        n+=1

        if args.train:

            best_val_acc = 0
            train_dataset = MeshData(train_, config, labels, dtype = 'train', template = template, pre_transform = Normalize())
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            valid_dataset = MeshData(valid_index, config, labels, dtype = 'test', template = template, pre_transform = Normalize())
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(1, total_epochs + 1):
                begin = time.time()
                train_loss, train_acc = train(net, dvae, train_loader, optimizer, device, criterion)
                val_loss, valid_acc, _ = evaluate(net,dvae, valid_loader, device, criterion)

                if valid_acc >= best_val_acc:
                    save_model(net, optimizer, n, train_loss, val_loss, checkpoint_dir)
                    best_val_acc = valid_acc

                duration = time.time() - begin
                print('epoch ', epoch,' Train loss ', train_loss, 'train acc',train_acc, ' Val loss ', val_loss, 'acc ', valid_acc)
                print('epoch ', epoch,' Train loss ', train_loss, 'train acc',train_acc, ' Val loss ', val_loss, 'acc ', valid_acc, file = my_log)

                history.append( {
                    "epoch" : epoch,
                    "begin" : begin,
                    "duration" : duration,
                    "training" : {
                        "loss" : train_loss,
                        "accuracy" : train_acc
                    },
                    "validation" : {
                        "loss" : val_loss,
                        "accuracy" : valid_acc
                    }
                } )


        if args.test:
            if not args.train:
                checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_'+ str(n)+'.pt')
                checkpoint = torch.load(checkpoint_file)
                net.load_state_dict(checkpoint['state_dict'])
                history.append( {} )

            test_dataset = MeshData(np.array(dataset_index)[test_index], config, labels, dtype = 'test', template = template, pre_transform = Normalize())  
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_loss, test_acc, _ = evaluate(net, dvae, test_loader, device, criterion, err_file = False)

            print( 'test loss ', test_loss, 'test acc',test_acc)
            history[ -1 ][ "test" ] = {
                "loss" : test_loss,
                "accuracy" : test_acc
            }
        with open(os.path.join(checkpoint_dir, 'history' + str( n ) + '.json'), 'w') as fp:
            json.dump(history, fp)

        plt = plotLosses.plotLosses( "Fold " + str( n ), history, config )
        plt.savefig( os.path.join( checkpoint_dir, 'losses' + str( n ) + '.pdf') )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-t', '--train',action='store_true')
    parser.add_argument('-s', '--test',action='store_true')
    parser.add_argument('--cpu',action='store_true', help = "force cpu")
    parser.add_argument('-o', '--output_path', type = str, default= "./")
    parser.add_argument('-i', '--inferenceDir',type = str )
    parser.add_argument('-a', '--all',action='store_true', help = "inference for all folds")
    parser.add_argument('-n', '--model', help = 'number of inference',type = int, default= 1)
    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), './files/default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)
    main(args)
