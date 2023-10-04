import argparse
from config_parser import read_config
from data import MeshData, listMeshes, save_obj
import json
from model import get_model, classifier_
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transform import Normalize
from utils import euclidean_distances
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:",device)

def inference(loader, net, output_path, mean, std, config, faces, args):
    results = {}
    net.eval()
    if not os.path.exists(output_path): os.makedirs(output_path)
    mesh_path = os.path.join(output_path, "sex_change" )
    if args.meshes and not os.path.exists(mesh_path): os.makedirs(mesh_path)

    with torch.no_grad():
        d = tqdm(loader)
        for data in d:
            x,x_gt, y, f, gt_mesh , R,m,s = data
            x, x_gt = x.to(device), x_gt.to(device)
            local_batch_size = x.num_graphs
            x_gt = x_gt.reshape(local_batch_size, -1, 3).float()
            pred = classifier_(net, x_gt)

            for i in range(x_gt.shape[0]):
                predicted_sex = pred[i].cpu().numpy()
                results[ f[ i ].split( "/" ).pop() ] = { "sex" : int( str( predicted_sex ) ) }

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
            maxDiff = euclidean_distances(recon_mesh, gt_mesh).max(-1)

            for i in range(diff.shape[0]):
                results[ f[ i ].split( "/" ).pop() ][ "reconstruction_error" ] = { "mean" : float( str ( diff[ i ] ) ), "max" : float( str ( maxDiff[ i ] ) ) }

            if not args.meshes : continue
            for i in range(local_batch_size):
                file = f[i].split('/')[-1]
                file = file.split('.')[0]
                recon_path = os.path.join(mesh_path, file+'_recon'+'.obj')
                save_obj(recon_path, recon_mesh[i], faces)
                gt_path = os.path.join(mesh_path, file+'_gt'+'.obj')
                save_obj(gt_path, gt_mesh[i], faces)
                oppo_path = os.path.join(mesh_path, file+'.obj')
                save_obj(oppo_path, oppo_mesh[i], faces)

    d.close()

    with open(os.path.join(output_path, 'inference.json'), 'w') as fp:
        json.dump(results, fp)

def main(args):
    assert os.path.exists( args.conf ), 'Config not found' + args.conf
    print(args.conf)
    config = read_config(args.conf)

    for option in args.parameter:
        value = option[ 1 ]
        if not isinstance( config[ option[ 0 ] ], str ) :
            value = json.loads( value )
        config[ option[ 0 ] ] = value

    print('Initializing parameters')

    checkpoint_dir = os.path.join( os.path.dirname( args.conf ), config['checkpoint_dir'] )
    config['checkpoint_dir'] = checkpoint_dir
    if not os.path.exists(checkpoint_dir) : os.makedirs(checkpoint_dir)

    config[ 'root_dir' ] = args.data_dir
    batch_size = config['batch_size']
    print( 'loading template...', config[ 'template' ] )
    net, template_mesh = get_model( config, device )
    template = np.array( template_mesh.v )
    faces = np.array( template_mesh.f )
    norm_dict = np.load(os.path.join(checkpoint_dir, 'norm.npz'), allow_pickle = True)
    mean = torch.FloatTensor( norm_dict[ 'mean' ] )
    std = torch.FloatTensor( norm_dict[ 'std' ] )

    if args.all : models = range( 1, 1 + config[ "folds" ] )
    else : models = [ args.model ]

    dataset_index, labels = listMeshes( config, False )
    dataset = MeshData(dataset_index, config, labels, dtype = 'test', template = template, pre_transform = Normalize())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i in models:
        checkpoint_file = os.path.join( checkpoint_dir, 'checkpoint_'+ str(i)+'.pt' )
        print( "Load checkpoint file : ", checkpoint_file )
        checkpoint = torch.load( checkpoint_file )
        net.load_state_dict( checkpoint['state_dict'] )
        path = os.path.join( args.output_path, str( i ) )
        inference(loader, net, path, mean, std, config, faces, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Trainer')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument( "-p", "--parameter", metavar=('parameter', 'value'), action='append', nargs=2, help = "config parameters", default = [] )
    parser.add_argument('-o', '--output_path',type = str, default= "./")
    parser.add_argument('-d', '--data_dir',type = str, default= " ")
    parser.add_argument('-m', '--meshes',action='store_true', help = "save meshes")
    parser.add_argument('-a', '--all',action='store_true', help = "inference for all folds")
    parser.add_argument('-n', '--model',type = int, default= 1)
    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), './files/default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
