import copy
import mesh_operations
import numpy as np
import os
from psbody.mesh import Mesh
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils import procrustes

# this function loads a mesh and reorders its vertices the same way open3d does
def Mesh2( filename = "none" ):
    mesh = Mesh( filename=filename )
    indices = np.full( shape = mesh.v.shape[ 0 ], fill_value= -1 )
    coords = np.copy( mesh.v )
    counter = 0
    for i in range( mesh.f.shape[ 0 ] ) :
        for j in range( 3 ) :
            vertex = mesh.f[ i ][ j ]
            if indices[ vertex ] < 0:
                indices[ vertex ] = counter
                coords[ counter ] = mesh.v[ vertex ]
                counter += 1
            mesh.f[ i ][ j ] = indices[ vertex ]

    mesh.v = coords
    return mesh

def save_obj(filename, vertices, faces):
    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def OnUnitCube(data):
    max_, _ = torch.max(data.x, dim = 0)  # [N, D]  =>  [1, D]
    min_, _ = torch.min(data.x, dim = 0) 
    c = max_ - min_
    s = torch.max(c)

    data.x  = data.x / s 
    m, _ = torch.min(data.x, dim = 0, keepdims = True)
    data.x = data.x - m
    return data, s, m

def listMeshes( config, getSexFromFileName = True ) :
    labels = {}
    dataset_index = []
    files = sorted( os.listdir( config[ "root_dir" ] ) )

    toRemove = {}
    error_file = config[ "error_file" ]
    if len( error_file ) > 0:
        with open( error_file ) as f:
            lines = f.read().split( "\n" )
            for line in lines:
                toRemove[ line.split( " " )[ 0 ] ] = True

    numberOfMeshes = 0
    numberOfRejectedMeshes = 0

    for name in files:
        if not name.endswith( ".obj" ) : continue
        numberOfMeshes += 1
        if name.split( "/" ).pop() in toRemove:
            numberOfRejectedMeshes += 1
            continue
        dataset_index.append( name )

        if getSexFromFileName:
            name_ = name.split( "_" )
            if name_[ 1 ] == "f":
                labels[ name ] = 0
            else:
                labels[ name ] = 1
        else: labels[ name ] = -1

    s = "Dataset : {} meshes, {} rejected meshes, {} remaining meshes"
    print( s.format( numberOfMeshes, numberOfRejectedMeshes, len( dataset_index ) ) )
    return dataset_index, labels


class MeshData(Dataset):
    def __init__(self, dataset_index , config, label,  template, dtype = 'train',   pre_transform = None):
        checkpoint_dir = config['checkpoint_dir']
        self.pre_transform = pre_transform
        self.filename = []
        self.data_label = []
        self.ori_mesh = []
        self.R = []
        self.s = []
        self.m = []
        self.ori_data = []
        self.edge_index = None
  
        for fileName in dataset_index:
            file = os.path.join(config[ 'root_dir' ], fileName )
            if not os.path.exists(file) : continue
            self.filename.append(file)
            mesh = Mesh(filename=file)
            # points, s, mean_points = OnUnitCube(np.array(mesh.v))
            points = np.array(mesh.v) 
            self.ori_mesh.append(torch.Tensor(points))
            mtx1, mtx2, disparity, res= procrustes(template,points)
            ori_mtx = copy.copy(mtx2)
            self.ori_data.append(ori_mtx)
            self.data_label.append(label[fileName])
            self.R.append(torch.FloatTensor(res[0]))
            self.s.append(torch.FloatTensor([res[1]]))
            self.m.append(torch.FloatTensor(np.array([res[2]])))
            if self.edge_index is None:
                adjacency = mesh_operations.get_vert_connectivity(mesh.v, mesh.f).tocoo()
                self.edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col))).long()

        if dtype == 'train' and not os.path.exists(os.path.join(checkpoint_dir,'norm')):
            mean_train = np.mean(self.ori_data, axis=0)
            std_train = np.std(self.ori_data, axis=0)
            self.norm_dict = {'mean': mean_train, 'std': std_train}
            np.savez(os.path.join(checkpoint_dir,'norm'), mean = mean_train, std = std_train)

        if pre_transform is not None:
            self.norm_dict = np.load(os.path.join(checkpoint_dir, 'norm.npz'), allow_pickle = True)
            mean = self.norm_dict['mean']
            std = self.norm_dict['std']
            if hasattr(pre_transform, 'mean') and hasattr(pre_transform, 'std'):
                if pre_transform.mean is None:
                    pre_transform.mean = mean
                if pre_transform.std is None:
                    pre_transform.std = std

        print( dtype, " dataset has been created, number of {} samples:".format(dtype), len(self.ori_data) )

    def __len__(self):
        return len( self.ori_data )

    def __getitem__(self, idx):
        normalized_data = (torch.tensor(self.ori_data[idx])-self.pre_transform.mean)/ self.pre_transform.std
        mesh_verts = copy.copy(normalized_data).float()
        data_ = Data(x=mesh_verts, y=mesh_verts, edge_index=self.edge_index)
        return data_, normalized_data , self.data_label[idx], self.filename[idx],self.ori_mesh[idx], self.R[idx], self.m[idx], self.s[idx]

