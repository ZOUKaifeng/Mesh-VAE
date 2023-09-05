import os
import torch
import pandas as pd
import numpy as np
#from scipy.spatial import procrustes
# from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from transform import Normalize
from torch.utils.data import Dataset
import mesh_operations
from psbody.mesh import Mesh
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from utils import procrustes
import copy

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
    '''
    root_dir: point cloud data path
    error_file: outlier list
    template: average point cloud
    '''
    def __init__(self, dataset_index , config, label,  template, dtype = 'train',   pre_transform = None):
        self.checkpoint_dir = config['checkpoint_dir']
        self.root_dir = config[ 'root_dir' ]
        self.error_file = config['error_file']
        self.label = label
        self.template = template  
        self.dataset_index = dataset_index
        self.pre_transform = pre_transform
        # self.random_state = config['random_seeds']
        self.dtype = dtype
        self.res = []
        self.preprocess()
        self.male = None
        self.female = None


    def __len__(self):

        return len(self.ori_data)   

    def __getitem__(self, idx):
        mtx2 = copy.copy(self.ori_data[idx])
        index = []
        
        ori_data = (torch.tensor(self.ori_data[idx])-self.pre_transform.mean)/ self.pre_transform.std
        mesh_verts = copy.copy(ori_data).float()
       # mesh_verts[index] = 0
        data_ = Data(x=mesh_verts, y=mesh_verts, edge_index=self.edge_index)
        return data_, ori_data , self.data_label[idx], self.filename[idx],self.ori_mesh[idx], self.R[idx], self.m[idx], self.s[idx]


    def preprocess(self):
        filename = []
        data = []  #Create an empty list
        data_label = []
        train_vertices = []

        self.ori_mesh = []
        self.R = []
        self.s = []
        self.m = []
        self.unit_s = []
        self.unit_min = []
        self.data = []
        self.ori_data = []
        self.edge_index = None
  
        for i in self.dataset_index:
            file = os.path.join(self.root_dir, i )
            if os.path.exists(file):   #i not in error and 
               # print(file)
                
                filename.append(file)
               # print(file)
               # points = pd.read_csv(file, header=None).values  # Load the points of the patient i
                # mesh = o3d.io.read_triangle_mesh(file)

                mesh = Mesh(filename=file)
                # points, s, mean_points = OnUnitCube(np.array(mesh.v))
                points = np.array(mesh.v) 
                self.ori_mesh.append(torch.Tensor(points))
                mtx1, mtx2, disparity, res= procrustes(self.template,points)
                ori_mtx = copy.copy(mtx2)
                train_vertices.append(ori_mtx)
                if self.edge_index is None:

                    adjacency = mesh_operations.get_vert_connectivity(mesh.v, mesh.f).tocoo()
                    self.edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col))).long()
               # data_ = Data(x=mesh_verts, y=mesh_verts, edge_index=self.edge_index)

                

                   # Procrustes surimposition of the patients points over the average points (and normalization)
                #data.append(data_)        # Add the registered points to the tensor vertices
                data_label.append(self.label[i])  # Add label i to label
               
                # self.res.append([torch.Tensor(res[0]),torch.Tensor([res[1]]), torch.Tensor([res[2]])])
                self.R.append(torch.FloatTensor(res[0]))
                self.s.append(torch.FloatTensor([res[1]]))
                self.m.append(torch.FloatTensor(np.array([res[2]])))



        if not os.path.exists(os.path.join(self.checkpoint_dir,'norm')):
            if self.dtype == 'train':

                mean_train = np.mean(train_vertices, axis=0)
                std_train = np.std(train_vertices, axis=0)

                self.norm_dict = {'mean': mean_train, 'std': std_train}
                np.savez(os.path.join(self.checkpoint_dir,'norm'), mean = mean_train, std = std_train)


        if self.pre_transform is not None:
            self.norm_dict = np.load(os.path.join(self.checkpoint_dir, 'norm.npz'), allow_pickle = True)
            mean = self.norm_dict['mean']
            std = self.norm_dict['std']
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_transform.mean is None:
                    self.pre_transform.mean = mean
                if self.pre_transform.std is None:
                    self.pre_transform.std = std

            #self.data = data
           # for v in train_vertices:
            self.ori_data =train_vertices

            self.filename = filename
            self.data_label = data_label


        else:
            self.data = data
            self.data_label = data_label

            self.filename = filename

        print(self.dtype ," dataset has been created, number of {} samples:".format(self.dtype), len(self.ori_data) )

        # if self.dtype == 'test':
        #     if self.pre_transform is not None:
        #         self.norm_dict = np.load(os.path.join(self.checkpoint_dir, 'norm.npz'), allow_pickle = True)
        #         mean = self.norm_dict['mean']
        #         std = self.norm_dict['std']
        #         if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
        #             if self.pre_transform.mean is None:
        #                 self.pre_transform.mean = mean
        #             if self.pre_transform.std is None:
        #                 self.pre_transform.std = std

        #         self.test = [self.pre_transform(v) for v in data]
        #         self.test_label = data_label
        #         self.filename = filename
        #     else:
        #         self.test = data
        #         self.test_label = data_label
        #         self.filename = filename
        
            # print("Test dataset has been created, the number of test data:", len(self.test))


if __name__ == '__main__':
    root_dir = "./transfo_points"
    dataset_index = list(range(100))
    template = np.array(pd.read_csv("./template/final_points.csv.gz", header=None).values)
    dataset = MeshData(dataset_index, dtype = 'test', template = template)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    for i in dataloader:
        print(i[0].shape)
