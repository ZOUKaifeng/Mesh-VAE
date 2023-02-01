"""
!/home/miv/kzou/anaconda3/envs/CTimage python3

Created on Mon Oct 05 13:43:10 2020

@author: Kaifeng

Version 2.0 add torch geometric data

To do:
    transform dont have attribute normalize
"""

import os
import torch
import pandas as pd
import numpy as np
#from scipy.spatial import procrustes
# from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from transform import Normalize
from torch_geometric.data import Data, Dataset
import mesh_operations
from psbody.mesh import Mesh

import open3d as o3d
from sklearn.model_selection import train_test_split
from utils import procrustes
import copy
def OnUnitCube(data):
    max_, _ = torch.max(data.x, dim = 0)  # [N, D]  =>  [1, D]
    min_, _ = torch.min(data.x, dim = 0) 
    c = max_ - min_
    s = torch.max(c)

    data.x  = data.x / s 
    m, _ = torch.min(data.x, dim = 0, keepdims = True)
    data.x = data.x - m

    return data, s, m

class CTimageData(Dataset):
    '''
    root_dir: point cloud data path
    error_file: outlier list
    label_file: 
    template: average point cloud
    '''
    def __init__(self, dataset_index , config, label, template, dtype = 'train',  transform=None, pre_transform = None):
        self.checkpoint_dir = config['checkpoint_dir']

        self.transform = transform  
        self.error_file = config['error_file']
        self.label = label

        self.template = template  
        self.dataset_index = dataset_index
        self.pre_transform = pre_transform
        # self.random_state = config['random_seeds']
        # self.test_size = config['test_size']
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
        return data_ , self.data_label[idx],self.ori_mesh[idx], self.R[idx], self.m[idx], self.s[idx]


    def preprocess(self):

        filename = []

        data = []  
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
        self.age_label = []


        for i, file in enumerate(self.dataset_index):

            
            filename.append(file)


            mesh = Mesh(filename=file)

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
            self.R.append(torch.FloatTensor(res[0]))
            self.s.append(torch.FloatTensor([res[1]]))
            self.m.append(torch.FloatTensor([res[2]]))



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
            self.ori_data =train_vertices

            self.filename = filename
            self.data_label = data_label


        else:
            self.data = data
            self.data_label = data_label

            self.filename = filename

        print(self.dtype ," dataset has been created, the number of train data:", len(self.ori_data) )


if __name__ == '__main__':
    root_dir = "./transfo_points"
    dataset_index = list(range(100))
    label_file = './files/files.txt'
    template = np.array(pd.read_csv("./template/final_points.csv.gz", header=None).values)
    dataset = CTimageData(root_dir, dataset_index, dtype = 'test',label_file = label_file, template = template)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    for i in dataloader:
        print(i[0].shape)
