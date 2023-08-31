"""
Created on Mon Oct 15 13:43:10 2020

@Author: Kaifeng

@Contact: kaifeng.zou@unistra.fr

chebyshev conv and surface pooling for graph classification
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
#from nn.pool import SurfacePool
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_scatter import scatter_add


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out




class graph_norm(torch.nn.Module):
    def __init__(self, num):
        super(graph_norm, self).__init__()
        self.num = num

        self.gamma = torch.nn.Parameter(torch.zeros((num)))
        self.beta = torch.nn.Parameter(torch.ones(num))
        self.eps = 1e-9


    def forward(self, x):
        batch_mean = torch.mean(x, dim = 0, keepdims = True)  # bs, 1, 8
        batch_var = (((x-batch_mean)**2).sum(0)/x.shape[0]).unsqueeze(0)  #
        # print(batch_mean.shape)
        # print(batch_var.shape)

        x = (x-batch_mean)/((batch_var+self.eps)**0.5)
        # print(std)
       
        x = self.beta * x + self.gamma

        return x

class cheb_GCN(torch.nn.Module):

    def __init__(self, num_feature, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(cheb_GCN, self).__init__()
        self.n_layers = config['n_layers']
        self.filters = config['num_conv_filters'].copy()
        self.filters.insert(0, num_feature)  # To get initial features per node
        self.z = config['num_classes']
        self.K = config['polygon_order']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices

#edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        self.A_edge_index=[]
        for i in range(len(num_nodes)):
            index, _ = remove_self_loops(self.adjacency_matrices[i]._indices()) 
            self.A_edge_index.append(index)


        self.cheb = torch.nn.ModuleList([ChebConv(self.filters[i], self.filters[i+1], self.K[i])
                                         for i in range(len(self.filters)-2)])

        # self.enc_batch_norm = torch.nn.ModuleList([graph_norm(self.filters[i+1]) for i in range(len(self.filters)-2)])
      
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-2], 128)
        # print(self.downsample_matrices[-1].shape[0]*self.filters[-1])
        self.cls_layer = torch.nn.Linear(128, self.z)
        self.reset_parameters()

    def forward(self, data):
        x = data
        batch_size = x.shape[0]
      #  x, edge_index, batch = data.x, data.edge_index, data.batch
       # batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters[0])
        # print(x.shape)
        for i in range(self.n_layers): 
          #  print(self.A_norm[i].shape)
            x = self.cheb[i](x, self.A_edge_index[i])
            # x = self.enc_batch_norm[i](x)
            x = F.relu(x)
            
            x = Pool(x, self.downsample_matrices[i])
        
        x = x.reshape(x.shape[0], self.enc_lin.in_features)

        x = self.cls_layer(F.relu(self.enc_lin(x)))

        return x



    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.cls_layer.weight, 0, 0.1)
        for i in range(self.n_layers):
            self.cheb[i].reset_parameters()
        print('Reset parameters...')
   


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(221,30,1)               #1D Convolutional layer : in_channels=221points, out_channels=30points, kernel_size=1
        self.bn0 = nn.BatchNorm1d(num_features=30)     #Batch Normalization layer : 30 points
        self.fc3 = nn.Linear(90, 10)                   #Fully-connected linear layer : in_features=90, out_features=10
        self.fc4 = nn.Linear(10, 5)                    #Fully-connected linear layer : in_features=10, out_features=5
        self.fc5 = nn.Linear(5, 1)                     #Fully-connected linear layer : in_features=5,  out_features=1
        
    # Principal function of the network
    def forward(self, x):                   # x.size() = (N,221,3)
        x = self.conv1(x)                   # x.size() = (N,30,3) 
        x = self.bn0(x)                     # x.size() = (N,30,3) 
        x = x.view(-1,90)                   # x.size() = (N,90) 
        x = torch.relu(self.fc3(x))         # x.size() = (N,10) ; ReLU activation function
        x = torch.relu(self.fc4(x))         # x.size() = (N,5)  ; ReLU activation function
        x = self.fc5(x)                     # x.size() = (N,1) 
        return x







