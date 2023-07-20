"""
Created on Mon Oct 19 17:20:10 2020

@Author: Kaifeng

@Contact: kaifeng.zou@unistra.fr

model selection
"""


import os
from models.cheb_VAE import cheb_VAE
from models.Spatial_conv import SpatialConv
from models.graph_attention import GAT

from models.DGCNN import DGCNN
# # from models.Dynqmic_graph import DGCNN_cls
from psbody.mesh import Mesh
import torch
import mesh_operations
import open3d as o3d
import numpy as np
from utils import *


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def get_model(config, device):
    template_mesh = o3d.io.read_triangle_mesh(config['template'])
    template_mesh = Mesh(v=template_mesh.vertices, f=template_mesh.triangles)
    num_feature = template_mesh.v.shape[1]

    if config['type'] == 'cheb_VAE':
        

        M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

        D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
        U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
        A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
        num_nodes = [len(M[i].v) for i in range(len(M))]

        for a in A_t:
            print(a.shape)
        print('Using model: cheb_GCN')
        net = cheb_VAE(num_feature, config, D_t, U_t, A_t, num_nodes, model = config['model']).to(device)
        for name,parameters in net.named_parameters():
            print(name,':',parameters.size())
        torch.save(net.state_dict(), os.path.join(config['checkpoint_dir'], 'initial_weight.pt'))


    elif config['type'] == 'saptial_conv':
        
        M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])


        D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
        U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
        A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
        num_nodes = [len(M[i].v) for i in range(len(M))]


        print('Using model: saptial_conv')
        net = SpatialConv(num_feature, config, D_t, A_t, num_nodes).to(device)
        torch.save(net.state_dict(), os.path.join(config['checkpoint_dir'], 'initial_weight.pt'))

    elif config['type'] == 'DGCNN':

        adj = mesh_operations.get_vert_connectivity(template_mesh.v, template_mesh.f).tocoo()
        adj = scipy_to_torch_sparse(adj).to(device)
        net = DGCNN(template_mesh.v.shape[1], config['z'], adj).to(device)

        print('Using model: DGCNN')
        torch.save(net.state_dict(), os.path.join(config['checkpoint_dir'], 'initial_weight.pt'))

    elif config['type'] == 'Graph attention network':

        M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])


        D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
        U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
        A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
        num_nodes = [len(M[i].v) for i in range(len(M))]


        print('Using model: Graph attention network')
        net = GAT(num_feature, config, D_t, A_t, num_nodes).to(device)

        torch.save(net.state_dict(), os.path.join(config['checkpoint_dir'], 'initial_weight.pt'))


    else:
        raise RuntimeError('No such model type, please choose model from ["cheb_GCN", "saptial_conv", "DGCNN", "raph attention network"]')

    return net

        