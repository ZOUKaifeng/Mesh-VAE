import torch
from torch_scatter import scatter_add
#from torch_geometric.nn.conv import MessagePassing
#from torch_geometric.nn.conv.cheb_conv import ChebConv
from .conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn import dense_diff_pool
# from torch_geometric.nn import global_sort_pool

from utils import normal


class SurfacePool(MessagePassing):
    def __init__(self):
        super(SurfacePool, self).__init__(flow='target_to_source')

    def forward(self, x, pool_mat,  dtype=None):
        x = x.transpose(0,1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out.transpose(0,1)

    def message(self, x_j, norm):
        return norm.view(-1, 1, 1) * x_j




class SortPool(torch.nn.Module):

    def __init__(self, k):
        super(SortPool, self).__init__()
        self.k = k

    def forward(self, x):
        # sorted_x = x[:,:,-1].sort(dim=-1, descending=True)
        # b = x.shape[0]
        # pooled_x = torch.zeros(b, self.k , x.shape[-1]).to(x.device)
        # for i in range(b):
        #     pooled_x[i,:,:] = x[i, sorted_x.indices[i,:self.k], :]
        fill_value = x.min().item() - 1
        k = self.k 
        batch_x = x
        B, N, D = batch_x.size()

        _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
        arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
        perm = perm + arange.view(-1, 1)

        batch_x = batch_x.view(B * N, D)
        batch_x = batch_x[perm]
        batch_x = batch_x.view(B, N, D)

        if N >= k:
            batch_x = batch_x[:, :k].contiguous()
        else:
            expand_batch_x = batch_x.new_full((B, k - N, D), fill_value)
            batch_x = torch.cat([batch_x, expand_batch_x], dim=1)

        batch_x[batch_x == fill_value] = 0
        x = batch_x.view(B, k * D)

        return x

        # return pooled_x 



class DIFFPool(torch.nn.Module):
    """
    Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper.
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: in num_nodes
        :param out_channels: out num_nodes
        """
        super(DIFFPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = nn.Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.s)

    def forward(self, x, adj):
        """
        Returns pooled node feature matrix, coarsened adjacency matrix and the
        auxiliary link prediction objective
        Args:
            adj: Adjacency matrix with shape [num_nodes, num_nodes]
        """
        out_x, out_adj, reg = dense_diff_pool(x, adj, self.s)
        out_adj = out_adj.squeeze(0) if out_adj.dim() == 3 else out_adj
        # with timeit('adj_to_edge_index'):
        #     out_edge_index, out_edge_attr = adj_to_edge_index(out_adj)
        # TODO: too slow
        return out_x, None, None, out_adj, reg

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

