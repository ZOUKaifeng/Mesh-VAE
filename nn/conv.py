
"""
Created on Mon Oct 28 13:43:10 2020

@Author: Kaifeng

@Contact: kaifeng.zou@unistra.fr

Convolutional layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
#from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import global_sort_pool

from utils import normal










import inspect
from collections import OrderedDict

import torch
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr
import torch
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian

import math
import torch
from torch_scatter import scatter_add

#from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.utils import remove_self_loops

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

msg_aggr_special_args = set([
    'adj_t',
])

msg_special_args = set([
    'edge_index_i',
    'edge_index_j',
    'size_i',
    'size_j',
])

aggr_special_args = set([
    'ptr',
    'index',
    'dim_size',
])

update_special_args = set([])


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers of the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"` or :obj:`None`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`0`)
    """

    def __init__(self, aggr="add", flow="source_to_target", node_dim=0):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__msg_aggr_params__ = inspect.signature(
            self.message_and_aggregate).parameters
        self.__msg_aggr_params__ = OrderedDict(self.__msg_aggr_params__)

        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__)
        self.__update_params__.popitem(last=False)

        msg_aggr_args = set(
            self.__msg_aggr_params__.keys()) - msg_aggr_special_args
        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        self.__user_args__ = set().union(msg_aggr_args, msg_args, aggr_args,
                                         update_args)

        self.__fuse__ = True

        # Support for GNNExplainer.
        self.__explain__ = False
        self.__edge_mask__ = None

    def __get_mp_type__(self, edge_index):
        if (torch.is_tensor(edge_index) and edge_index.dtype == torch.long
                and edge_index.dim() == 2 and edge_index.size(0)):
            return 'edge_index'
        elif isinstance(edge_index, SparseTensor):
            return 'adj_t'
        else:
            return ValueError(
                ('`MessagePassing.propagate` only supports `torch.LongTensor` '
                 'of shape `[2, num_messages]` or `torch_sparse.SparseTensor` '
                 'for argument :obj:`edge_index`.'))

    def __set_size__(self, size, idx, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[idx] is None:
            size[idx] = tensor.size(self.node_dim)
        elif size[idx] != tensor.size(self.node_dim):
            raise ValueError(
                (f'Encountered node tensor with size '
                 f'{tensor.size(self.node_dim)} in dimension {self.node_dim}, '
                 f'but expected size {size[idx]}.'))

    def __collect__(self, edge_index, size, mp_type, kwargs):
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {'_i': i, '_j': j}

        out = {}
        for arg in self.__user_args__:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)

                if mp_type == 'edge_index':
                    out[arg] = data.index_select(self.node_dim,
                                                 edge_index[idx])
                elif mp_type == 'adj_t' and idx == 1:
                    rowptr = edge_index.storage.rowptr()
                    for _ in range(self.node_dim):
                        rowptr = rowptr.unsqueeze(0)
                    out[arg] = gather_csr(data, rowptr)
                elif mp_type == 'adj_t' and idx == 0:
                    col = edge_index.storage.col()
                    out[arg] = data.index_select(self.node_dim, col)

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        if mp_type == 'edge_index':
            out['edge_index_j'] = edge_index[j]
            out['edge_index_i'] = edge_index[i]
            out['index'] = out['edge_index_i']
        elif mp_type == 'adj_t':
            out['adj_t'] = edge_index
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['index'] = edge_index.storage.row()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_attr'] = edge_index.storage.value()

        out['size_j'] = size[j]
        out['size_i'] = size[i]
        out['dim_size'] = out['size_i']

        return out

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs.get(key, inspect.Parameter.empty)
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            adj (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                message propagation.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                Hence, the only difference between those formats is that we
                need to input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix in case :obj:`edge_index` is a
                :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """

        # We need to distinguish between the old `edge_index` format and the
        # new `torch_sparse.SparseTensor` format.
        mp_type = self.__get_mp_type__(edge_index)

        if mp_type == 'adj_t' and self.flow == 'target_to_source':
            raise ValueError(
                ('Flow direction "target_to_source" is invalid for message '
                 'propagation based on `torch_sparse.SparseTensor`. If you '
                 'really want to make use of a reverse message passing flow, '
                 'pass in the transposed sparse tensor to the message passing '
                 'module, e.g., `adj.t()`.'))

        if mp_type == 'edge_index':
            if size is None:
                size = [None, None]
            elif isinstance(size, int):
                size = [size, size]
            elif torch.is_tensor(size):
                size = size.tolist()
            elif isinstance(size, tuple):
                size = list(size)
        elif mp_type == 'adj_t':
            size = list(edge_index.sparse_sizes())[::-1]

        assert isinstance(size, list)
        assert len(size) == 2

        # We collect all arguments used for message passing in `kwargs`.
        kwargs = self.__collect__(edge_index, size, mp_type, kwargs)

        # Try to run `message_and_aggregate` first and see if it succeeds:
        if mp_type == 'adj_t' and self.__fuse__ and not self.__explain__:
            msg_aggr_kwargs = self.__distribute__(self.__msg_aggr_params__,
                                                  kwargs)
            out = self.message_and_aggregate(**msg_aggr_kwargs)
            if out == NotImplemented:
                self.__fuse__ = False

        # Otherwise, run both functions in separation.
        if mp_type == 'edge_index' or not self.__fuse__ or self.__explain__:
            msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
            out = self.message(**msg_kwargs)

            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                if out.size(0) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(0) == edge_mask.size(0)
                out = out * edge_mask.view(-1, 1)

            aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
            out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.update(out, **update_kwargs)

        return out

    def message(self, x_j):
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """

        return x_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """

        if ptr is not None:
            for _ in range(self.node_dim):
                ptr = ptr.unsqueeze(0)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def message_and_aggregate(self, adj_t):
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """

        return NotImplemented

    def update(self, inputs):
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """

        return inputs




class ChebConv(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}

    where :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a scalar when
            operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 bias=True, **kwargs):
        super(ChebConv, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, normalization, lambda_max,
             dtype=None, batch=None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if batch is not None and torch.is_tensor(lambda_max):
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight[edge_weight == float('inf')] = 0

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1,
                                                 num_nodes=num_nodes)

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None, batch=None,
                lambda_max=None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        lambda_max = 2.0 if lambda_max is None else lambda_max

        edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                     edge_weight, self.normalization,
                                     lambda_max, dtype=x.dtype, batch=batch)

        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)










class ChebConv_batch(ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization=None, bias=True):
        super(ChebConv_batch, self).__init__(in_channels, out_channels, K, normalization, bias)

    def reset_parameters(self):
        normal(self.weight, 0, 0.1)
        normal(self.bias, 0, 0.1)


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, norm, edge_weight=None):
        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])
        x = x.transpose(0,1)
        Tx_0 = x
        if self.weight.size(0) > 1:

            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            Tx_1_transpose = Tx_1.transpose(0, 1)
            out = out + torch.matmul(Tx_1_transpose, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            Tx_2_transpose = Tx_2.transpose(0, 1)
            out = out + torch.matmul(Tx_2_transpose, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):

        return norm.view(-1, 1, 1) * x_j




#Spatial graph convolution  
###  An End-to-End Deep Learning Architecture for Graph Classification  ###

class Spatial_conv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Spatial_conv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = x.transpose(0,1)
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, edge_weight = None, num_nodes=x.size(0)) #A~ = A+I
        # Linearly transform node feature matrix.
        x = self.lin(x)
        
        

        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # sqrt(D)*sqrt(D)

        # Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        
        return out.transpose(0,1)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1, 1) * x_j

    def reset_parameters(self):
        torch.nn.init.normal_(self.lin.weight, 0, 0.1)




class graph_attention(nn.Module):
    """
    GRAPH ATTENTION NETWORKS
    https://arxiv.org/abs/1710.10903
    """
    def __init__(self, num_heads, input_dim, output_dim):
        super(graph_attention, self).__init__()
        
        self.num_heads = num_heads
        self.value = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.att_q = nn.ModuleList([nn.Linear(input_dim, output_dim, bias = True) for _ in range(num_heads)])
        self.att_k =  nn.ModuleList([nn.Linear(input_dim, output_dim, bias = True) for _ in range(num_heads)])
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.drop_out = nn.Dropout(0.3)


    def forward(self, x, edge_index):
        out = []
        # x: [B,N,F] => x: [B, N, F']
        dk = sqrt(x.shape[-1])
        for i in range(self.num_heads):
            features = self.value[i](x)  # features => [B,N,F'] 
          
            query = self.att_q[i](x)   # => [B, N, F']
          
            key = self.att_k[i](x)   # => [B, N, F']
            # print(query.shape)
            # print(key.shape)

            dense = torch.bmm(query, key.permute(0,2,1))/dk # [B * N * N]  
          
            dense = self.leakyReLU(dense)
         
            #Mask inspired by https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/graph_attention_layer.py
            mask = -10e9 * (1 - edge_index)  
           

            dense += mask
       
            prop = F.softmax(dense, dim=2)
      

            drop_feat = self.drop_out(features)  # [B*N*F']
            # drop_prop = self.drop_out(prop)    #  [B*N*N]
            #print(drop_prop)

            features = torch.bmm(prop, drop_feat)  #[B,N,F']
           
            out.append(features)
      
        out = torch.mean(torch.stack(out), dim = 0)  
        out = self.leakyReLU(out)
     
        return out

    def reset_parameters(self):
        torch.nn.init.normal_(self.att_self_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.att_nbr_lin .weight, 0, 0.1)
        torch.nn.init.normal_(self.linear.weight, 0, 0.1)











        



