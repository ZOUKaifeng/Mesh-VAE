"""
Created on Mon Oct 15 13:43:10 2020

@Author: Kaifeng

@Contact: kaifeng.zou@unistra.fr

chebyshev conv and surface pooling for graph classification
"""
import logpdf
from math import sqrt
from nn.pool import SurfacePool
from nn.conv import ChebConv_batch
import torch
import torch.nn as nn
import torch.nn.functional as F

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        self.linear = self.linear.to(input.device)
        return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        #self.norm = nn.InstanceNorm1d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style, EPS=1e-9):
        # print(style.shape)
        style = self.style(style).unsqueeze(1)
        # print(style.shape)
        gamma, beta = style.chunk(2, 2)
        # print(gamma.shape)
        # print(beta.shape)

        mean = torch.mean(input, dim = 1, keepdim = True)
        var = torch.var(input, dim = 1, keepdim = True)
        out = (input - mean)/torch.sqrt(var + EPS)
        # print(out.shape)

        out = gamma * out + beta

        return out


class cheb_VAE(torch.nn.Module):

    def __init__(self, num_features, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes, model = 'MSE_VAE'):
        super(cheb_VAE, self).__init__()
        self.n_layers = config['n_layers']
        self.filters = list(config['num_conv_filters'])

        self.filters.insert(0, num_features)  # To get initial features per node
        self.K = config['polygon_order']

        
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_batch.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])

        # convolution layer
        self.cheb = torch.nn.ModuleList([ChebConv_batch(self.filters[i], self.filters[i+1], self.K[i])
                                         for i in range(len(self.filters)-2)])


        self.cheb_dec = torch.nn.ModuleList([ChebConv_batch(self.filters[-i-1], self.filters[-i-2], self.K[i])
                                             for i in range(len(self.filters)-1)])
        # for sa in self.A_edge_index:

        #     print(sa.shape)




        self.cheb_dec[-1].bias = None  # No bias for last convolution layer

        self.pool = SurfacePool()

        self.num_class = config['num_classes']

        # self.num_style = config['num_style']


        #Varient latent space
        self.z = config['num_style']

        self.num_hidden = config['num_hidden']

        self.classifier_layer = nn.Linear(self.num_hidden, self.num_class )


        self.z_mean = nn.Linear(self.num_hidden+self.num_class, self.z )

        self.z_log_var = nn.Linear(self.num_hidden+self.num_class, self.z)


        #Linear layer

        # self.num_hidden = config['num_hidden']

        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-1], self.num_hidden)

        self.dec_lin = torch.nn.Linear(self.z+self.num_class, self.num_hidden)

        self.dec_lin_1 = torch.nn.Linear(self.z+self.num_class, self.num_hidden)

        self.dec_lin_2 = torch.nn.Linear(self.num_hidden, self.downsample_matrices[-1].shape[0]*self.filters[-1] )
        self.dropout = nn.Dropout(p=config['dropout'])

        self.reset_parameters()
        self.type = model
        

        # print(self.filters)
        
        # self.AdaIN = [AdaptiveInstanceNorm(self.filters[-i-1], self.num_class) for i in range(len(self.filters)-1)]

       # self.log_sigma  = torch.tensor([0.0]).cuda()
        # if self.type == 'sigma_VAE':
        # ## Sigma VAE
        #     self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0], requires_grad= True)

        # elif self.type == 'non_shared_sigma_VAE':
        #     self.dec_sigma = torch.nn.Linear(num_features*self.adjacency_matrices[0].shape[0], num_features*self.adjacency_matrices[0].shape[0] )

    def set_param(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
  
    def forward(self, data, x_gt, y, supervise = True, m_type = "test"):
        self.supervise = supervise
        #x = data


        x, edge_index = data.x, data.edge_index

       # print(x.shape)
        batch_size = data.num_graphs
       # batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.filters[0])
      #  print(x.shape)
        x = self.encoder(x)
        y_hat = self.classifier(x)  
        # index_pred = torch.argmax(y_hat,  dim = 1)

        # y_hat[:, index_pred] = 1.0
        # y_hat[y_hat!=1] = 0.0

        x = torch.cat([y, x], -1)

        x_mean = self.z_mean(x)

        x_var = self.z_log_var(x)

        if m_type == "train":

            z_ = self.reparameterize(x_mean, x_var)
            

        else:
            z_ = x_mean
        #    z = torch.cat([y_hat, z_], -1)

        # if self.supervise:

        z = torch.cat([y, z_], -1)

        # oppo_y = torch.abs(1-y).float()
        # oppo_z = torch.cat([oppo_y, z_], -1)
        # # # z = y
        # # else:

        # #     z = torch.cat([labels_onehot, z_], -1)
        x = self.decoder(z)
        # oppo_x = self.decoder(oppo_z)
        # if self.type == 'non_shared_sigma_VAE':
        #     self.log_sigma = self.dec_sigma(x.view(batch_size, -1)).reshape(batch_size, -1, self.filters[0])
        


     #   x = x.reshape(batch_size, -1)
      #  x_gt = data.x.reshape(batch_size, -1, self.filters[0])
        x = x.reshape(batch_size, -1, self.filters[0])

        loss, correct, kld, rec_loss = self.loss_function(x_gt, x, z, x_mean, x_var, y, y_hat)

        # index_pred = torch.argmax(y_hat)

        
       
        return loss, correct, x, [kld, rec_loss, z_], y_hat

    def classifier(self, x):
    #    print(self.classifier_layer(x).shape)
        x = self.dropout(x)
        y_hat = torch.nn.functional.softmax(self.classifier_layer(x), dim = 1)
     #   print(y_hat)
        return y_hat


    def encoder(self, x):
        for i in range(self.n_layers):
        #    print(x.shape)
            x = F.relu(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            x = self.pool(x, self.downsample_matrices[i])
            # x = self.dropout(x)
           
            

        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = F.relu(self.enc_lin(x))
        x = self.dropout(x)
        return x

    def decoder(self, x):

        x = F.relu(self.dec_lin(x))
        x = self.dropout(x)
        x = F.relu(self.dec_lin_2(x))
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):
            # x = self.AdaIN[i](x, style)
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
            # x = self.dropout(x)
        # x = self.AdaIN[-1](x, style)
        recon_x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])

      
 
        return recon_x

    def sample(self, y, z):
       

        batch_size = z.shape[0]

        z_ = torch.cat([y, z], -1)
        # z = y
        x = self.decoder(z_)

        x = x.reshape(batch_size, -1, self.filters[0])

        return x



    def reparameterize(self, mu, logvar):

        batch_size = mu.shape[0]
        dim = logvar.shape[1]

        std = torch.exp(logvar * 0.5)
  
        z = torch.normal(mean = 0, std = 1, size=(batch_size, dim)).to(mu.device)
        z = z*std + mu

        return z

    def loss_function(self, x, recon_x, z, mu_z, logvar_z, y, y_hat):
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        batch_size = x.shape[0]
        kld = logpdf.KLD(mu_z, logvar_z)
     
        log_sigma =   torch.Tensor([1]).to(device = x.device) # decrease 1 to  higher for better reconstruction
        log_sigma = logpdf.softclip(log_sigma, -6)


            #rec_loss = logpdf.bernoulli(recon_x, x)
        rec_loss = logpdf.gaussian_nll(recon_x, log_sigma, x).sum(-1).sum(-1) # test wether recon_x contains nan? maybe change sum(-1).sum(-1) to mean() ?
   
        index_pred = torch.argmax(y_hat,  dim = 1)
        index = torch.argmax(y,  dim = 1)

        correct = torch.sum(index_pred == index)


        logqy = (y_hat*y).sum(-1).log() # classification accuracy
        loss = (kld + rec_loss - 2*logqy).mean()

       # loss =  - logqy.mean()
              # loss = -(-kld  + p(x|z)) = kld + rec_loss
        return loss, correct, kld, rec_loss


    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)


