import torch
import torch.nn as nn
from torch.nn import Module
from torch_geometric.nn import GCNConv, HypergraphConv
import torch.nn.functional as F
import math
from args_mag import get_citation_args

args = get_citation_args()
class SMN(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, nhid, nssf, sp_rate, nhop, nheads, negative_slope, dropout, dataset): 
        super(SMN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.nssf = nssf
        self.sp_rate = sp_rate
        self.nhop = nhop
        self.nheads = nheads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.dataset = dataset
        self.pre_ln_2 = nn.Linear(self.nfeat, self.nheads * self.nhid) # Pre-attention trasnform (shared between nodes and hops)
        self.att = nn.Parameter(torch.Tensor(self.nheads, self.nhop, self.nhid)) # attention matrix size
        self.post_ln_2 = nn.Linear(self.nheads * self.nhid, self.nssf)
        self.sparse_gate = nn.Parameter(torch.Tensor(self.nssf, self.nclass))
        self.ssf = nn.Parameter(torch.Tensor(self.nssf, self.nclass))
        self.sigma = nn.Parameter(torch.ones(2)) # learn a weight between losses
        self.l1 = nn.Parameter(torch.tensor(0.01))  # L1 regularization term as a learnable parameter

        self.reset_parameters()

    def reset_parameters(self): 
        nn.init.xavier_uniform_(self.pre_ln_2.weight.data)# xavier initiallization, can try kaiming
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.post_ln_2.weight.data)
        nn.init.xavier_uniform_(self.sparse_gate)
        nn.init.xavier_uniform_(self.ssf)


    def forward(self, x):
        hops, heads, hidden = self.nhop, self.nheads, self.nhid
        x = self.pre_ln_2(x).reshape(-1, hops, heads, hidden).swapaxes(1, 2) 
        x = F.leaky_relu(x, self.negative_slope)

        attention_matrix = x * self.att
        root_matrix = attention_matrix[:, :, 0, :] # get root features matrix

        attention_score = [] 
        for hop in range(self.nhop): 
            hop_matrix = attention_matrix[:, :, hop, :]
            attention_score.append((root_matrix + hop_matrix))
        attention_score = torch.stack(attention_score).swapaxes(0, 1).swapaxes(1, 2) 
        fianl_weight = F.softmax(attention_score, dim = 2) 

        x = x * fianl_weight # sum by weight
        x = torch.sum(x, dim = 2) # (n, heads, hidden) sum by hop

        x = x.reshape(-1, hidden * heads) # (n, hidden*heads) concate heads

        x = self.post_ln_2(x) 
        x = F.leaky_relu(x, self.negative_slope)


        if args.ssf == 'hard':
        # --------------------------Hard-filters by sparse_rate----------------------------#
            ssf = self.ssf
            flattened_weights = abs(ssf).view(-1)
            sorted_weights, _ = torch.sort(flattened_weights)
            sparsity_ratio = self.sp_rate
            filter_idx = round(flattened_weights.shape[0]* sparsity_ratio)
            threshold_value = sorted_weights[filter_idx]
            gated_weight = torch.where(abs(ssf) >= threshold_value, ssf,  0) 
        # --------------------------Hard-filters by sparse_rate----------------------------#
        elif args.ssf == 'soft':
        # --------------------------Learnable Sparse Gate----------------------------#
            gate_probs = torch.sigmoid(self.sparse_gate)
            binary_gate = (gate_probs > 0.5).float()
            gated_weight = self.ssf * binary_gate
        #--------------------------Learnable Sparse Gate----------------------------#
        else:
            gated_weight = self.ssf * self.sparse_gate

        ssf_norm = F.normalize(gated_weight, p=1.0, dim=1)
        if self.dataset.startswith('mag_'):
            ssf_norm = F.normalize(gated_weight, p=2.0, dim=1)
        
        out = torch.matmul(x, ssf_norm)


        #--------------------------  Spatial Loss  --------------------------#

        spatial_dist = -1 * torch.cdist(x, ssf_norm.transpose(0, 1), p = 2)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_simi = []
        for i in ssf_norm.transpose(0, 1):
            cos_simi.append(cos(x, i))
        spatial_similarity = torch.stack(cos_simi, dim = 1)
        spatial_loss = (F.log_softmax(spatial_dist, dim = 1) + F.log_softmax(spatial_similarity, dim = 1)) * 0.5
        
        if self.dataset == 'facebook' or self.dataset.startswith('mag_'):
            subspace_x =[]
            for i in ssf_norm.transpose(0, 1):
                subspace_x.append(x * i.bool()*1)
            subspace_x = torch.stack(subspace_x, dim = 2).cuda()
            spatial_similarity = torch.cosine_similarity(subspace_x, ssf_norm, dim = 1)
            spatial_dist = -1 * torch.pairwise_distance(subspace_x.transpose(1, 2), ssf_norm.transpose(0, 1), p = 2)
            spatial_loss = (F.sigmoid(spatial_dist) + F.sigmoid(spatial_similarity)) * 0.5

        sigma = self.sigma
        #--------------------------  Spatial Loss  --------------------------#
        return out, ssf_norm, x, spatial_loss, sigma # reture filter as well to benefit community search
    
    def l1_penalty(self):
            if args.ssf == 'soft':
                return args.lammda * torch.norm(self.sparse_gate, 1)
            else:
                return 0

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, nhid, nssf, sp_rate, nhop, nheads, negative_slope, dropout, dataset): 
        super(SGC, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.nssf = nssf
        self.sp_rate = sp_rate
        self.nhop = nhop
        self.nheads = nheads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.dataset = dataset

        # self.dropout = dropout
        self.subspace_filter = nn.Parameter(torch.Tensor(self.nssf, self.nclass))
        self.sigma = nn.Parameter(torch.ones(2)) # learn a weight between losses
        self.W = nn.Linear(self.nfeat, self.nssf)
        
    def forward(self, x):     
        x = self.W(x)
        x = F.relu(x)

        ssf = self.subspace_filter
        flattened_weights = abs(ssf).view(-1)
        sorted_weights, _ = torch.sort(flattened_weights)
        sparsity_ratio = self.sp_rate
        filter_idx = round(flattened_weights.shape[0]* sparsity_ratio)
        threshold_value = sorted_weights[filter_idx]
        ssf = torch.where(abs(ssf) >= threshold_value, ssf,  0)  # unstablise the model

        out = torch.matmul(x, ssf)

        spatial_dist = -1 * torch.cdist(x, ssf.transpose(0, 1), p = 2)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_simi = []
        for i in ssf.transpose(0, 1):
            cos_simi.append(cos(x, i))
        spatial_similarity = torch.stack(cos_simi, dim = 1)
        spatial_loss = (F.log_softmax(spatial_dist, dim = 1) + F.log_softmax(spatial_similarity, dim = 1)) * 0.5
        
        if self.dataset == 'facebook' or self.dataset.startswith('mag_'):
            out = F.sigmoid(out)
            subspace_x =[]
            for i in torch.ceil(ssf).transpose(0, 1):
                subspace_x.append(x * i)
            subspace_x = torch.stack(subspace_x, dim = 2).cuda()
            spatial_dist = -1 * torch.pairwise_distance(subspace_x.transpose(1, 2), ssf.transpose(0, 1), p = 2)
            spatial_similarity = torch.cosine_similarity(subspace_x, ssf, dim = 1)
            spatial_loss = (F.sigmoid(spatial_dist) + F.sigmoid(spatial_similarity)) * 0.5

        sigma = self.sigma

        return out, ssf, x, spatial_loss, sigma 
    
    def l1_penalty(self):
            return 0
            
class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, nssf, sp_rate, dropout):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.nssf = nssf
        self.sp_rate = sp_rate

        self.gc1 = GraphConvolution(self.nfeat, self.nhid)
        self.gc2 = GraphConvolution(self.nhid, self.nssf)
        self.subspace_filter = nn.Parameter(torch.Tensor(self.nssf, self.nclass))

        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GCN_SSF(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nclass, nhid, nssf, sp_rate): 
        super(GCN_SSF, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.nssf = nssf
        self.sp_rate = sp_rate

        self.gc1 = GraphConvolution(self.nfeat, self.nhid)
        self.gc2 = GraphConvolution(self.nhid, self.nhid)
        self.gc3 = GraphConvolution(self.nhid, self.nssf)
        self.subspace_filter = nn.Parameter(torch.Tensor(self.nssf, self.nclass))

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = self.gc2(x, adj)
        x = F.relu(x)
        x = self.gc3(x, adj)


        ssf = self.subspace_filter
        flattened_weights = abs(ssf).view(-1)
        sorted_weights, _ = torch.sort(flattened_weights)
        sparsity_ratio = self.sp_rate
        filter_idx = round(flattened_weights.shape[0]* sparsity_ratio)
        threshold_value = sorted_weights[filter_idx]
        ssf = torch.where(abs(ssf) >= threshold_value, ssf,  0)  # unstablise the model

        out = torch.matmul(x, ssf)

        spatial_dist = -1 * torch.cdist(x, ssf.transpose(0, 1), p = 2)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_simi = []
        for i in ssf.transpose(0, 1):
            cos_simi.append(cos(x, i))
        spatial_similarity = torch.stack(cos_simi, dim = 1)
        spatial_loss = (F.log_softmax(spatial_dist, dim = 1) + F.log_softmax(spatial_similarity, dim = 1)) * 0.5
        
        sigma = self.sigma

        return out, ssf, x, spatial_loss, sigma 


def get_model(model_opt, nfeat, nclass, nhid, nssf, sp_rate, nhop, nheads, negative_slope, dropout, dataset, cuda=True):
    if model_opt == "GCN":
        model = GCN_SSF(nfeat=nfeat,
                    nclass=nclass,
                    nhid=nhid,
                    nssf = nssf,
                    sp_rate = sp_rate)
        
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass,
                    nhid=nhid,
                    nssf = nssf,
                    sp_rate = sp_rate,
                    nhop = nhop, 
                    nheads = nheads,
                    negative_slope = negative_slope, 
                    dropout=dropout,
                    dataset = dataset)
    
    elif model_opt == "SMN":
        model = SMN(nfeat=nfeat,
                    nclass=nclass,
                    nhid=nhid,
                    nssf = nssf,
                    sp_rate = sp_rate,
                    nhop = nhop, 
                    nheads = nheads,
                    negative_slope = negative_slope, 
                    dropout=dropout,
                    dataset = dataset)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model
