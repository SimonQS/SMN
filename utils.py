import numpy as np
import os
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import random, math
from args_mag import get_citation_args
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter


args = get_citation_args()

def set_seed(seed, cuda):
    if cuda: torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)   

# setting random seeds
set_seed(args.seed, args.cuda)

def deterministic_sample(weights, num_samples, seed):
    np.random.seed(seed)
    indices = np.random.choice(len(weights), 
                               size=num_samples, 
                               replace=False, 
                               p=weights/np.sum(weights))
    return list(indices)

def load_data(dataset, normalization, cuda = True):
    if args.dataset in ["cora", "citeseer", "pubmed"]:
        return load_citation(dataset, normalization, cuda)
    elif args.dataset == 'reddit':
        return load_reddit_data(dataset, normalization, cuda)
    elif args.dataset == 'facebook':
        print(args.dataset)
        return load_facebook(source_node=args.fb_num, 
                             normalization=normalization, 
                             cuda=cuda)
    elif args.dataset.startswith('mag_'):
        return load_mag(dataset, normalization, cuda)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN", model = args.model): # Degree normalisation
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj, model)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    set_seed(args.seed, args.cuda)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        set_seed(args.seed, args.cuda)
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # idx_test = test_idx_range.tolist()
    idx_test = range(len(y)+500, len(labels))
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def load_mag(dataset_str="mag_eng", normalization="AugNormAdj", cuda=True):
    """
    Load MAG Datasets.
    """
    set_seed(args.seed, args.cuda)
    if not dataset_str.endswith('.npz'):
        dataset_str += '.npz'
        dataset_str = 'data/MAG/'+ dataset_str
    with np.load(dataset_str, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_matrix.data'], loader['adj_matrix.indices'],
                           loader['adj_matrix.indptr']), shape=loader['adj_matrix.shape'])

        if 'attr_matrix.data' in loader.keys():
            X = sp.csr_matrix((loader['attr_matrix.data'], loader['attr_matrix.indices'],
                               loader['attr_matrix.indptr']), shape=loader['attr_matrix.shape'])
        else:
            X = None

        Z = sp.csr_matrix((loader['labels.data'], loader['labels.indices'],
                           loader['labels.indptr']), shape=loader['labels.shape'])

        # Remove self-loops
        A = A.tolil()
        A.setdiag(0)
        A = A.tocsr()

        # Convert label matrix to numpy
        if sp.issparse(Z):
            Z = Z.toarray().astype(np.float32)

        graph = {
            'A': A,
            'X': X,
            'Z': Z
        }

        node_names = loader.get('node_names')
        if node_names is not None:
            node_names = node_names.tolist()
            graph['node_names'] = node_names

        attr_names = loader.get('attr_names')
        if attr_names is not None:
            attr_names = attr_names.tolist()
            graph['attr_names'] = attr_names

        class_names = loader.get('class_names')
        if class_names is not None:
            class_names = class_names.tolist()
            graph['class_names'] = class_names


    adj, features, labels = graph['A'], graph['X'], graph['Z']    
    idx_test = [*range(0, len(labels), 1)]
    overlap_labels = labels[idx_test]
    adj = adj[idx_test, :][:, idx_test] 
    features = features[idx_test]
    new_idx_test = [*range(0, len(idx_test), 1)]

    label_weight = 1/np.sum(overlap_labels, axis = 0) # oversampling
    sample_weight = np.sum(label_weight * overlap_labels, axis = 1)
    set_seed(args.seed, args.cuda)

    idx_train = deterministic_sample(sample_weight, 
                                     num_samples=int(0.1 * len(sample_weight)), 
                                     seed=args.seed)
    idx_val = deterministic_sample(sample_weight, 
                                   num_samples=int(0.1 * len(sample_weight)), 
                                   seed=args.seed + 1)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(overlap_labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(new_idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def load_facebook(source_node, normalization="AugNormAdj", cuda = True):
    set_seed(args.seed, args.cuda)
    print('seed:',args.seed)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data','facebook','data')
    print('Load facebook data')

    file_circle= path + f'//{str(source_node)}.circles'
    file_edges=path + f'//{str(source_node)}.edges'
    file_egofeat=path + f'//{str(source_node)}.egofeat'
    file_feat=path + f'//{str(source_node)}.feat'
    edges=[]
    node=[]
    feature = {}
    with open(file_egofeat) as f:
        feature[source_node] = [int(i) for i in f.readline().split()]
    with open(file_feat) as f:
        for line in f:
            line = [int(i) for i in line.split()]
            feature[int(line[0])] = line[1:]
            node.append(int(line[0]))
    with open(file_edges,'r') as f:
        for line in f:
            u,v=line.split()
            u=int(u)
            v=int(v)
            if(u in feature.keys() and v in feature.keys()):
                edges.append((u,v))

    for i in node:
        edges.append((source_node, i))
    node=sorted(node+[source_node])
    mapper = {n: i for i, n in enumerate(node)}
    edges=[(mapper[u],mapper[v]) for u,v in edges]

    node=[mapper[u] for u in node]
    idx_test = feature.keys()
    features=[0]*len(node)
    for i in list(feature.keys()):
        features[mapper[i]]=feature[i]
    circle=[]
    with open(file_circle) as f:
        for line in f:
            line=line.split()
            line=[ mapper[int(i)] for i  in line[1:]]
            if(len(line)<30):continue # only include communities with size no less than 30
            circle.append(line)

    source_node=mapper[source_node]

    args.ego = 'facebook_' + str(source_node)
    features = np.array(features)
    adj = nx.from_edgelist(edges)
    adj = nx.adjacency_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = [[] for _ in range(len(features))]
    for i in range(len(features)): 
        for c in circle:
            labels[i].append((i in c)*1)
    idx_test = sorted(list(set([i for sub in circle for i in sub]))) # only indclude nodes with at leat one community
    labels = torch.LongTensor(labels)
    labels = labels[idx_test]
    
    label_weight = 1/torch.sum(labels, dim = 0) # oversampling
    sample_weight = torch.sum(label_weight * labels, dim = 1)
    set_seed(args.seed, args.cuda)

    new_idx_train = list(torch.utils.data.WeightedRandomSampler(sample_weight, 
                                                                num_samples=int(0.1*len(sample_weight)), 
                                                                replacement = False))
    new_idx_val = list(torch.utils.data.WeightedRandomSampler(sample_weight, 
                                                              num_samples=int(0.1*len(sample_weight)), 
                                                              replacement = False))
    new_idx_test = [*range(0, len(idx_test), 1)]

    test_adj = adj[idx_test, :][:, idx_test] 
    test_features = features[idx_test]

    test_features = sp.lil_matrix(test_features, dtype = float)
    test_adj, test_features = preprocess_citation(test_adj, test_features, normalization)
    test_features = torch.FloatTensor(np.array(test_features.todense())).float()

    test_adj = sparse_mx_to_torch_sparse_tensor(test_adj).float()
    
    new_idx_train = torch.LongTensor(new_idx_train)
    new_idx_val = torch.LongTensor(new_idx_val)
    new_idx_test = torch.LongTensor(new_idx_test)

    if cuda:
        test_features = test_features.cuda()
        test_adj = test_adj.cuda()
        labels = labels.cuda()
        new_idx_train = new_idx_train.cuda()
        new_idx_val = new_idx_val.cuda()
        new_idx_test = new_idx_test.cuda()

    return test_adj, test_features, labels, new_idx_train, new_idx_val, new_idx_test # adj, features, edge_idx, 

def smn_precompute(features, adj, hop): # k-hop
    t = perf_counter()
    features_channel = [features]
    for i in range(hop-1):
        features = torch.spmm(adj, features)
        features_channel.append(features)
    features_channel = torch.stack(features_channel).swapaxes(0, 1)
    features_channel = features_channel.cuda()
    precompute_time = perf_counter()-t
    return features_channel, precompute_time

'''
Conmmunity search based on nodes cohesiveness in vector space.
'''
def centroid_distance(community, features):
    features_c = features[torch.LongTensor(community)]
    centroid = torch.mean(features_c, dim=0)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(features_c, centroid)
    return similarity, centroid

def update_centroid(centroid, least_similar_node, new_node, community_size, features):
    updated_centroid = centroid - (1 / community_size) * (features[least_similar_node] - features[new_node])
    return updated_centroid

def sub_cs(adjacency_matrix, features, query_nodes, 
           low_passing_filters, community_size, 
           early_stop, lp_filter):
    communities = []
    cs_start = perf_counter()

    for query in query_nodes:
        if lp_filter:
            features_processed = features * (low_passing_filters[len(communities)].bool() * 1)
        else:
            features_processed = features

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # Compute cosine similarity for the query node with all other nodes
        cos_simi = cos(features_processed, features_processed[query].reshape(1, -1)).squeeze()

        # Find the top 2*k similar nodes
        topk_prob, topk_idx = torch.topk(cos_simi, community_size * early_stop)
        
        # Initialize the community with the top k nodes
        initial_community = topk_idx[:community_size].tolist()
        community = initial_community[:]
        
        # Calculate initial centroid and similarities
        similarity, centroid = centroid_distance(community, features_processed)
        min_simi, min_idx = torch.min(similarity, dim=0)
        least_similar_node = community[min_idx]

        # Iterate over the remaining nodes in the top 2*k list
        for idx in topk_idx[community_size:]:
            query_simi = cos(features_processed[idx].reshape(1, -1), centroid)
            if query_simi > min_simi:
                community[min_idx] = idx.item()
                centroid = update_centroid(centroid, least_similar_node, idx, 
                                           community_size, features_processed)
                
                similarity, _ = centroid_distance(community, features_processed)
                min_simi, min_idx = torch.min(similarity, dim=0)
                least_similar_node = community[min_idx]
                if least_similar_node == query:
                    break

        communities.append(community)
        
    cs_time = (perf_counter() - cs_start) / len(query_nodes)
    return communities, cs_time


def sub_topk(adjacency_matrix, features, query_nodes, 
             low_passing_filters, community_size, 
             early_stop, lp_filter = True):
                
    communities = []
    cs_start = perf_counter()
    for query in query_nodes:
        if lp_filter == True:
            features_processed = features * (low_passing_filters[len(communities)].bool()*1)

        else:
            features_processed = features 


        community = []
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_simi = cos(features_processed, features_processed[query])
        dot_sim = torch.matmul(features_processed, features_processed[query])


        topk_prob, topk_idx = torch.topk(cos_simi, community_size)
        community.extend(topk_idx)

        communities.append(community)
    cs_time = (perf_counter()-cs_start)/len(query_nodes)

    return communities, cs_time


def sgc_precompute(features, adj, hop):
    t = perf_counter()
    for i in range(hop):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", model = args.model, cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test

    idx_train = val_index[:int(len(val_index)/2)] # 5% of data as training set
    idx_val = test_index
    idx_test = np.concatenate((train_index, val_index[int(len(val_index)/2):]))
    adj = adj + adj.T
    train_adj = adj[train_index, :][:, train_index]
    # print(train_index, y_val.shape)
    # raise Exception
    features = np.array(features)
    # features = np.nan_to_num(features)
    features[np.isnan(features)] = 0.
    features = torch.FloatTensor(features)
    features = (features-features.mean(dim=0))/features.std(dim=0)
    
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj, model)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    
    train_adj = adj_normalizer(train_adj, model)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, idx_train, idx_val, idx_test 


class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(torch.nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()