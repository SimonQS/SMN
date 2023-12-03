import numpy as np
import os
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import random, math
from args import get_citation_args
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from collections import deque
import heapq

args = get_citation_args()

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
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
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
    idx_test = range(len(ty) + len(ally))
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
    idx_train = list(torch.utils.data.WeightedRandomSampler(sample_weight, num_samples=int(0.1*len(sample_weight)), replacement = False))
    idx_val = list(torch.utils.data.WeightedRandomSampler(sample_weight, num_samples=int(0.1*len(sample_weight)), replacement = False))

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
    # adj = nx.to_scipy_sparse_array(adj, format = 'csr')    
    adj = nx.adjacency_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # label=circle[random.randint(0, len(circle) - 1)]
    labels = [[] for _ in range(len(features))]
    for i in range(len(features)): 
        for c in circle:
            labels[i].append((i in c)*1)
    idx_test = list(set([i for sub in circle for i in sub])) # only indclude nodes with at leat one community
    
    labels = torch.LongTensor(labels)
    labels = labels[idx_test]
    
    # label_weight = torch.sum(labels, dim = 0)/labels.shape[0]
    label_weight = 1/torch.sum(labels, dim = 0) # oversampling
    sample_weight = torch.sum(label_weight * labels, dim = 1)

    new_idx_train = list(torch.utils.data.WeightedRandomSampler(sample_weight, num_samples=int(0.1*len(sample_weight)), replacement = False))
    new_idx_val = list(torch.utils.data.WeightedRandomSampler(sample_weight, num_samples=int(0.1*len(sample_weight)), replacement = False))
    new_idx_test = [*range(0, len(idx_test), 1)]


    # edge_idx = adj[idx_train, :][:, idx_train] # only indclude nodes with at leat one community - update adj
    test_adj = adj[idx_test, :][:, idx_test] 
    test_features = features[idx_test]

    ###
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
    centroid = torch.mean(features_c, dim = 0)
    # Cosine
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(features_c, centroid)
    return similarity, centroid

def sub_cs(adjacency_matrix, features, query_nodes, low_passing_filters, community_size, early_stop, lp_filter):
        
    # Visited vector to so that a
    # vertex is not visited more than
    # once Initializing the vector to
    # false as no vertex is visited at
    # the beginning
    communities = []
    cs_start = perf_counter()

    for query in query_nodes:
        if lp_filter:
            features_processed = features * (low_passing_filters[len(communities)].bool()*1)
            # print("True")
        else:
            features_processed = features
            # print("False")

        community = []
        n = adjacency_matrix.shape[0]
        visited = [False] * n
        queue = deque([query])
        hop = 0
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # Set source as visited
        visited[query] = True

        while queue:
            vis = queue[0]


            if len(community) < community_size:
                community.append(queue[0])
                queue.popleft()
                change = True
            else: 
                ''' 
                If node is closer to the community and 
                query node is not the most far away node,
                then replace the most far awau node.
                '''
                if change:
                    similarity, centroid = centroid_distance(community, features_processed) # update centroid and distance only if changed
                    query_simi= cos(features_processed[queue[0]].reshape(1, features_processed.shape[1]), centroid)
                    min_simi, min_idx = torch.min(similarity, dim = 0)

                if community[min_idx] == query:
                        break

                elif query_simi > min_simi: 
                    # the second most far away node
                    community.remove(community[min_idx])
                    community.append(queue[0])
                    queue.popleft()
                    change = True
                else:
                    queue.popleft()
                    change = True

            for i in adjacency_matrix[vis]._indices().cpu().numpy()[0]:
                if not visited[i]:
                    if hop > early_stop: # early stop at k hop
                        break
                    else:        
                        # Push the adjacent node
                        # in the queue
                        queue.append(i)
                        
                        # set
                        visited[i] = True
            hop += 1
        communities.append(community)
    cs_time = (perf_counter()-cs_start)/len(query_nodes)
    # print(communities[0])
    # raise Exception
    return communities, cs_time


def sub_topk(adjacency_matrix, features, query_nodes, low_passing_filters, community_size, early_stop, lp_filter = True):

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

def set_seed(seed, cuda):
    if cuda: torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)   


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
    features = np.array(features)
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
