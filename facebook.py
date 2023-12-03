import time
import random, math
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from utils import sgc_precompute, set_seed, smn_precompute, sub_cs, load_facebook, sub_topk
from models import get_model
from metrics import overlap_cs_eval_metrics
import pickle as pkl
import networkx as nx
from args_facebook import get_citation_args
from time import perf_counter
# Arguments
args = get_citation_args()

# setting random seeds
set_seed(args.seed, args.cuda)
    

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, low_passing_filter, emb, spatial_loss, sigma = model(train_features)
        class_loss = F.cross_entropy(output, train_labels.float())
        spatial_loss = F.cross_entropy(spatial_loss, train_labels.float())
        if args.ssf:
            total_loss = 0.5 * class_loss / sigma[0]**2 + 0.5 * spatial_loss / sigma[1]**2 
        else:
            total_loss = class_loss

        total_loss.backward()
        optimizer.step()
        f1_train = f1_score(train_labels.cpu().detach(), 
                            output.round().type(torch.IntTensor).cpu().detach(), average = 'samples', zero_division = 0)
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output, low_passing_filter, emb, spatial_loss, sigma = model(val_features)
        f1_val = f1_score(val_labels.cpu().detach(), 
                            output.round().type(torch.IntTensor).cpu().detach(), average = 'samples')


    return model, f1_train, f1_val, train_time, low_passing_filter

def test_regression(model, test_features, test_labels):
    model.eval()
    output, low_passing_filter, emb, spatial_loss, sigma = model(test_features)
    f1_test = f1_score(test_labels.cpu().detach(), 
                        output.round().type(torch.IntTensor).cpu().detach(),
                        average = 'samples')
    return f1_test, output, low_passing_filter.transpose(0, 1), emb

ave_acc = []
ave_f1 = []
if args.dataset == 'facebook':
    test_adj, features, labels, idx_train, idx_val, idx_test = load_facebook(source_node=args.fb_num, normalization = args.normalization, cuda = True) #adj, features, edge_idx, 

    model = get_model(args.model, features.size(1), len(labels[0]), 
                    args.hidden, args.ssf_dim, args.sp_rate, args.hop, args.heads, args.negative_slope, 
                    args.dropout, args.dataset, cuda = True) 

    if args.model == "SMN": features_channel, precompute_time = smn_precompute(features, test_adj, args.hop)
    elif args.model == "SGC": features_channel, precompute_time = sgc_precompute(features, test_adj, args.hop)

    if args.model == "SMN" or args.model == "SGC":
        set_seed(args.seed, args.cuda)
        (model, f1_train, f1_val,
        train_time, low_passing_filter) = train_regression(model, features_channel[idx_train], labels[idx_train], 
                                                                        features_channel[idx_val], labels[idx_val],
                                                                        args.epochs, args.weight_decay, args.lr, args.dropout)
        f1_test, model_output, low_passing_filter_test, emb= test_regression(model, features_channel, labels)

        # Community Search 
        query_nodes = torch.multinomial(idx_test.type(torch.FloatTensor), 50, replacement = True)#.tolist()
        query_labels = model_output[query_nodes].round().type(torch.LongTensor)
        if args.case == 1:
            one_hot_label = []
            labels_idx = []
            for i in query_labels:
                one_hot_label.extend(F.one_hot(i.nonzero().reshape(-1), num_classes = i.shape[0]).tolist())
                labels_idx.extend(i.nonzero().reshape(-1))
            labels_idx = torch.LongTensor(labels_idx)
            query_nodes = query_nodes[query_labels.nonzero()[:,0]]
            ssf = low_passing_filter_test[labels_idx]
            target_label_pred = one_hot_label
            if args.cs == 'sub_cs':
                communities, cs_time = sub_cs(test_adj, emb, query_nodes, ssf, 
                                community_size = 30, early_stop = 16, lp_filter = args.ssf)
                print('test', args.ssf)
            if args.cs == 'sub_topk':
                communities, cs_time = sub_topk(test_adj, emb, query_nodes, ssf, 
                                community_size = 30, early_stop = 16, lp_filter = args.ssf)
            cs_acc, cs_f1 = overlap_cs_eval_metrics(communities, query_nodes, target_label_pred, labels[idx_test], args.case)

        elif args.case == 2:
            overlapped_ssf = []
            labels_idx = []
            for i in query_labels:
                ssfs = torch.matmul(i.float().cuda(), low_passing_filter_test) # get the overlapped subspace
                overlapped_ssf.append(ssfs) 
            ssf = torch.stack(overlapped_ssf)
            target_label_pred = query_labels.tolist()
            if args.cs == 'sub_cs':
                communities, cs_time = sub_cs(test_adj, emb, query_nodes, ssf, 
                                community_size = 5, early_stop = 16, lp_filter = args.ssf)
            if args.cs == 'sub_topk':
                communities, cs_time = sub_topk(test_adj, emb, query_nodes, ssf, 
                                community_size = 5, early_stop = 16, lp_filter = args.ssf)
            cs_acc, cs_f1 = overlap_cs_eval_metrics(communities, query_nodes, target_label_pred, labels[idx_test], args.case)

    print("Training Time: {}".format(round(precompute_time+train_time, 4)))
    print("Query Time: {}".format(round(cs_time, 6)))
    print("Accuracy: {}".format(round(cs_acc, 4)))
    print("F1 Score: {}".format(round(cs_f1, 4)))
