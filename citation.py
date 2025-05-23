import time
import random, math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import f1_score
from utils import load_citation, sgc_precompute, set_seed, smn_precompute, centroid_distance, sub_cs, load_facebook, sub_topk
from models import get_model
from metrics import accuracy, f1, cs_accuracy, cs_eval_metrics
import pickle as pkl
import networkx as nx
from args import get_citation_args
from time import perf_counter
import csv

# Arguments
args = get_citation_args()

# setting random seeds
set_seed(args.seed, args.cuda)
    
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, 
                                                                args.normalization, 
                                                                cuda = True)
print(labels.shape, idx_train.shape, idx_val.shape, idx_test.shape)
model = get_model(args.model, features.size(1), labels.max().item()+1, 
                  args.hidden, args.ssf_dim, args.sp_rate, 
                  args.hop, args.heads, args.negative_slope, 
                  args.dropout, args.dataset, cuda = True) 

if args.model == "SMN": features_channel, precompute_time = smn_precompute(features, adj, args.hop)
elif args.model == "GCN": features_channel, precompute_time = smn_precompute(features, adj, 0)
elif args.model == "SGC": features_channel, precompute_time = sgc_precompute(features, adj, args.hop)

print("{:.4f}s".format(precompute_time))

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):
    set_seed(args.seed, args.cuda)

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    gamma = 2.0  # Focusing parameter
    alpha = 0.25  # Balance parameter
    gamma_neg=0 # 4
    gamma_pos=4 # 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, low_passing_filter, emb, spatial_loss, sigma = model(train_features)
        class_loss = F.cross_entropy(output, train_labels)

        spatial_loss = F.cross_entropy(spatial_loss, train_labels)
        if args.ssf:
            total_loss = 0.5 * class_loss / sigma[0]**2 + 0.5 * spatial_loss / sigma[1]**2 + model.l1_penalty()
        else:
            total_loss = class_loss
        total_loss.backward()
        optimizer.step()
        acc_train = accuracy(output, train_labels)
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output, low_passing_filter, emb, spatial_loss, sigma = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_train, acc_val, train_time, low_passing_filter.transpose(0, 1)

def test_regression(model, test_features, test_labels):
    set_seed(args.seed, args.cuda)
    model.eval()
    return (accuracy(model(test_features)[0], test_labels), 
            model(test_features)[0], 
            model(test_features)[1].transpose(0, 1), 
            model(test_features)[2],
            model(test_features)[-1])

set_seed(args.seed, args.cuda)
if args.model == "SMN" or args.model == "SGC":
    (model, acc_train, acc_val, 
     train_time, low_passing_filter) = train_regression(model, features_channel[idx_train], 
                                                        labels[idx_train], features_channel[idx_val], 
                                                        labels[idx_val], args.epochs, 
                                                        args.weight_decay, args.lr, args.dropout)
    (acc_test, model_output, 
     low_passing_filter_test, emb, sigma) = test_regression(model, 
                                                            features_channel, 
                                                            labels)
    print(low_passing_filter[0])


    #----------------------------- Community Search -----------------------------#
    sample_weight = [1/adj[i]._nnz() for i in range(adj.shape[0])]
    query_nodes = torch.multinomial(idx_test.type(torch.FloatTensor).cuda(), 
                                    50, replacement = False).tolist()
    
    query_labels = torch.LongTensor(torch.argmax(model_output[torch.LongTensor(query_nodes)], dim = 1).cpu())

    if args.cs == 'sub_cs':
        communities, cs_time = sub_cs(adj, emb, query_nodes, 
                                      low_passing_filter_test[query_labels], 
                                      community_size = args.comm_size, 
                                      early_stop = 2, lp_filter = args.ssf)
    if args.cs == 'sub_topk':
        communities, cs_time = sub_topk(adj, emb, query_nodes, 
                                        low_passing_filter_test[query_labels], 
                                        community_size = args.comm_size, 
                                        early_stop = 2, lp_filter = args.ssf)
    cs_acc, cs_f1 = cs_eval_metrics(communities, query_nodes, labels)
    #----------------------------- Community Search -----------------------------#


print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s, CS Time: {:.4f}s".format(precompute_time, 
                                                                                                train_time, 
                                                                                                precompute_time+train_time, 
                                                                                                cs_time))
print("Training Time: {}".format(round(precompute_time+train_time, 4)))
print("Query Time: {}".format(round(cs_time, 6)))
print('Test_acc/f1: {}'.format(torch.round(acc_test, decimals = 4)))
print("Accuracy: {}".format(round(cs_acc, 4)))
print("F1 Score: {}".format(round(cs_f1, 4)))


