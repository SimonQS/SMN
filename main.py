import time
import random, math
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from utils import load_mag, load_data, sgc_precompute, set_seed, smn_precompute, centroid_distance, sub_cs, load_facebook, sub_topk, AsymmetricLoss, AsymmetricLossOptimized
from models import get_model
from metrics import accuracy, f1, cs_accuracy, cs_eval_metrics, multilabel_metrics, overlap_cs_eval_metrics
from torchvision.ops import sigmoid_focal_loss
import pickle as pkl
import networkx as nx
from args_mag import get_citation_args 
from time import perf_counter
import csv

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
    set_seed(args.seed, args.cuda)
    gamma = args.gamma # 2.0  Focusing parameter
    alpha = args.alpha # 0.25  Balance parameter
    gamma_neg = args.gamma_neg  # 0
    gamma_pos = args.gamma_pos  # 4

    pos_counts = train_labels.sum(dim=0)
    neg_counts = train_labels.size(0) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-5)
    loss_func = AsymmetricLossOptimized(gamma_neg=gamma_neg, 
                                        gamma_pos=gamma_pos, 
                                        clip=0.05, 
                                        disable_torch_grad_focal_loss=True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, low_passing_filter, emb, spatial_dist, sigma = model(train_features)
        if args.loss == 'focal':
            class_loss = sigmoid_focal_loss(output, 
                                            train_labels.float(), 
                                            alpha=alpha, 
                                            gamma=gamma, 
                                            reduction="mean")
            spatial_loss = sigmoid_focal_loss(spatial_dist, 
                                              train_labels.float(), 
                                              alpha=alpha, 
                                              gamma=gamma, 
                                              reduction="mean")
        elif args.loss == 'asl':
            class_loss = loss_func(output, train_labels.float())
            spatial_loss = loss_func(spatial_dist, train_labels.float())
        total_loss = 0.5 * class_loss / sigma[0]**2 + 0.5 * spatial_loss / sigma[1]**2 + model.l1_penalty()
        if args.loss == 'class':
            total_loss = class_loss + model.l1_penalty()
        elif args.loss == 'spatial':
            total_loss = spatial_loss + model.l1_penalty()
        elif args.loss == 'both':
            total_loss = class_loss + spatial_loss

        total_loss.backward()
        optimizer.step()
        output = (torch.sigmoid(output) > 0.5).float()
        f1_train = f1_score(train_labels.cpu().detach(), 
                            output.round().type(torch.IntTensor).cpu().detach(), 
                            average = 'macro', zero_division = 0)
        
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output, low_passing_filter, emb, spatial_dist, sigma = model(val_features)
        output = (torch.sigmoid(output) > 0.5).float()
        f1_val = f1_score(val_labels.cpu().detach(), 
                            output.round().type(torch.IntTensor).cpu().detach(), 
                            average = 'macro')

    return model, f1_train, f1_val, train_time, low_passing_filter.transpose(0, 1)

def test_regression(model, test_features, test_labels):
    set_seed(args.seed, args.cuda)
    model.eval()
    output, low_passing_filter, emb, spatial_dist, sigma = model(test_features)
    hamming_test, jacard_test = multilabel_metrics(model(test_features)[0], test_labels)
    output = (torch.sigmoid(output) > 0.5).float()
    f1_test = f1_score(test_labels.cpu().detach(), 
                        output.round().type(torch.IntTensor).cpu().detach(),
                        average = 'macro')

    return f1_test, output, low_passing_filter.transpose(0, 1), emb

ave_acc = []
ave_f1 = []

adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, 
                                                                args.normalization, 
                                                                cuda = True)
count = 0
overlap_idx = []
for i, idx in zip(torch.sum(labels, dim = 1), idx_test):
    if i > 1:
        count += 1
        overlap_idx.append(idx)

overlap_idx = torch.FloatTensor(overlap_idx).cuda()

model = get_model(args.model, features.size(1), len(labels[0]), 
                args.hidden, args.ssf_dim, args.sp_rate, 
                args.hop, args.heads, args.negative_slope, 
                args.dropout, args.dataset, cuda = True) 

if args.model == "SMN": features_channel, precompute_time = smn_precompute(features, adj, args.hop)
elif args.model == "SGC": features_channel, precompute_time = sgc_precompute(features, adj, args.hop)

if args.model == "SMN" or args.model == "SGC":
    set_seed(args.seed, args.cuda)
    model, f1_train, f1_val,train_time, low_passing_filter = train_regression(model, 
                                                                              features_channel[idx_train], 
                                                                              labels[idx_train], 
                                                                              features_channel[idx_val], 
                                                                              labels[idx_val], 
                                                                              args.epochs, 
                                                                              args.weight_decay, 
                                                                              args.lr, 
                                                                              args.dropout)
    
    f1_test, model_output, low_passing_filter_test, emb = test_regression(model, 
                                                                          features_channel, 
                                                                          labels)

    #----------------------------- Community Search -----------------------------#
    query_nodes = torch.multinomial(idx_test.type(torch.FloatTensor).cuda(), 
                                    50, replacement = False).type(torch.LongTensor)
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
            communities, cs_time = sub_cs(adj, emb, query_nodes, 
                                          ssf, community_size = args.comm_size, 
                                          early_stop = 2, lp_filter = args.ssf)
            
        if args.cs == 'sub_topk':
            communities, cs_time = sub_topk(adj, emb, query_nodes, 
                                            ssf, community_size = args.comm_size, 
                                            early_stop = 2, lp_filter = args.ssf)
            
        cs_f1, cs_jaccard, cs_h_loss = overlap_cs_eval_metrics(communities, 
                                                               query_nodes, 
                                                               target_label_pred, 
                                                               labels[idx_test], 
                                                               args.case)


    elif args.case == 2:
        overlapped_ssf = []
        labels_idx = []

        for i in query_labels:
            ssfs = torch.matmul(i.float().cuda(), low_passing_filter_test) # get the overlapped subspace
            overlapped_ssf.append(ssfs) 
        ssf = torch.stack(overlapped_ssf)
        target_label_pred = query_labels.tolist()

        if args.cs == 'sub_cs':
            communities, cs_time = sub_cs(adj, emb, query_nodes, 
                                          ssf, community_size = int(args.comm_size/5), 
                                          early_stop = 2, lp_filter = args.ssf)
            
        if args.cs == 'sub_topk':
            communities, cs_time = sub_topk(adj, emb, query_nodes, 
                                            ssf, community_size = int(args.comm_size/5), 
                                            early_stop = 2, lp_filter = args.ssf)
            
        cs_f1, cs_jaccard, cs_h_loss = overlap_cs_eval_metrics(communities, 
                                                               query_nodes, 
                                                               target_label_pred, 
                                                               labels[idx_test], 
                                                               args.case)
    #----------------------------- Community Search -----------------------------#

    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s, CS Time: {:.4f}s".format(precompute_time, 
                                                                                                    train_time, 
                                                                                                    precompute_time+train_time, 
                                                                                                    cs_time))    

    print("DATASET: {}".format(args.fb_num))
    print("Training Time: {}".format(round(precompute_time+train_time, 4)))
    print("Query Time: {}".format(round(cs_time, 6)))
    print('Test_acc/f1: {}'.format(round(f1_test, 4)))
    print("Hamming: {}".format(round(cs_h_loss, 4)))
    print("Jaccard: {}".format(round(cs_jaccard, 4)))
    print("F1 Score: {}".format(round(cs_f1, 4)))
