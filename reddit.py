from time import perf_counter
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_reddit_data, set_seed, smn_precompute, sub_cs, sub_topk
from metrics import accuracy, cs_eval_metrics, comm_ari
from args_reddit import get_citation_args
from models import SGC, SMN

# Args
args = get_citation_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)

adj, train_adj, features, labels, idx_train, idx_val, idx_test = load_reddit_data(args.normalization, 
                                                                                  model = args.model,
                                                                                  cuda=args.cuda)
print("Finished data loading.")
model = SMN(features.size(1), labels.max().item()+1, 
                  args.hidden, args.ssf_dim, args.sp_rate, args.hop, args.heads, args.negative_slope, args.dropout, args.dataset) 
if args.cuda: model.cuda()
features_channel, precompute_time = smn_precompute(features, adj, args.hop)

print("{:.4f}s".format(precompute_time))

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
        class_loss = F.cross_entropy(output, train_labels)
        spatial_loss = F.cross_entropy(spatial_loss, train_labels)
        total_loss = 0.5 * class_loss / sigma[0]**2 + 0.5 * spatial_loss / sigma[1]**2 
        # Fuse two losses
        total_loss.backward()
        optimizer.step()
        acc_train = accuracy(output, train_labels)

    train_time = perf_counter()-t

    with torch.enable_grad():
        model.eval()
        output, low_passing_filter, emb, spatial_loss, sigma = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_train, acc_val, train_time, low_passing_filter

def test_regression(model, test_features, test_labels):
    model.eval()
    return (accuracy(model(test_features)[0], test_labels), 
            model(test_features)[0], 
            model(test_features)[1].transpose(0, 1), 
            model(test_features)[2],
            model(test_features)[-1])


set_seed(args.seed, args.cuda)
model, acc_train, acc_val, train_time, low_passing_filter = train_regression(model, features_channel[idx_train], labels[idx_train], 
                                                                    features_channel[idx_val], labels[idx_val],
                                                                    args.epochs, args.weight_decay, args.lr, args.dropout)
acc_test, model_output, low_passing_filter_test, emb, sigma = test_regression(model, 
                                                                            features_channel, 
                                                                            labels)

# Community Search 
query_nodes = torch.multinomial(torch.from_numpy(idx_test).type(torch.FloatTensor).cuda(), 50, replacement = False).tolist()
query_labels = torch.LongTensor(torch.argmax(model_output[torch.LongTensor(query_nodes)], dim = 1).cpu())

if args.cs == 'sub_cs':
    communities, cs_time = sub_cs(adj, emb, query_nodes, low_passing_filter_test[query_labels], 
                    community_size = args.comm_size, early_stop = 8, lp_filter = args.ssf)
if args.cs == 'sub_topk':
    communities, cs_time = sub_topk(adj, emb, query_nodes, low_passing_filter_test[query_labels], 
                    community_size = args.comm_size, early_stop = 8, lp_filter = args.ssf)
cs_acc, cs_f1 = cs_eval_metrics(communities, query_nodes, labels)

print("Training Time: {}".format(round(precompute_time+train_time, 4)))
print("Query Time: {}".format(round(cs_time, 6)))
print("Accuracy: {}".format(round(cs_acc, 4)))
print("F1 Score: {}".format(round(cs_f1, 4)))
