from sklearn.metrics import adjusted_rand_score, f1_score, accuracy_score, recall_score, jaccard_score, normalized_mutual_info_score, hamming_loss
import scipy.sparse as sp
import numpy as np
import torch
import warnings

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def comm_ari(output, labels):
    preds = output.max(1)[1].type_as(labels)
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    ari = adjusted_rand_score(labels, preds)
    return ari

def cs_accuracy(output, labels):
    return accuracy_score(labels,output).round(4)

def f1(output, labels):
    f1 = f1_score(labels, output, average='binary')
    recall = recall_score(labels, output, average='binary')
    return  f1.round(4)#, recall.round(4)

def multilabel_metrics(output, labels):
    output = sp.coo_matrix(output.cpu().detach()).todense()
    labels = sp.coo_matrix(labels.cpu().detach()).todense()
    nmi = normalized_mutual_info_score(labels, output)
    hamming_loss = hamming_loss(labels, output)
    jaccard_score = jaccard_score(labels, output, average='micro')
    return  nmi.round(4), hamming_loss.round(4), jaccard_score.round(4)


def cs_eval_metrics(communities, query_nodes, labels):
    true = [1] * (len([element for sublist in communities for element in sublist]) - len(query_nodes))
    pred = []
    for community , query in zip(communities, query_nodes):
        pred.append(np.equal(labels[torch.LongTensor(community[1:])].tolist(), int(labels[query])) * 1)

    pred = [element for sublist in pred for element in sublist]
    jaccard = round(jaccard_score(true, pred, average='binary'), 4)
    f1 = f1_score(true, pred, average='binary').round(4)
    return jaccard, f1

def overlap_cs_eval_metrics(communities, query_nodes, target_label_pred, labels, case):
    if case == 2:
        true = []
        pred = []
        for i in range(len(target_label_pred)): # consolidate target labels for all nodes in communities
            for j in range(len(communities[i])-1):
                true.append(target_label_pred[i])
        for community , one_label in zip(communities, target_label_pred):
            pred.append(torch.mul(labels[torch.LongTensor(community[1:])].cpu(), torch.LongTensor(one_label)).tolist())
        
        pred = [element for sublist in pred for element in sublist]

        jaccard = jaccard_score(true, pred, average='samples', zero_division=0).round(4)
        f1 = f1_score(true, pred, average='samples', zero_division=0).round(4)
    else:
        true = [1] * (len([element for sublist in communities for element in sublist]) - len(query_nodes))
        pred = []
        for community , one_label in zip(communities, target_label_pred):
            pred.append(torch.matmul(labels[torch.LongTensor(community[1:])].cpu(), torch.LongTensor(one_label)).tolist() * 1)

        pred = [element for sublist in pred for element in sublist]
        jaccard = jaccard_score(true, pred, average='binary', zero_division=0).round(4)
        f1 = f1_score(true, pred, average='binary', zero_division=0).round(4)

    return jaccard, f1
