import numpy as np
import scipy.sparse as sp
import torch

def aug_normalized_adjacency(adj, model):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)) # get degree
    row_sum[np.isnan(row_sum)] = 0.
    d_inv_sqrt = np.power(row_sum, -0.5).flatten() # D^{-1/2}
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo() # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2 
                                                            # Save in cooridinate format
def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

def row_normalize(mx):
    """Row-normalize sparse matrix""" # normalize the row to get rowsum = 1
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
