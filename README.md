
This home repo contains the implementation for Facebook, MAG, citation networks (Cora, Citeseer, and Pubmed), and Reddit.

### Dependencies
Our implementation works with PyTorch>=1.0.0 Install other dependencies: `$ pip install -r requirement.txt`

### Data
We provide the citation network datasets under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).
Due to space limit, please download reddit dataset from [FastGCN](https://github.com/matenure/FastGCN/issues/9) and put `reddit_adj.npz`, `reddit.npz` under `data/`.

### Usage
```
# Reproduce all results in the main experiment table
./exp.sh

# training with default hyperparameters (e.g. OCS on FACEBOOK)
$ python main.py --dataset facebook

# training with default hyperparameters (e.g. OCS on MAG: Computer Science)
$ python main.py --dataset mag_cs

# training with default hyperparameters (e.g. Disjoint datasets, CORA)
$ python citation.py --dataset cora

# training with default hyperparameters (e.g. Disjoint datasets, Reddit)
$ python reddit.py --dataset reddit
```
### OCS vs OCIS
'--case 1' = OCS, 

'--case 2' = OCIS

### Model and Training Parameters
--no-cuda: Disables CUDA, using CPU instead (default: False).

--seed: Sets a random seed for reproducibility.

--epochs: Specifies the number of training epochs.

--heads: Sets the number of attention heads for multi-head models.

--lr: Initial learning rate.

--weight_decay: Weight decay for L2 regularization.

### Model Design Choices
--hidden: Number of hidden units in the model.

--ssf: Method to apply sparsity, options include hard, soft, or none.

--loss: Loss function to use, with several options including focal and spatial losses.

--ssf_dim: Dimensionality of sparse subspace filters.

--sp_rate: Sparsity rate for sparse subspace filters.

--lammda: Penalty coefficient to control the regularization term.

--gamma, --alpha, --gamma_neg, --gamma_pos: Parameters controlling the loss function's focus and balance aspects.

### Community and Feature Specifications
--comm_size: Sets the community size for the search.

--cs: Community Search (CS) algorithm to apply; options include sub_cs and sub_topk.

--dropout: Dropout rate to prevent overfitting.

--dataset: Dataset to use for training/testing (default: mag_cs).

--model: Model architecture to use, including options such as SGC, GCN, and SMN.

--feature: Feature type for input processing, with options like mul, cat, and adj.


### Scenarios and Problem Settings
--case: Specifies the task scenario (1 for OCS, 2 for OCIS).

--normalization: Normalization strategy for the adjacency matrix, offering methods like NormLap, Lap, and AugNormAdj.

--hop: Degree of approximation, indicating k-hop adjacency.

--fb_num: Facebook dataset option, specifying a subset (0, 107, 348, 414, or 686).
