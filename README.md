
This home repo contains the implementation for Facebook, MAG, citation networks (Cora, Citeseer, and Pubmed), and Reddit.

### Dependencies
Our implementation works with PyTorch>=1.0.0 Install other dependencies: `$ pip install -r requirement.txt`

### Data
We provide the citation network datasets under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).
Due to space limit, please download reddit dataset from [FastGCN](https://github.com/matenure/FastGCN/issues/9) and put `reddit_adj.npz`, `reddit.npz` under `data/`.

### Usage
```
$ python main.py --dataset facebook
$ python main.py --dataset mag_cs
$ python citation.py --dataset cora
$ python reddit.py --dataset reddit
```
