# SMN
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


#### Authors: 
* [Qing Sima]
* [Xiaoyang Wang]
* [Jianke Yu]
* [Wenjie Zhang]
* [Ying Zhang]
* [Xuemin Lin]


### Overview
This repo contains an example implementation of the Simplified Multi-hop Attention Network (SMN) model, 
described in the paper [Deep Overlapping Community Search via Subspace Embedding].

A Simplified Multi-hop Attention Network (SMN) is proposed, accompanied
by sparse subspace embedding techniques. SMN is the first study of deep overlapping community search.
For an illustration, ![](./model.jpg "SMN")

This home repo contains the implementation for citation networks (Cora, Citeseer, and Pubmed) , Facebook, MAG and Reddit.

### Dependencies
Our implementation works with PyTorch>=1.0.0 Install other dependencies: `$ pip install -r requirement.txt`

### Data
We provide the citation network datasets under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).
Due to space limit, please download reddit dataset from [FastGCN](https://github.com/matenure/FastGCN/issues/9) and put `reddit_adj.npz`, `reddit.npz` under `data/`.

### Usage
```
$ python citation.py --dataset cora
$ python facebook.py --dataset facebook
$ python mag_data.py --dataset mag_cs
$ python reddit.py --dataset reddit
```


### Acknowledgement
This repo is modified from [sgc](https://github.com/Tiiiger/SGC).

