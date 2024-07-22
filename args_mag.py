import argparse
import torch

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--heads', type=int, default=2,
                        help='Number of heads for attention.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--ssf', type=str, default="soft",
                        choices=["hard", "soft", "none"],
                        help='Apply SSF or not.')
    parser.add_argument('--loss', type=str, default="asl",
                        choices=["focal", "asl", "class", "spatial", "both"],
                        help='Loss to use.')
    parser.add_argument('--ssf_dim', type=int, default=64,
                        help='Dimension of sparse subspace filters.')
    parser.add_argument('--sp_rate', type=float, default=0.7,
                        help='SSF Sparsity rate.')
    parser.add_argument('--lammda', type=float, default=0.1,
                        help='lammda control penalty.')    
    parser.add_argument('--gamma', type=float, default=3.0,
                        help='gamma control Focusing parameter.')      
    parser.add_argument('--alpha', type=float, default=0.25,
                        help='alpha control Balance parameter.')    
    parser.add_argument('--gamma_neg', type=float, default=2.0,
                        help='gamma_neg control Focusing parameter.')    
    parser.add_argument('--gamma_pos', type=float, default=2.0,
                        help='gamma_pos control Focusing parameter.')    
    parser.add_argument('--comm_size', type=int, default=1000,
                        help='size of community.')
    parser.add_argument('--cs', type=str, default="sub_cs",
                        choices=["sub_cs", "sub_topk"],
                        help='CS algorithm to use.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="mag_cs",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="SMN",
                        choices=["SGC", "GCN", 'SMN'],
                        help='model to use.')
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--case', type=int, default=1,
                        choices=[1, 2, 3], help='which scenarios are we looking at')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                        choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                            'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                        help='Normalization method for the adjacency matrix.')
    parser.add_argument('--hop', type=int, default=4,
                        help='degree of the approximation.') # k-hop
    parser.add_argument('--fb_num', type=int, default=0,
                        choices=[0, 107, 348, 414, 686], help='FB num')
    parser.add_argument('--negative_slope', type=float, default=0.2,
                        help='negative_slope for LeakyReLU.')
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    parser.add_argument('--experiment', type=str, default="base-experiment",
                        help='feature-type')
    parser.add_argument('--inductive', action='store_true', default=False,
                        help='inductive training.')
    parser.add_argument('--test', action='store_true', default=False,
                        help='inductive training.')
                        
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')

    args, _ = parser.parse_known_args()

    if args.dataset in ('facebook',) :
        args.loss = 'focal'
        args.hidden = 128
        args.ssf_dim = 64
        args.lr = 0.02
        args.hop = 4
        args.comm_size = 30

    elif args.dataset.startswith('mag_'):
        args.hidden = 128
        args.ssf_dim = 64
        args.lr = 0.02
        args.hop = 4

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(6)
    return args
