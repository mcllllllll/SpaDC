import numpy as np
import torch
import scanpy as sc
import pysam
import sys
import anndata as ad
import pandas as pd
from torch import Tensor
import os
import random
import itertools
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import hnswlib
import scipy
from torch.utils.data import Dataset, DataLoader
from model import SpaDC
import pybedtools


class dataset(Dataset):
    def __init__(self, seq, atac) -> None:
        super().__init__()
        self.seq = seq
        self.atac = atac

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, index):
        return self.seq[index], self.atac[index]

def dna_1hot_2vec(seq, seq_len=None):
    """dna_1hot
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
                 rather than sampling.
    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim: seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len,), dtype="int8")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i] = 0
            elif nt == "C":
                seq_code[i] = 1
            elif nt == "G":
                seq_code[i] = 2
            elif nt == "T":
                seq_code[i] = 3
            else:
                seq_code[i] = 4
    return seq_code

def getNClusters(adata, n_cluster, method='leiden', min=0.0, max=3.0, max_steps=30):
    step = 0
    while step < max_steps:
        res = min + ((max - min) / 2)
        if method == 'leiden':
            sc.tl.leiden(adata, resolution=res)
            count = adata.obs['leiden'].nunique()
        elif method == 'louvain':
            sc.tl.louvain(adata, resolution=res)
            count = adata.obs['louvain'].nunique()

        if count > n_cluster:
            max = res
        elif count < n_cluster:
            min = res
        else:
            return (res, adata)
        step += 1
    print("Resolution is not found.")
    return (None, None)

def make_bed_seqs_from_df(input_bed, fasta_file, seq_len, stranded=False):
    """Return BED regions as sequences and regions as a list of coordinate
    tuples, extended to a specified length."""
    """Extract and extend BED sequences to seq_len."""
    fasta_open = pysam.Fastafile(fasta_file)

    seqs_dna = []
    seqs_coords = []

    for i in range(input_bed.shape[0]):
        chrm = input_bed.iloc[i, 0]
        start = int(input_bed.iloc[i, 1])
        end = int(input_bed.iloc[i, 2])
        strand = "+"

        # determine sequence limits
        mid = (start + end) // 2
        seq_start = mid - seq_len // 2
        seq_end = seq_start + seq_len

        # save
        if stranded:
            seqs_coords.append((chrm, seq_start, seq_end, strand))
        else:
            seqs_coords.append((chrm, seq_start, seq_end))
        # initialize sequence
        seq_dna = ""
        # add N's for left over reach
        if seq_start < 0:
            print(
                "Adding %d Ns to %s:%d-%s" % (-seq_start, chrm, start, end),
                file=sys.stderr,
            )
            seq_dna = "N" * (-seq_start)
            seq_start = 0

        # get dna
        seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()

        # add N's for right over reach
        if len(seq_dna) < seq_len:
            print(
                "Adding %d Ns to %s:%d-%s" % (seq_len - len(seq_dna), chrm, start, end),
                file=sys.stderr,
            )
            seq_dna += "N" * (seq_len - len(seq_dna))
        # append
        seqs_dna.append(seq_dna)
    fasta_open.close()
    return seqs_dna, seqs_coords


def construct_graph_by_coordinate(cell_position, n_neighbors=6):
    # print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def lap_reg(A, weight):
    degree = A.sum(dim=1)
    D = torch.diag(degree)
    D_sqrt_inv = torch.diag(1 / degree.sqrt())
    L = D_sqrt_inv @ (D - A) @ D_sqrt_inv

    return torch.trace(torch.mm(torch.mm(weight.t(), L), weight))

def trans_undirected_graph(graph: Tensor):
    a = graph.T > graph
    return graph + graph.T * a - graph * a


def is_undirected_graph(graph: Tensor):
    A = graph - graph.T
    return (torch.any(A) == 0).item()


def find_overlapping_coordinates(coord_list, coord, maxgap=0):
    coord_list = [x.split('-') for x in coord_list]
    coord = coord.split('-')
    result = []
    for i, c in enumerate(coord_list):
        if c[0] == coord[0] and c[2] > coord[1] and coord[2] > c[1]:
            result.append(i)
    return result

def set_seed(seed):   
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# Modified from https://github.com/lkmklsmn/insct
def create_dictionary_mnn(adata, use_rep, batch_name, k = 50, save_on_disk = True, approx = True, verbose = 1, iter_comb = None):

    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    mnns = dict()

    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[key_name1] = {} # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        if(verbose > 0):
            print('Processing datasets {}'.format((i, j)))

        new = list(cells[j])
        ref = list(cells[i])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk = save_on_disk, approx = approx)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            mnns[key_name1][key]= names
    return(mnns)

def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def mnn(ds1, ds2, names1, names2, knn = 20, save_on_disk = True, approx = True):
    if approx: 
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

def create_dictionary_knn(adata, use_rep, batch_name):
    section_ids = np.array(adata.obs['batch'].unique())
    knn_dict = dict()    
    for batch_id in range(len(section_ids)):
        batch_embed = adata.obsm[use_rep][adata.obs[batch_name] == section_ids[batch_id]]
        batch_cellname = adata.obs_names[adata.obs[batch_name] == section_ids[batch_id]].values
        nbrs = NearestNeighbors(n_neighbors=2).fit(batch_embed)
        _, indices = nbrs.kneighbors(batch_embed)
        for i in range(len(indices)):
            knn_dict[batch_cellname[indices[i, 0]]] = batch_cellname[indices[i, 1]]
        
    return knn_dict

def get_denoise_adata(adata, seq, model_state_dict, hidden_size=32, batch_size=1024):  
    adata.X[adata.X != 0] = 1
    atac = torch.FloatTensor(adata.X.todense().transpose())

    seqs_dna = seq['seq']
    seqs_dna = [dna_1hot_2vec(x) for x in seqs_dna]
    seqs_dna = torch.tensor(seqs_dna)

    train_data = dataset(seqs_dna, atac)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    model = SpaDC(atac.shape[1], hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_state_dict))

    denoise = torch.tensor([])

    model.eval()
    with torch.no_grad():
        for train in train_dataloader:
            data, _ = train 
            output, _ = model(data)

            denoise = torch.cat((denoise, output), axis=0)

    adata_denoise = ad.AnnData(np.array(denoise).transpose())
    return adata_denoise

def batch_entropy_mixing_score(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    """
    Calculate batch entropy mixing score

    Algorithm
    ---------
         * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
         * 2. Define 100 nearest neighbors for each randomly chosen cell
         * 3. Calculate the mean mixing entropy as the mean of the regional entropies
         * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.
     
     Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.

    Returns
    -------
    Batch entropy mixing score
    """
#     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i]/P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i]/P[i])/a
            entropy = entropy - adapt_p[i]*np.log(adapt_p[i]+10**-8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
            P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
                                          [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))

def chr_split(x):
    chr_special = {'GL456233.1':'chrX_GL456233_random', 'GL456212.1':'chr1_GL456212_random', 'JH584304.1':'chrUn_JH584304', 
                   'GL456216.1':'chr4_GL456216_random', 'JH584292.1':'chr4_JH584292_random', 'JH584295.1':'chr4_JH584295_random'}
    chr, start, end = x.split('-')
    if chr in chr_special:
        chr = chr_special[chr]
    return chr, start, end

def integrate_data(adata1, adata2, save_folder, fasta_file, seq_len):
    index = adata1.var_names
    index = pd.DataFrame(chr_split(x) for x in index)
    index.to_csv(save_folder+'/adata1.bed',sep='\t',header=False, index=False)

    index = adata2.var_names
    index = pd.DataFrame(chr_split(x) for x in index)
    index.to_csv(save_folder+'/adata2.bed',sep='\t',header=False, index=False)

    adata1_bed = pybedtools.BedTool(save_folder+'/adata1.bed')
    adata2_bed =   pybedtools.BedTool(save_folder+'/adata2.bed')                  

    overlap = adata1_bed.intersect(adata2_bed,wo=True)
    overlap.moveto(save_folder+'/overlap.bed')

    df = pd.read_csv(save_folder+'/overlap.bed', sep='\t', header=None)

    df['start'] = ''
    df['end'] = ''

    df['start'] = np.where(df.iloc[:, 1] <= df.iloc[:, 4], df.iloc[:, 1], df.iloc[:, 4])
    df['end'] = np.where(df.iloc[:, 2] >= df.iloc[:, 5], df.iloc[:, 2], df.iloc[:, 5])
    index = df.loc[:, [0, 'start', 'end']]

    seq, _ = make_bed_seqs_from_df(index, fasta_file, seq_len)

    file = open(save_folder+'/overlap_seqs.txt','a')
    file.write('seq\n')
    for i in range(len(seq)):
        s = seq[i] + '\n'
        file.write(s)
    file.close()

    adata1_matrix = np.array(adata1.X.todense().transpose())
    adata2_matrix = np.array(adata2.X.todense().transpose())

    adata1_var_name= []
    adata2_var_name = []

    for i in range(len(df)):
        adata1_var_name.append('-'.join('%s'%i for i in df.iloc[i, 0:3]))
        adata2_var_name.append('-'.join('%s'%i for i in df.iloc[i, 3:6]))

    adata1_index =  list(adata1.var_names.get_indexer(adata1_var_name))
    adata2_index  =  list(adata2.var_names.get_indexer(adata2_var_name))

    new_matrix = np.hstack((adata1_matrix[adata1_index], adata2_matrix[adata2_index]))
    new_matrix[np.where(new_matrix != 0)] = 1
    new_matrix = scipy.sparse.csr_matrix(new_matrix)

    adata = ad.AnnData(new_matrix.transpose())
    adata.obs['batch'] = np.array(['0'] * adata1.n_obs + ['1'] * adata2.n_obs)
    adata.var['chr'] = index.loc[:, 0].values
    adata.var['start'] = index.loc[:, 'start'].values
    adata.var['end'] = index.loc[:, 'end'].values
    adata.obs.index = np.concatenate((adata1.obs_names, adata2.obs_names),axis=0)

    sc.write(save_folder+'/integrate.h5ad', adata)






# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement. 
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#             path (str): Path for the checkpoint to be saved to.
#                             Default: 'checkpoint.pt'
#             trace_func (function): trace print function.
#                             Default: print            
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.path = path
#         self.trace_func = trace_func
#     def __call__(self, val_loss, model):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), self.path)
#         self.val_loss_min = val_loss



















