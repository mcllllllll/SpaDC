from torch.utils.data import DataLoader
from .utils import dataset, set_seed, construct_graph_by_coordinate, trans_undirected_graph, dna_1hot_2vec, lap_reg, create_dictionary_mnn, create_dictionary_knn
import torch
from .model import SpaDC
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix
    
def train_SpaDC_bc(adata, ATAC, H3K27ac, seq, hidden_size=32, n_epochs1=100, n_epochs2 = 100, 
                       batch_size=1024, lr=1e-2, lambda1=1e-8, lambda2 = 1e-9, random_seed=40, 
                       save_model=False, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    # seed_everything()
    set_seed(random_seed)

    matrix = torch.FloatTensor(adata.X.todense().transpose()) 

    spatial1 = construct_graph_by_coordinate(ATAC.obsm['spatial'], n_neighbors=6)
    spatial2 = construct_graph_by_coordinate(H3K27ac.obsm['spatial'], n_neighbors=6)

    adj1 = coo_matrix((spatial1['value'], (spatial1['x'],spatial1['y'])), shape=(ATAC.n_obs,ATAC.n_obs),dtype=int)
    adj2 = coo_matrix((spatial2['value'], (spatial2['x'],spatial2['y'])), shape=(H3K27ac.n_obs,H3K27ac.n_obs),dtype=int)
    adj1 = torch.FloatTensor(adj1.todense()).to(device)
    adj2 = torch.FloatTensor(adj2.todense()) .to(device)
    
    adj1 = trans_undirected_graph(adj1)
    adj2 = trans_undirected_graph(adj2)

    #peak × 1344
    seqs_dna = seq['seq']
    seqs_dna = [dna_1hot_2vec(x) for x in seqs_dna]
    seqs_dna = torch.tensor(seqs_dna)

    train_data = dataset(seqs_dna, matrix)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = SpaDC(n_cells=adata.n_obs, hidden_size=hidden_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('Pretrain with SpaDC...')
    for epoch in tqdm(range(1, n_epochs1+1)):
        # train
        model.train()
        train_loss = 0
        n = 0
        for train in train_dataloader:
            data, label = train
            data = data.to(device)
            label = label.to(device)
            output, _ = model(data)

            bce_loss = F.binary_cross_entropy(output, label)
      
            for name,p in model.named_parameters():
                if 'cell_embedding.weight' in name:
                    lap_loss1 = lap_reg(adj1, p[0:ATAC.n_obs, :])  
                    lap_loss2 = lap_reg(adj2, p[ATAC.n_obs:ATAC.n_obs+H3K27ac.n_obs, :])            

            loss = bce_loss + lambda1 * (lap_loss1 + lap_loss2)

            train_loss += loss.item()
            n += 1
            
            optimizer.zero_grad()
            loss.backward()     
            optimizer.step()

        train_loss =  train_loss / n

        print_msg = (f'[{epoch}/{n_epochs1}] ' + 
                     f'train_loss: {train_loss:.5f} ')
        
        print(print_msg)

    cell_embedding = model.get_embedding().to('cpu').detach().numpy()  
    
    adata.obsm['SpaDC'] = cell_embedding
    
    # unique index
    adata.obs.index = [str(i) for i in np.arange(adata.n_obs)]

    anchor_ind = []
    positive_ind = []
    negative_ind = []   
    print('Train with SpaDC_bc...')
    for epoch in tqdm(range(n_epochs1+1, n_epochs1+n_epochs2+1)):               
        if epoch == n_epochs1+1:
            # If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
            # not all points have MNN achors
            mnn_dict = create_dictionary_mnn(adata, use_rep='SpaDC', batch_name='batch', k=50, verbose=0)
            knn_dict = create_dictionary_knn(adata, use_rep='SpaDC', batch_name='batch')

            for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in mnn_dict[batch_pair].keys():
                    anchor_list.append(anchor)
                    positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                    positive_list.append(positive_spot)
                    negative_list.append(knn_dict[anchor])  # select the first knn spot 

                batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

        model.train()  
        train_loss = 0
        n = 0      
        for train in train_dataloader:    
            data, label = train
            data = data.to(device)
            label = label.to(device)
            output, _ = model(data)

            bce_loss = F.binary_cross_entropy(output, label)

            anchor_arr = []
            positive_arr = []
            negative_arr = [] 

            for name,p in model.named_parameters():
                if 'cell_embedding.weight' in name:
                    lap_loss1 = lap_reg(adj1, p[0:ATAC.n_obs, :])  
                    lap_loss2 = lap_reg(adj2, p[ATAC.n_obs:ATAC.n_obs+H3K27ac.n_obs, :]) 
                    anchor_arr = p[anchor_ind,]
                    positive_arr = p[positive_ind,]
                    negative_arr = p[negative_ind,]   

            triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
            tri_loss = triplet_loss(anchor_arr, positive_arr, negative_arr)  

            loss = bce_loss + lambda2 * (lap_loss1 + lap_loss2) + tri_loss

            train_loss += loss.item()
            n += 1
          
            optimizer.zero_grad() 
            loss.backward()    
            optimizer.step()
        
        train_loss = train_loss / n                

        print_msg = (f'[{epoch-n_epochs1}/{n_epochs2}]' + 
                     f'train_loss: {train_loss:.5f}')       
        print(print_msg)    
        
    if save_model == True:
        torch.save(model.state_dict(), 'result/model.pt')     

    adata.obsm['SpaDC_bc'] = model.get_embedding().to('cpu').detach().numpy()  
    return adata







             











        
