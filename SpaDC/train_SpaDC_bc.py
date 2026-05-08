from torch.utils.data import DataLoader
from .utils import dataset, set_seed, construct_graph_by_coordinate, trans_undirected_graph, dna_1hot_2vec, lap_reg, create_dictionary_mnn
import torch
from .model import SpaDC
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix
import os
    
def train_SpaDC_bc(integrate, adata1, adata2, seq, hidden_size=32, n_epochs1=100, n_epochs2=100, 
                       batch_size=1024, lr=1e-2, lambda1=1e-7, lambda2=1e-7, random_seed=40, 
                       save_model=True, out_dir='', device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    # seed_everything()
    set_seed(random_seed)

    matrix = torch.FloatTensor(integrate.X.todense().transpose()) 

    spatial1 = construct_graph_by_coordinate(adata1.obsm['spatial'], n_neighbors=6)
    spatial2 = construct_graph_by_coordinate(adata2.obsm['spatial'], n_neighbors=6)

    adj1 = coo_matrix((spatial1['value'], (spatial1['x'],spatial1['y'])), shape=(adata1.n_obs,adata1.n_obs),dtype=int)
    adj2 = coo_matrix((spatial2['value'], (spatial2['x'],spatial2['y'])), shape=(adata2.n_obs,adata2.n_obs),dtype=int)
    adj1 = torch.FloatTensor(adj1.todense()).to(device)
    adj2 = torch.FloatTensor(adj2.todense()) .to(device)
    
    adj1 = trans_undirected_graph(adj1)
    adj2 = trans_undirected_graph(adj2)

    #peak × 1344
    seqs_dna = seq['seq']
    seqs_dna = [dna_1hot_2vec(x) for x in seqs_dna]
    seqs_dna = torch.tensor(np.array(seqs_dna))

    train_data = dataset(seqs_dna, matrix)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = SpaDC(n_cells=integrate.n_obs, hidden_size=hidden_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('Pretrain with SpaDC...')
    pbar = tqdm(range(1, n_epochs1+1))
    for epoch in pbar:
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
                    lap_loss1 = lap_reg(adj1, p[0:adata1.n_obs, :])  
                    lap_loss2 = lap_reg(adj2, p[adata1.n_obs:adata1.n_obs+adata2.n_obs, :])            

            loss = bce_loss + lambda1 * (lap_loss1 + lap_loss2)

            train_loss += loss.item()
            n += 1
            
            optimizer.zero_grad()
            loss.backward()     
            optimizer.step()

        train_loss =  train_loss / n

        pbar.set_postfix(
            train_loss=f'{train_loss:.6f}'
        )

    integrate.obsm['SpaDC_raw'] = model.get_embedding().to('cpu').detach().numpy() 

    # unique index
    integrate.obs.index = [str(i) for i in np.arange(integrate.n_obs)]
    section_ids = np.array(integrate.obs['batch'].unique())
    
    anchor_ind = []
    positive_ind = []
    negative_ind = []    
    
    print('Train with SpaDC_bc...')
    pbar = tqdm(range(n_epochs1+1, n_epochs1+n_epochs2+1))
    for epoch in pbar:               
        if epoch % 20 == 1 or epoch == n_epochs1+1:
            integrate.obsm['SpaDC'] = model.get_embedding().to('cpu').detach().numpy() 

            mnn_dict = create_dictionary_mnn(integrate, use_rep='SpaDC', batch_name='batch', k=50, verbose=0)
            for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                batchname_list = integrate.obs['batch'][mnn_dict[batch_pair].keys()]
                embedding_dict = dict(zip(integrate.obs_names, integrate.obsm['SpaDC']))

                cellname_by_batch_dict = dict()
                for batch_id in range(len(section_ids)):
                    cellname_by_batch_dict[section_ids[batch_id]] = integrate.obs_names[
                        integrate.obs['batch'] == section_ids[batch_id]].values

                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in mnn_dict[batch_pair].keys():
                    anchor_list.append(anchor)
                    positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                    positive_list.append(positive_spot)

                    negative_pool = np.setdiff1d(cellname_by_batch_dict[batchname_list[positive_spot]], mnn_dict[batch_pair][anchor])

                    # hard
                    d_negs = [np.linalg.norm(embedding_dict[anchor] - embedding_dict[neg]) for neg in negative_pool]
                    nearest_idx = np.argmin(d_negs)
                    negative_list.append(negative_pool[nearest_idx])

                batch_as_dict = dict(zip(list(integrate.obs_names), range(0, integrate.shape[0])))
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

        model.train()  
        train_loss = 0
        bce_loss_a = 0
        lap_loss_a = 0
        tri_loss_a = 0
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
                    lap_loss1 = lap_reg(adj1, p[0:adata1.n_obs, :])  
                    lap_loss2 = lap_reg(adj2, p[adata1.n_obs:adata1.n_obs+adata2.n_obs, :]) 
                    anchor_arr = p[anchor_ind,]
                    positive_arr = p[positive_ind,]
                    negative_arr = p[negative_ind,]   

            triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
            tri_loss = triplet_loss(anchor_arr, positive_arr, negative_arr)  

            loss = bce_loss + lambda2 * (lap_loss1 + lap_loss2) + tri_loss

            bce_loss_a += bce_loss.item()
            lap_loss_a += (lap_loss1 + lap_loss2).item()
            tri_loss_a += tri_loss.item()

            train_loss += loss.item()
            n += 1
          
            optimizer.zero_grad() 
            loss.backward()   
            optimizer.step()
        
        train_loss = train_loss / n  
        bce_loss_a = bce_loss_a / n 
        lap_loss_a = lap_loss_a / n
        tri_loss_a = tri_loss_a / n

        pbar.set_postfix(
            train_loss=f'{train_loss:.6f}'
        )   
          
    if save_model == True:
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))     

    integrate.obsm['SpaDC_bc'] = model.get_embedding().to('cpu').detach().numpy()  
    return integrate