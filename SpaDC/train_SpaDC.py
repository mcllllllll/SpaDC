from torch.utils.data import DataLoader
from .utils import dataset, set_seed, construct_graph_by_coordinate, trans_undirected_graph, dna_1hot_2vec, lap_reg
import torch
from .model import SpaDC
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
    
def train_SpaDC(adata, seq, hidden_size=32, n_epochs=100, batch_size=1024, 
                    lr=1e-2, lambda1=1e-7, random_seed=40, show_loss=False, save_model=False, 
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    # seed_everything()
    set_seed(random_seed)

    # Binarization
    adata.X[adata.X != 0] = 1
    atac = torch.FloatTensor(adata.X.todense().transpose())

    spatial = construct_graph_by_coordinate(adata.obsm['spatial'], n_neighbors=6)

    adj = coo_matrix((spatial['value'], (spatial['x'],spatial['y'])), shape=(adata.n_obs,adata.n_obs),dtype=int)
    adj = torch.FloatTensor(adj.todense()).to(device)   
    adj = trans_undirected_graph(adj)

    # peak × 1344
    seqs_dna = seq['seq']
    seqs_dna = [dna_1hot_2vec(x) for x in seqs_dna]
    seqs_dna = torch.tensor(seqs_dna)
    X_train, X_valid, y_train, y_valid = train_test_split(seqs_dna, atac, test_size=0.2, shuffle=True, random_state=0)

    train_data = dataset(X_train, y_train)
    valid_data = dataset(X_valid, y_valid)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    model = SpaDC(atac.shape[1], hidden_size=hidden_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list=[]
    valid_loss_list=[]

    for epoch in tqdm(range(1, n_epochs+1)):
        # train
        model.train()
        train_loss = 0
        n1 = 0
        for train in train_dataloader:
            data, label = train
            data = data.to(device)
            label = label.to(device)

            output, _ = model(data)

            bce_loss = F.binary_cross_entropy(output, label)
        
            for name, p in model.named_parameters():
                if 'cell_embedding.weight' in name:
                    lap_loss = lap_reg(adj, p)         
            
            loss = bce_loss + lambda1 * lap_loss

            train_loss += loss.item()
            n1 += 1
            
            optimizer.zero_grad()
            loss.backward()     
            optimizer.step()
        
        train_loss =  train_loss / n1
        train_loss_list.append(train_loss)

        # valid
        model.eval()
        valid_loss = 0
        n2 = 0
        with torch.no_grad():
            for valid in valid_dataloader:
                data, label = valid
                data = data.to(device)
                label = label.to(device)

                output, _ = model(data)

                bce_loss = F.binary_cross_entropy(output, label)

                for name, p in model.named_parameters():
                    if 'cell_embedding.weight' in name:                      
                        lap_loss = lap_reg(adj, p)
                
                loss = bce_loss + lambda1 * lap_loss

                valid_loss += loss.item()
                n2 += 1
            
            valid_loss = valid_loss / n2                
            valid_loss_list.append(valid_loss)

        print_msg = (f'[{epoch}/{n_epochs}] ' + 
                     f'train_loss: {train_loss:.6f} ' + 
                     f'valid_loss: {valid_loss:.6f}')     
        print(print_msg)
    
    if show_loss == True:
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1,len(train_loss_list)+1),train_loss_list, label='Train Loss')
        plt.plot(range(1,len(valid_loss_list)+1),valid_loss_list,label='Valid Loss')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.xlim(0, len(train_loss_list)+1) 
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig('figures/loss.png', bbox_inches='tight')

    if save_model == True:
        torch.save(model.state_dict(), 'result/model.pt')

    cell_embedding = model.get_embedding().to('cpu').detach().numpy()

    adata.obsm['SpaDC'] = cell_embedding
    return adata











             





