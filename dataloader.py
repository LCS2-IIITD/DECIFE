import torch
import torch.utils.data
import numpy as np
import pickle

def loader(args):
    # load and preprocess dataset
    with open('data/graph.pkl', 'rb') as f:
        graph, features, labels, _, _, _ = pickle.load(f)

    #load edge_weights
    with open('data/edge_weights.pkl','rb') as f:
            e_wgts =pickle.load(f)

    labels_inv=np.zeros(len(labels),dtype=int)
    labels_inv[np.where(labels.numpy()==1)]=0
    labels_inv[np.where(labels.numpy()==0)]=1
    labels = torch.tensor(labels_inv)

    train_mask,val_mask,test_mask=labels2mask(labels.numpy())
    print ("Labels: ",labels)
    # print (np.unique(labels,return_counts=True))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    print("""----Statistics------'
    Graph info: %s
    Graph edges: %d
    Graph nodes: %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (graph, graph.number_of_edges(), graph.number_of_nodes(),
          train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item()))


    datadict={'g':graph,'features':features,'edge_weights':e_wgts,'labels':labels,
    'train_mask':train_mask,'val_mask':val_mask,'test_mask': test_mask}

    return datadict


def labels2mask(labels):
    collusive_idx=np.where(labels==0)[0]
    noise_idx=np.where(labels==1)[0]

    train_mask=np.ones(len(labels),dtype='bool')
    val_mask,test_mask=np.zeros(labels.shape,dtype='bool'),np.zeros(labels.shape,dtype='bool')

    print (len(collusive_idx),len(noise_idx))

    start=0.0
    end=start+0.2
    ratio=1 

    n_col,n_gen=len(collusive_idx),len(noise_idx)
   
    #80 percent collusive users for training
    train_mask[noise_idx]=0
    train_mask[collusive_idx[int(start*n_col):int(end*n_col)]]=0
    print (np.unique(train_mask,return_counts=True))

    #5percent collusive and all noise(genuine) for validation
    val_mask[collusive_idx[int(start*n_col):int(start*n_col)+(ratio*n_gen)]]=1
    val_mask[noise_idx]=1

    #rest 5 percent collusive for test and all genuine for validation
    test_mask[collusive_idx[int(start*n_col)+(ratio*n_gen):int(start*n_col)+(2*ratio*n_gen)]]=1
    test_mask[noise_idx]=1

    return train_mask,val_mask,test_mask  

