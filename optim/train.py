import time,tqdm
import numpy as np
import torch
import dgl
from optim.loss import loss_function,init_center,get_radius
from HSA import HSA
from utils import fixed_graph_evaluate


# G = dgl.heterograph({
#     ('user','a','user') # 0-hop
#     ('user','b','user') # 1hop e2r 
#     ('user','e','user')  #common followers   
#     ('user','c','user') #topical relationship
# },idtype=torch.int64)
# ebac


def train(args,data,path):
    device = torch.device('cpu')
    checkpoints_path=path
    num_classes=2

    model = HSA(relations=[['a'],['b'],['c'],['e']],
    # model = HSA(relations=[['b'],['a']],
                    edge_weights=data['edge_weights'],
                    in_size=data['features'].shape[1],
                    hidden_size=32,
                    out_size=num_classes,
                    num_heads=(2))

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    data_center= init_center(args,data['g'],data['features'], model)
    radius=torch.tensor(0, device=device)# radius R initialized with 0 by default.
    best_f1=0
    best_epoch=0

    model.train()
    for epoch in range(args.n_epochs):
        print ('@'*30,epoch,'@'*30)

        outputs= model(data['g'],data['features'])
        loss,dist,_=loss_function(args.nu, data_center,outputs,radius,data['train_mask'])
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        radius.data=torch.tensor(get_radius(dist, args.nu), device=device)

        auc,ap,f1,acc,precision,recall,val_loss = fixed_graph_evaluate(args,checkpoints_path, model, data_center,data,radius,data['val_mask'],test=False)

        if(f1>best_f1):
            best_f1=f1
            best_epoch=epoch

        print("Epoch {:05d} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f}". format(epoch, loss.item(), val_loss.item(), auc, ap))
        print(f'Val f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')

        auc_t,ap_t,f1_t,acc_t,precision_t,recall_t,loss_t = fixed_graph_evaluate(args,checkpoints_path,model, data_center,data,radius,data['test_mask'],test=True)
        print("Test AUROC {:.4f} | Test AUPRC {:.4f}".format(auc_t,ap_t))
        print(f'Test f1:{round(f1_t,4)},acc:{round(acc_t,4)},pre:{round(precision_t,4)},recall:{round(recall_t,4)}')
        
    
    # model = HSA(relations=[['a'],['b'],['c'],['e']],
    # model = HSA(relations=[['b']],
    #                 in_size=data['features'].shape[1],
    #                 hidden_size=32,
    #                 out_size=num_classes,
    #                 num_heads=(3))
    # PATH='./model_checkpoints/'+str(best_epoch)+'_'+str(round(best_f1,2))
    # print (PATH)
    # model.load_state_dict(torch.load(PATH))
    # auc,ap,f1,acc,precision,recall,loss = fixed_graph_evaluate(args,checkpoints_path,model, data_center,data,radius,data['test_mask'],test=True)
    # print("Test AUROC {:.4f} | Test AUPRC {:.4f}".format(auc,ap))
    # print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    return model
