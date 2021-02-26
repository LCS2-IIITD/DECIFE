from sklearn.metrics import confusion_matrix,classification_report,f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve
import torch
from optim.loss import loss_function,anomaly_score
import numpy as np
import torch.nn as nn
import time

def fixed_graph_evaluate(args,path,model, data_center,data,radius,mask,test=None):
    model.eval()
    with torch.no_grad():
        labels = data['labels'][mask]
        loss_mask=mask.bool() & data['labels'].bool()
        outputs= model(data['g'],data['features'])  

        _,scores=anomaly_score(data_center,outputs,radius,mask)
        loss,_,_=loss_function(args.nu,data_center,outputs,radius,loss_mask)
 
        labels=labels.cpu().numpy()
        print ('val distribution: ',np.unique(labels,return_counts=True))
        scores=scores.cpu().numpy()

        threshold=0
        pred=thresholding(scores,threshold)

        auc=roc_auc_score(labels, scores)
        ap=average_precision_score(labels, scores)

        print (confusion_matrix(labels,pred))
        print (classification_report(labels,pred))

        acc=accuracy_score(labels,pred)
        recall=recall_score(labels,pred)
        precision=precision_score(labels,pred)
        f1=f1_score(labels,pred)

        return auc,ap,f1,acc,precision,recall,loss

def thresholding(recon_error,threshold):
    ano_pred=np.zeros(recon_error.shape[0])
    for i in range(recon_error.shape[0]):
        if recon_error[i]>threshold:
            ano_pred[i]=1
    return ano_pred

def baseline_evaluate(datadict,y_pred,y_score,val=True):
    
    if val==True:
        mask=datadict['val_mask']
    if val==False:
        mask=datadict['test_mask']

    auc=roc_auc_score(datadict['labels'][mask],y_score)
    ap=average_precision_score(datadict['labels'][mask],y_score)
    acc=accuracy_score(datadict['labels'][mask],y_pred)
    recall=recall_score(datadict['labels'][mask],y_pred)
    precision=precision_score(datadict['labels'][mask],y_pred)
    f1=f1_score(datadict['labels'][mask],y_pred)
    cm=confusion_matrix(datadict['labels'][mask],y_pred,labels=[0,1])
    cr=classification_report(datadict['labels'][mask],y_pred,labels=[0,1])

    print ("----Classification Matrix------")
    print (cm)

    print ("----Classification Report------")
    print (cr)
    return auc,ap,f1,acc,precision,recall

