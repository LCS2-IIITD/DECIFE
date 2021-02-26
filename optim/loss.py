import torch    
import numpy as np
import torch.nn.functional as F
    
def loss_function(nu,data_center,outputs,radius=0,mask=None):
    dist,scores=anomaly_score(data_center,outputs,radius,mask)
    loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss,dist,scores

def anomaly_score(data_center,outputs,radius=0,mask= None):
    if mask == None:
        dist = torch.sum((outputs - data_center) ** 2, dim=1)
    else:
        dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)
    scores = dist - radius ** 2
    return dist,scores

def init_center(args,input_g,input_feat, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    model.eval()
    with torch.no_grad():

        outputs= model(input_g,input_feat)

        # get the inputs of the batch
        n_samples = outputs.shape[0]
        c =torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    radius=np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    # if radius<0.1:
    #     radius=0.1
    return radius
