
import torch
from torch.optim import Adam, lr_scheduler
import numpy as np
import math

def lin_log(x, threshold=1.0001):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255 TODO assumes 8 bit
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping

    Returns: the log value
    """
    # converting x into np.float64.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()
    rounding = 1e-8
    f = (1./threshold) * math.log(threshold)
    #y = torch.where(x <= threshold, x*f, torch.log(x))
    y = torch.where(x <= threshold, x * f, torch.log(x+rounding))
    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    # rounding = 1e8
    # y = torch.round(y*rounding)/rounding

    #return y.float()
    return y.float()

def  forward_grad(x):
    x_d = torch.roll(x, -1, dims=1) - x
    y_d = torch.roll(x, -1, dims=0) - x
    return y_d, x_d
def l2_loss(x):
    y = torch.mean(torch.square(x))
    return y

def tv_loss_weight(x, tv_order=1, tv_tau=1e-4, iso=True, w_xy =[1, 1]):
    '''The smaller tv_tau, the smoother the image.'''
    arr_size = 1
    for idx in range(len(x.shape)):
        arr_size = arr_size * x.shape[idx]

    if tv_order == 1:
        x_d = x - torch.roll(x, -1, dims=1)
        y_d = x - torch.roll(x, -1, dims=0)
    elif tv_order == 2:
        '''need check'''
        x_d = x - 2 * torch.roll(x, -1, dims=1) + torch.roll(x, 2, dims=1)
        y_d = x - 2 * torch.roll(x, -1, dims=0) + torch.roll(x, 2, dims=0)

    # x_d[:, -tv_order:-1] = torch.tensor(0.0, device=x.device) * w_xy[0]
    # y_d[:tv_order - 1, :] = torch.tensor(0.0, device=x.device) * w_xy[1]
    # x_d[:, :tv_order-1] = torch.zeros(1, device=x.device)
    # y_d[-tv_order:-1, :] = torch.zeros(1, device=x.device)

    if iso == True:
        TV_amp = torch.sqrt((x_d.abs()* w_xy[0]) ** 2 + (y_d.abs()* w_xy[1]) ** 2 + tv_tau).sum()
    elif iso == False:
        TV_amp = (x_d.abs()* w_xy[0] + y_d.abs()* w_xy[1]).sum()
    return TV_amp / arr_size
def forward_model(I,Dxy):
    L = lin_log(I * 255, 1.0001 * torch.tensor(1).to(device)) / np.log(255)
    return -forward_grad(L)[1] * Dxy[0] - forward_grad(L)[0] * Dxy[1]
def retrieve_intensity(iwe, flow_xy, lr =2e-1, iter  = 1000, kappa_tv = 1e-2):
    I_pred = torch.nn.Parameter(torch.ones_like(iwe).float() / 2, requires_grad=True)
    vars = [];vars += [{'params': I_pred, 'lr': lr}]; optimizer = Adam(vars);
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200 // 2, gamma=0.7, verbose=False);
    loss_hist =[]
    flow_np = flow_xy.detach().cpu().numpy()
    max_scl = np.max(np.abs(flow_np));
    w_y, w_x = np.abs(flow_np) / max_scl
    for i in range(iter):
        optimizer.zero_grad()
        iwe_pred =  forward_model(I_pred,flow_xy)
        L_tot = 0
        L_D = l2_loss(iwe_pred - iwe.detach()/ torch.norm(iwe.detach(), 2))
        L_tot += L_D
        L_tot  += tv_loss_weight(x=I_pred, tv_order=1, tv_tau=1e-3, iso=True, w_xy=[kappa_tv * w_x, kappa_tv * w_y])
        loss_hist.append(L_tot.cpu().detach())
        L_tot.backward(retain_graph=True)  # Calculate the derivatives
        optimizer.step()
        scheduler.step()
        if iter % 100 == 0:
            print("iter = {}: loss = {}, d_phi={}".format(iter, L_tot.data.cpu().numpy(),
                                                          I_pred.grad.mean().cpu().numpy()))
    return I_pred


if __name__=='__main__':
    ## parameter
    iwe = None      ## array
    flow_xy = None  ## [1,0]
    ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); eve_dtype ="aedat4"
    iwe =torch.from_numpy(iwe).to(device)
    flow_xy=torch.from_numpy(np.asarray([flow_xy[0],flow_xy[1]])).to(device)     ## dx dy  位移像素
    I_pred = retrieve_intensity(iwe, flow_xy, lr =2e-2, iter  = 1000, kappa_tv = 5e-1) ## retrieved intensity
