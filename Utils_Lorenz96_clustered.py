import numpy as np
import torch
import sympy as sym

# =============================================================================
# Functions
# =============================================================================
def build_ref_eqs():
    x0, x1, x2, x3, x4, x5, x6, x7 = sym.symbols('x0, x1, x2, x3, x4, x5, x6, x7')

    ref_eq_list = [sym.Matrix([[-1.*x0 + 1.*x1*x7 - 1.*x6*x7 + 8.]])]
    ref_eq_list.append(sym.Matrix([[-1.*x1 + 1.*x0*x2 - 1.*x0*x7 + 8.]]))
    ref_eq_list.append(sym.Matrix([[-1.*x2 + 1.*x1*x3 - 1.*x0*x1 + 8.]]))
    ref_eq_list.append(sym.Matrix([[-1.*x3 + 1.*x2*x4 - 1.*x1*x2 + 8.]]))
    ref_eq_list.append(sym.Matrix([[-1.*x4 + 1.*x3*x5 - 1.*x2*x3 + 8.]]))
    ref_eq_list.append(sym.Matrix([[-1.*x5 + 1.*x4*x6 - 1.*x3*x4 + 8.]]))
    ref_eq_list.append(sym.Matrix([[-1.*x6 + 1.*x5*x7 - 1.*x4*x5 + 8.]]))
    ref_eq_list.append(sym.Matrix([[-1.*x7 + 1.*x0*x6 - 1.*x5*x6 + 8.]]))
    return ref_eq_list

def Train(epochs, FNN, SNNs_list, X, t, writer, device, downsample_step, SNN_coeff = 0, loss_eq_coeff = 1, lr_decay = False, fix_DNN = False):
    SNNs_weights_list = []
    for SNN in SNNs_list:
        SNNs_weights_list += SNN.linear_weights_all
    if fix_DNN:
        optimizer = torch.optim.Adam([
                    {'params': SNNs_weights_list},
                ], lr=1e-3)
    else:
        optimizer = torch.optim.Adam([
                    {'params': FNN.parameters()},
                    {'params': SNNs_weights_list},
                ], lr=1e-3)
    if lr_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs/8), gamma = 0.6)
    loss_fn = torch.nn.MSELoss()

    t, X, dX, t_min, t_max, _, X_norm_coeff, dX_norm_coeff, t_collo = preprocessing(X, t, device, downsample_step)

    # split data
    ind_tr = np.arange(int(t.shape[0]))
    t_tr = torch.from_numpy(t[ind_tr]).to(device)
    X_tr_norm = (torch.from_numpy(X[ind_tr]).to(device))*X_norm_coeff
    dX_tr_norm = (torch.from_numpy(dX[ind_tr]).to(device))*dX_norm_coeff

    for iter in range(epochs):
        # FNN forward pass
        X_tr_pred_norm, _, valid_ind = X_dX_pred(FNN, t_tr, t_min, t_max, X_norm_coeff, dX_norm_coeff)
        loss_X_tr = loss_fn(X_tr_pred_norm, X_tr_norm[valid_ind,:])

        # SNN forward pass
        X_collo_pred_norm, dX_collo_pred_norm,_ = X_dX_pred(FNN, t_collo, t_min, t_max, X_norm_coeff, dX_norm_coeff)
        Phi_norm = torch.cat((torch.ones_like(X_collo_pred_norm[:,[1]]), X_collo_pred_norm), 1)
        f_pred_norm_list = []
        for SNN in SNNs_list:
            f_pred_norm_list.append(SNN(Phi_norm))
        f_pred_norm = torch.cat(f_pred_norm_list, dim = 1)
        loss_eq = loss_fn(f_pred_norm - dX_collo_pred_norm, torch.zeros_like(dX_collo_pred_norm))

        # SNN penalty.
        SNN_regu = 0 # need to update loss_NNparas in every train loop
        for weight in SNNs_weights_list:
            # smooth L0.5. 
            SNN_regu += ell_half_norm(weight) 
        SNN_regu *= SNN_coeff

        loss_tr = loss_X_tr + loss_eq_coeff*loss_eq + SNN_regu # + loss_dX_tr
        
        if iter % 10 == 0:
            writer.add_scalar('loss_X_tr', loss_X_tr.item(), iter)
            writer.add_scalar('loss_eq', loss_eq.item(), iter)
            writer.add_scalar('SNN_regu', SNN_regu.item(), iter)
               
        optimizer.zero_grad()
    
        loss_tr.backward()
    
        optimizer.step()
        
        if lr_decay:
            scheduler.step()

    # evalute results
    X_tr_pred_norm, dX_tr_pred_norm, valid_ind = X_dX_pred(FNN, t_tr, t_min, t_max, X_norm_coeff, dX_norm_coeff)

    return X_tr_pred_norm.detach().cpu().numpy(), X_tr_norm.cpu().numpy(), \
            dX_tr_pred_norm.detach().cpu().numpy(), dX_tr_norm.cpu().numpy(), \
                valid_ind, \
                    X_collo_pred_norm.detach().cpu().numpy(), \
                        X_norm_coeff.cpu().numpy(), dX_norm_coeff.cpu().numpy()

def preprocessing(X, t, device, downsample_step, smooth_flag = False):
    # smooth X
    if smooth_flag:
        X, t = move_average(X, t)

    # finite difference
    t, X, dX, _ = FiniteDiff_Truncate(X, t) # np arrays

    # normalization coeff
    t_min = np.amin(t)
    t_max = np.amax(t)
    ones_norm = np.linalg.norm(np.ones_like(X[:,0:1]))
    X_norm_coeff = torch.from_numpy(ones_norm/np.linalg.norm(X, axis = 0, keepdims = True)).to(device)
    dX_norm_coeff = torch.from_numpy(ones_norm/np.linalg.norm(dX, axis = 0, keepdims = True)).to(device)

    # generate collo pts
    t_collo = torch.linspace(t_min, t_max, downsample_step*t.shape[0], device = device).reshape((-1,1))
    return t, X, dX, t_min, t_max, ones_norm, X_norm_coeff, dX_norm_coeff, t_collo

def move_average(X, t, w=9):
    # w should be an odd number
    X_new_list = []
    for i_state in range(X.shape[1]):
        X_new_temp = np.convolve(X[:, i_state].flatten(), np.ones(w,dtype=np.float32), 'valid') / w
        X_new_list.append(X_new_temp.reshape((-1,1)))
    
    X_new = np.concatenate(X_new_list, axis = 1)
    t_new = t[int((w-1)/2):-int((w-1)/2), :]
    return X_new, t_new

def ell_half_norm(w):
    a = 0.01
    
    big_ind = torch.nonzero(torch.abs(w) >= a, as_tuple=True)
    regu_weight_big = torch.sum(torch.sqrt(torch.abs(w[big_ind])))

    small_ind = torch.nonzero(torch.abs(w) < a, as_tuple=True)
    regu_weight_small = torch.sum(torch.sqrt(-w[small_ind]**4/8/(a**3) + 3*w[small_ind]**2/4/a + 3*a/8))
   
    return regu_weight_big+regu_weight_small

def X_dX_pred(FNN, t_tr, t_min, t_max, X_norm_coeff, dX_norm_coeff):
    # finite diff. 4TH ORDER CENTRAL DIFFERENCE.
    # https://www.math.uakron.edu/~kreider/num2/CD4.pdf
    X_tr_pred_norm = DNN_w_InputNormalization(FNN, t_min, t_max, t_tr)
    X_tr_pred = X_tr_pred_norm/X_norm_coeff
    dt = t_tr[1,0]-t_tr[0,0]
    dX_tr_pred = (-X_tr_pred[4:,:]+8*X_tr_pred[3:-1,:]-8*X_tr_pred[1:-3,:]+X_tr_pred[:-4,:])/12/dt
    dX_tr_pred_norm = dX_tr_pred*dX_norm_coeff
    valid_ind = np.arange(2,X_tr_pred.shape[0]-2)
    X_tr_pred_norm = X_tr_pred_norm[valid_ind, :]
    return X_tr_pred_norm, dX_tr_pred_norm, valid_ind

def FiniteDiff_Truncate(X, t):
    dt = t[1,0] - t[0,0]
    # first-order backward difference. low-order diff to lessens noise effect
    dX = (X[1:, :] - X[:-1, :])/dt # valid range: (start+1:end)

    # second-order central
    ddX = (X[2:,:]-2*X[1:-1,:]+X[:-2,:])/dt**2 # valid range: (start+1:end-1)
    return t[1:-1,:], X[1:-1,:], dX[:-1,:], ddX

def DNN_w_InputNormalization(model, lb, ub, X):
    H = 2*(X - lb)/(ub - lb) - 1
    return model(H)

def CountNonZeros(SNN):
    weights_SNN_list = [para.detach().to('cpu').numpy().flatten() for para in SNN.linear_weights_all]
    weights_SNN = np.concatenate(weights_SNN_list)
    No_Nonzero = np.count_nonzero(weights_SNN)
    return No_Nonzero

def HardThreshold_networkwise(SNN, threshold, GradMask = True):   
    with torch.no_grad():
        # apply hard threshold to SNN weights            
        ind1 = torch.abs(SNN.linearP1_2.weight) < threshold
        SNN.linearP1_2.weight[ind1] = 0
        SNN.linearP1_2.weight.requires_grad = True
        # fix zero weights as zeros: Create Gradient mask
        if GradMask == True:
            gradient_mask1 = torch.ones_like(SNN.linearP1_2.weight)
            gradient_mask1[ind1] = 0
            SNN.linearP1_2.weight.register_hook(lambda grad: grad.mul_(gradient_mask1))
    
        # apply hard threshold to SNN weights          
        ind2 = torch.abs(SNN.linearP1_2_3.weight) < threshold
        SNN.linearP1_2_3.weight[ind2] = 0
        SNN.linearP1_2_3.weight.requires_grad = True
        # fix zero weights as zeros: Create Gradient mask
        if GradMask == True:
            gradient_mask2 = torch.ones_like(SNN.linearP1_2_3.weight)
            gradient_mask2[ind2] = 0
            SNN.linearP1_2_3.weight.register_hook(lambda grad: grad.mul_(gradient_mask2))

        # apply hard threshold to SNN weights          
        ind3 = torch.abs(SNN.linearP2_3.weight) < threshold
        SNN.linearP2_3.weight[ind3] = 0
        SNN.linearP2_3.weight.requires_grad = True
        # fix zero weights as zeros: Create Gradient mask
        if GradMask == True:
            gradient_mask3 = torch.ones_like(SNN.linearP2_3.weight)
            gradient_mask3[ind3] = 0
            SNN.linearP2_3.weight.register_hook(lambda grad: grad.mul_(gradient_mask3))

        # apply hard threshold to SNN weights          
        ind4 = torch.abs(SNN.linear1_2_4.weight) < threshold
        SNN.linear1_2_4.weight[ind4] = 0
        SNN.linear1_2_4.weight.requires_grad = True
        # fix zero weights as zeros: Create Gradient mask
        if GradMask == True:
            gradient_mask4 = torch.ones_like(SNN.linear1_2_4.weight)
            gradient_mask4[ind4] = 0
            SNN.linear1_2_4.weight.register_hook(lambda grad: grad.mul_(gradient_mask4))

        # apply hard threshold to SNN weights          
        ind5 = torch.abs(SNN.linear2_4.weight) < threshold
        SNN.linear2_4.weight[ind5] = 0
        SNN.linear2_4.weight.requires_grad = True
        # fix zero weights as zeros: Create Gradient mask
        if GradMask == True:
            gradient_mask5 = torch.ones_like(SNN.linear2_4.weight)
            gradient_mask5[ind5] = 0
            SNN.linear2_4.weight.register_hook(lambda grad: grad.mul_(gradient_mask5))

        # apply hard threshold to SNN weights
        ind6 = torch.abs(SNN.linear_out.weight) < threshold
        SNN.linear_out.weight[ind6] = 0
        SNN.linear_out.weight.requires_grad = True
        # fix zero weights as zeros: Create Gradient mask
        if GradMask == True:
            gradient_mask6 = torch.ones_like(SNN.linear_out.weight)
            gradient_mask6[ind6] = 0
            SNN.linear_out.weight.register_hook(lambda grad: grad.mul_(gradient_mask6))
    
        # relink linear_weights_all
        SNN.linear_weights_all = [SNN.linearP1_2.weight,
                SNN.linearP1_2_3.weight, SNN.linearP2_3.weight,
                SNN.linear1_2_4.weight, SNN.linear2_4.weight,
                SNN.linear_out.weight]
    return SNN
    
def ComputeSNNError(FNN, SNN, counter, t_collo, t_min, t_max, device, X_norm_coeff, dX_norm_coeff):
    X_collo_pred_norm, dX_collo_pred_norm, _ = X_dX_pred(FNN, t_collo, t_min, t_max, X_norm_coeff, dX_norm_coeff)
    Phi_norm = torch.cat((torch.ones_like(X_collo_pred_norm[:,[1]]), X_collo_pred_norm), 1)
    f_counter_pred_norm = SNN(Phi_norm)
    dx_collo_pred_norm = dX_collo_pred_norm[:, [counter]]
    SNN_err = torch.norm(f_counter_pred_norm - dx_collo_pred_norm)/torch.norm(dx_collo_pred_norm)*100

    return SNN_err.item()

def print_eq_network_prune(SNN, counter, best_i, X_norm_coeff, dX_norm_coeff, ref_eq):
    # a networkwise grid search from 0 to 100 percentile
    threshold_list = []
    weights_all = np.concatenate([para.detach().to('cpu').numpy().flatten() for para in SNN.linear_weights_all], axis=0)
    threshold_list = [np.percentile(np.abs(weights_all), My_Percentile) for My_Percentile in range(0,101)]

    # best model
    SNN = HardThreshold_networkwise(SNN, threshold_list[best_i])

    ## print equation
    x0, x1, x2, x3, x4, x5, x6, x7 = sym.symbols('x0, x1, x2, x3, x4, x5, x6, x7')

    inp = sym.Matrix([[1, x0, x1, x2, x3, x4, x5, x6, x7]])

    # normalize inputs
    inp[:,1:] = sym.matrix_multiply_elementwise(inp[:,1:], sym.Matrix(X_norm_coeff.cpu().numpy()))
    
    P0 = inp[:,0:1]
    P1 = inp[:,1:]
    weight_linearP1_2 = SNN.linear_weights_all[0].detach().to('cpu').numpy()
    P2_subs = P1@(weight_linearP1_2.T)
    P2 = P2_subs[:,0:1]*P2_subs[:,1:2]

    weight_linearP1_2_3 = SNN.linear_weights_all[1].detach().to('cpu').numpy()
    P2_3_subs = P1@(weight_linearP1_2_3.T)
    P2_3_temp = P2_3_subs[:,0:1]*P2_3_subs[:,1:2]
    P2_3 = P1.row_join(P2_3_temp)

    weight_linearP2_3 = SNN.linear_weights_all[2].detach().to('cpu').numpy()
    P3_subs = P2_3@(weight_linearP2_3.T)
    P3 = P3_subs[:,0:1]*P3_subs[:,1:2]

    weight_linear1_2_4 = SNN.linear_weights_all[3].detach().to('cpu').numpy()
    P1_2_4_subs = P1@(weight_linear1_2_4.T)
    P1_2_4_temp1 = P1_2_4_subs[:,0:1]*P1_2_4_subs[:,1:2]
    P1_2_4_temp2 = P1_2_4_subs[:,2:3]*P1_2_4_subs[:,3:4]
    P1_2_4 = P1_2_4_temp1.row_join(P1_2_4_temp2)

    weight_linear2_4 = SNN.linear_weights_all[4].detach().to('cpu').numpy()
    P4_subs = P1_2_4@(weight_linear2_4.T)
    P4 =  P4_subs[:,:1]*P4_subs[:,1:2]

    P_all = P0.row_join(P1.row_join(P2.row_join(P3.row_join(P4))))
    weight_linear_out = SNN.linear_weights_all[-1].detach().to('cpu').numpy()
    out = P_all@(weight_linear_out.T)

    # denormalize output
    out = out/(dX_norm_coeff[0,counter].item())

    eq = sym.expand(out)
    print('The discovered Equation is dx = ' + str(eq[0, 0]))

    ## compare the discovered eq with the reference. 
    # return true positives, false positives, false negatives, relative error for each correct equation coeff
    true_pos = 0
    false_pos = 0
    false_neg = 0
    error_list = []
    for term in eq[0,0]._sorted_args:
        pos = False # this term is false positive by default
        for term_ref in ref_eq[0,0]._sorted_args:
            if term.args[1:] == term_ref.args[1:]:
                true_pos += 1
                pos = True 

                if term.args[1:] == ():
                    term_coeff = np.array(term).astype(np.float32)
                    term_ref_coeff = np.array(term_ref).astype(np.float32)
                    error_list.append(np.abs(term_coeff-term_ref_coeff)/np.abs(term_ref_coeff)*100)
                else:
                    error_list.append(np.abs(term.args[0]-term_ref.args[0])/np.abs(term_ref.args[0])*100)
        if pos == False:
            false_pos += 1
    
    for term_ref in ref_eq[0,0]._sorted_args:
        neg = False # by default this term has not been discovered
        for term in eq[0,0]._sorted_args:
            if term.args[1:] == term_ref.args[1:]:
                neg = True 
        if neg == False:
            false_neg += 1

    return true_pos, false_pos, false_neg, error_list

# =============================================================================
# Classes
# =============================================================================
class SymbolicNet_independent(torch.nn.Module):
    def __init__(self, D_P1, D_out):
        super(SymbolicNet_independent, self).__init__()
        self.D_P0 = 1
        self.D_P1 = D_P1
        self.D_P2 = 1
        self.D_P3 = 1
        self.D_P4 = 1
        self.D_out = D_out

        self.linearP1_2 = torch.nn.Linear(self.D_P1, 2, bias = False)

        self.linearP1_2_3 = torch.nn.Linear(self.D_P1, 2, bias = False) 
        self.linearP2_3 = torch.nn.Linear(self.D_P1+self.D_P2, 2, bias = False) 

        self.linear1_2_4 = torch.nn.Linear(self.D_P1, 4, bias = False) 
        self.linear2_4 = torch.nn.Linear(2, 2, bias = False)

        self.linear_out = torch.nn.Linear(self.D_P0+self.D_P1+self.D_P2+self.D_P3+self.D_P4, self.D_out, bias = False)

        self.linear_weights_all = [self.linearP1_2.weight,
         self.linearP1_2_3.weight, self.linearP2_3.weight,
          self.linear1_2_4.weight, self.linear2_4.weight,
          self.linear_out.weight]

    def forward(self, x):
        P0 = x[:,[0]]
        P1 = x[:,1:]
        P2 = self.multi(self.linearP1_2(P1))

        P3 = self.multi(self.linearP2_3(torch.cat([P1, self.multi(self.linearP1_2_3(P1))], dim=-1)))

        P4 = self.multi(self.linear2_4(torch.cat([self.multi(self.linear1_2_4(P1)[:,:2]), self.multi(self.linear1_2_4(P1)[:,2:])], dim = -1)))

        y = self.linear_out(torch.cat([P0, P1,P2,P3,P4],dim=-1))
        return y

    def multi(self,x):
        return x[:,[0]]*x[:,[1]]

class Fourier_MLP(torch.nn.Module):
    def __init__(self, H, N_hidd_layers, D_out, t_fourier_std, device):
        super(Fourier_MLP, self).__init__()
        # the same set of NN parameters learn both spatial and temporal embedding
        # no linear input layer. input transformation is done by fourier embedding
        self.linears = torch.nn.ModuleList([torch.nn.Linear(H, H).to(device) for _ in range(N_hidd_layers - 1)])
        # output layer will digest N*H contents which are concatenated from temporal NNs
        self.linears.append(torch.nn.Linear(len(t_fourier_std)*H, D_out).to(device))

        # sample fourier std's 
        D_in_t = 1
        torch.manual_seed(0) # need to fix random seeds for reusing a trained model, because next line randomly samples t_embed_freq
        self.t_embed_freq = [torch.normal(mean=0,std=std,size=(D_in_t, H//2)).to(device) for std in t_fourier_std] # divided by 2 because we need sin and cos

        self.swish = torch.nn.SiLU()
    def forward(self, t):
        # fourier embedding transformation
        t_embed_inputs = [torch.cat([torch.sin(t@freq), torch.cos(t@freq)], dim=1) for freq in self.t_embed_freq]

        # hidden layers passing
        t_outputs = [self.hidd_pass(input) for input in t_embed_inputs]
        
        # output layer
        y = self.linears[-1](torch.cat(t_outputs, dim=1))
        return y

    def hidd_pass(self, x):
        for i in range(len(self.linears)-1):
            x = self.swish(self.linears[i](x))
        return x
