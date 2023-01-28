#%%

import torch
import time
import numpy as np
from Utils_Lorenz96_clustered import *
from torch.utils.tensorboard import SummaryWriter
import scipy.io
import os

if torch.cuda.is_available():
    cuda_tag = "cuda:0"
    device = torch.device(cuda_tag)  
    print("Running on " + cuda_tag)
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
writer = SummaryWriter("DNN-SNN")

start_time = time.time()
# =============================================================================
# fix random seed
# =============================================================================
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# Prepare data
# =============================================================================
data = scipy.io.loadmat(os.path.join(os.path.dirname(os.getcwd()),'Lorenz96.mat'))

X = data['X'].astype(np.float32) 
t = data['t'].astype(np.float32) 

# add noise based on ref data
noise_lvl = 0.1
X = X + noise_lvl*np.std(X, axis = 0, keepdims=True)*np.random.normal(size=X.shape).astype(np.float32) 

# downsample
downsample_step = 10
duration = 1001 # 10 sec
X = X[:duration:downsample_step,:]
t = t[:duration:downsample_step,:]

# =============================================================================
# Set up fourier NN
# =============================================================================
H, D_out = 100, X.shape[1]
    
t_fourier_std = [1, 10, 100]
N_hidd_layers = 3
FNN = Fourier_MLP(H, N_hidd_layers, D_out, t_fourier_std, device)

# =============================================================================
#     Set up symbolic NN
# =============================================================================
D_P1, D_out_r = X.shape[1], 1
SNNs_list = [SymbolicNet_independent(D_P1, D_out_r).to(device) for _ in range(X.shape[1])]

# =============================================================================
# Pretrain
# =============================================================================
Train
X_tr_pred_norm, X_tr_norm, \
            dX_tr_pred_norm, dX_tr_norm,\
                valid_ind, \
                    X_collo_pred_norm, \
                        X_norm_coeff, dX_norm_coeff = Train(60000, FNN, SNNs_list, X, t, writer, device, downsample_step, SNN_coeff = 0, loss_eq_coeff = 1e2, 
                        lr_decay = False) # 60000

# Train
X_tr_pred_norm, X_tr_norm, \
            dX_tr_pred_norm, dX_tr_norm,\
                valid_ind, \
                    X_collo_pred_norm, \
                        X_norm_coeff, dX_norm_coeff = Train(20000, FNN, SNNs_list, X, t, writer, device, downsample_step, SNN_coeff = 1e-3, loss_eq_coeff = 1e2, 
                        lr_decay = False) # 10000

save_dict = {'FNN_state_dict': FNN.state_dict()}
for counter, SNN in enumerate(SNNs_list):
    save_dict['SNN'+str(counter)+'_state_dict'] = SNN.state_dict()
torch.save(save_dict, 'model_Pre.tar')

scipy.io.savemat('pred.mat',{'X_tr_pred_norm':X_tr_pred_norm, 'X_tr_norm':X_tr_norm[valid_ind,:],
'dX_tr_pred_norm':dX_tr_pred_norm, 'dX_tr_norm':dX_tr_norm[valid_ind,:],
'X_collo_pred_norm':X_collo_pred_norm,
'X_norm_coeff':X_norm_coeff, 'dX_norm_coeff':dX_norm_coeff
})

#  accuracy
error_X_tr = np.linalg.norm(X_tr_pred_norm - X_tr_norm[valid_ind,:])/np.linalg.norm(X_tr_norm[valid_ind,:])*100

writer.add_text('error_X_tr', 'Train Error(%):' + str(error_X_tr))

#%%
# =============================================================================
# Adaptive network pruning
# just prune. no training.
# =============================================================================
# load model
checkpoint = torch.load('model_Pre.tar', map_location=device)
FNN.load_state_dict(checkpoint['FNN_state_dict'])
for counter, SNN in enumerate(SNNs_list):
    SNN.load_state_dict(checkpoint['SNN'+str(counter)+'_state_dict'])

t, X, dX, t_min, t_max, ones_norm, X_norm_coeff, dX_norm_coeff, t_collo = preprocessing(X, t, device, downsample_step)

for counter, SNN in enumerate(SNNs_list):
        
    # a networkwise grid search from 0 to 100 percentile
    threshold_list = []
    weights_all = np.concatenate([para.detach().to('cpu').numpy().flatten() for para in SNN.linear_weights_all], axis=0)
    threshold_list = [np.percentile(np.abs(weights_all), My_Percentile) for My_Percentile in range(0,101)]

    # hard-threshold SNN
    SNN_err_list = []
    No_Nonzero_list = []
    i_list = [] 
    for i, threshold in enumerate(threshold_list):
        SNN = HardThreshold_networkwise(SNN, threshold, GradMask = False)
        SNN_err = ComputeSNNError(FNN, SNN, counter, t_collo, t_min, t_max, device, X_norm_coeff, dX_norm_coeff)
        No_Nonzero = CountNonZeros(SNN)

        SNN_err_list.append(SNN_err)
        No_Nonzero_list.append(No_Nonzero)
        i_list.append(i)


    scipy.io.savemat('SNN_'+str(counter)+'_prune.mat', {'SNN_err':np.stack(SNN_err_list), 'No_Nonzero_all':np.stack(No_Nonzero_list), 'i_all':np.stack(i_list)})

# =============================================================================
# Print pruned equation
# =============================================================================
best_i_all = [91, 88, 75, 90, 94, 94, 94, 94]

# load model
checkpoint = torch.load('model_Pre2.tar', map_location=device)
FNN.load_state_dict(checkpoint['FNN_state_dict'])

save_dict = {'FNN_state_dict': FNN.state_dict()}

t, X, dX, t_min, t_max, ones_norm, X_norm_coeff, dX_norm_coeff, t_collo = preprocessing(X, t, device, downsample_step)

ref_eq_list = build_ref_eqs() 

true_pos_all = 0
false_pos_all = 0
false_neg_all = 0
error_list_all = []

for counter, SNN in enumerate(SNNs_list):
    SNN.load_state_dict(checkpoint['SNN'+str(counter)+'_state_dict'])

    best_i = best_i_all[counter]
    print('the best percentile for SNN', str(counter), 'is',str(best_i))
    true_pos, false_pos, false_neg, error_list = print_eq_network_prune(SNN, counter, best_i, X_norm_coeff, dX_norm_coeff, ref_eq_list[counter])  

    true_pos_all += true_pos
    false_pos_all += false_pos
    false_neg_all += false_neg
    error_list_all += error_list

recall =  true_pos_all/(true_pos_all+false_neg_all)*100
precision = true_pos_all/(true_pos_all + false_pos_all)*100
coeff_error = np.mean(np.stack(error_list_all))

print('recall:'+str(recall))
print('precision:'+str(precision))
print('coeff_error:'+str(coeff_error))

# =============================================================================
# Record results
# =============================================================================
elapsed = time.time() - start_time  
writer.add_text('Time', 'Training time:' + str(elapsed))
 
writer.flush()
writer.close()


# %%
