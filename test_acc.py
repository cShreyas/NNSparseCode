import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.datasets import mnist, fashion_mnist 
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import torchmetrics as tm 
import matplotlib.pyplot as plt 
from utils import *
from tqdm import tqdm

## Load model 
dataset = 'MNIST'
name = 'L4H64'

model = MLP.load_from_checkpoint(checkpoint_path="./tb_logs/" + dataset + '/' + name + "/version_0/checkpoints/epoch=99-step=46900.ckpt").cpu().eval()
model.requires_grad_(False)
x_train, y_train, x_test, y_test = load_data(mnist)

## Get intermediate features for each class
with torch.no_grad():
    train_feats, _ = model(x_train)

with torch.no_grad():
    test_feats, test_preds = model(x_test)

test_acc = torch.sum(test_preds.argmax(dim=-1) == y_test)/y_test.numel()
print('Test Accuracy: ', test_acc.item())


## Plot the test features at each layer
for i in range(len(test_feats)):
    fig1, axes1 = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))
    for j in range(len(axes1.flat)):
        class_feats = test_feats[i][y_test==j,:]
        im1 = axes1.flat[j].imshow(class_feats, cmap='viridis', interpolation='nearest', aspect='auto')
        axes1.flat[j].set_xlabel('feature index')
        axes1.flat[j].set_ylabel('sample index')
        axes1.flat[j].set_title('Class ' + str(j))
    fig1.colorbar(im1, ax=axes1.ravel().tolist())
    plt.savefig('./figures/test_featsL'+str(i)+'.png')
    plt.close()


# Compute accuracy using Taylor approximation
print('Computing Taylor Approximation Accuracy:')
modules = []
for l in range(len(model.linears) - 1):
    modules.append(model.linears[l])
    modules.append(nn.ReLU())
modules.append(model.linears[-1])
net = nn.Sequential(*modules)
taylor_acc = 0

for j in tqdm(range(x_test.shape[0])):

    s = x_test[j].view(1,-1)
    
    # find nearest sample in trianing dataset
    idx = torch.argmin(torch.linalg.vector_norm(x_train - s, dim=-1))
    z = x_train[idx].view(1,-1)

    # take Taylor approximation around z
    J = torch.autograd.functional.jacobian(net, z).squeeze()
    with torch.no_grad():
        out = net(z).view(-1,1) + J @ (z.T - s.T)
    taylor_acc += out.argmax() == y_test[j]

taylor_acc = taylor_acc / y_test.numel()
print('\nTaylor Approx Acc: ', taylor_acc)


# plot accuracy vs threshold 
print('\n\nPlotting Accuracy vs Threshold')
thresh_vals = torch.linspace(1e-4,6,50)
acc = torch.zeros(len(thresh_vals),10)
counts = torch.zeros(10)
for c in range(10):
    counts[c] = torch.sum(y_test == c)

for i in tqdm(range(len(thresh_vals))):
    thresh = thresh_vals[i]

    D_mats = {}
    for c in range(10):
        temp = {}
        for l in range(len(train_feats)):
            feats = train_feats[l][y_train==c,:]
            feats[feats < thresh] = 0
            feats[feats > 0] = 1
            temp[l] = torch.mean(feats,dim=0)

        D_mats[c] = temp

    for j in range(x_test.shape[0]):

        s = x_test[j].view(1,-1)
        c = y_test[j].item()

        with torch.no_grad():
            for k in range(len(model.linears)-1):
                s = model.linears[k](s)
                s = D_mats[c][k] * s
            s = model.linears[-1](s)
        acc[i,c] += (s.argmax() == c)


class_accs = acc / counts
total_accs = torch.sum(acc,dim=-1)/torch.sum(counts)
plt.figure()
plt.plot(thresh_vals, total_accs)
plt.axhline(taylor_acc, color='r',linestyle='--', label='Taylor Approx Accuracy')
plt.axhline(test_acc, color='k',linestyle='--', label='Test Accuracy')
plt.xlabel('Threshold Value')
plt.ylabel('Accuracy')
plt.savefig('./figures/acc_vs_thresh.png')


# plot accuracy vs topk
print('\n\nPlotting Accuracy vs TopK')
k_vals = torch.arange(1,32,2)
acc = torch.zeros(len(k_vals),10)
counts = torch.zeros(10)
for c in range(10):
    counts[c] = torch.sum(y_test == c)

for i in tqdm(range(len(k_vals))):
    k_val = k_vals[i]

    D_mats = {}
    for c in range(10):
        temp = {}
        for l in range(len(train_feats)):
            feats = train_feats[l][y_train==c,:]
            nonzero_counts = torch.count_nonzero(feats,dim=0)
            vals,idx = torch.topk(nonzero_counts, k_val)
            D = torch.zeros(train_feats[l].shape[-1]) 
            D[idx] = 1
            temp[l] = D

        D_mats[c] = temp

    for j in range(x_test.shape[0]):

        s = x_test[j].view(1,-1)
        c = y_test[j].item()

        with torch.no_grad():
            for k in range(len(model.linears)-1):
                s = model.linears[k](s)
                s = D_mats[c][k] * s
            s = model.linears[-1](s)
        acc[i,c] += (s.argmax() == c)

class_accs = acc / counts
total_accs = torch.sum(acc,dim=-1)/torch.sum(counts)
plt.figure()
plt.plot(k_vals, total_accs)
plt.axhline(taylor_acc, color='r',linestyle='--', label='Taylor Approx Acc')
plt.axhline(test_acc, color='k',linestyle='--', label='Test Acc')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.savefig('./figures/acc_vs_kval.png')


# plot accuracy vs topk (assume class is unknown)
# print('\n\nPlotting Accuracy vs TopK (class is unkown)')
# k_vals = torch.arange(1,32,2)
# acc = torch.zeros(len(k_vals),10)
# counts = torch.zeros(10)
# for c in range(10):
#     counts[c] = torch.sum(y_test == c)

# for i in tqdm(range(len(k_vals))):
#     k_val = k_vals[i]

#     D_mats = {}
#     for c in range(10):
#         temp = {}
#         for l in range(len(train_feats)):
#             feats = train_feats[l][y_train==c,:]
#             nonzero_counts = torch.count_nonzero(feats,dim=0)
#             _, idx = torch.topk(nonzero_counts, k_val)
#             D = torch.zeros(train_feats[l].shape[-1]) 
#             D[idx] = 1
#             temp[l] = D

#         D_mats[c] = temp

#     for j in range(x_test.shape[0]):

#         s_vals = []
#         with torch.no_grad():
#             for c in range(10):
#                 s = x_test[j].view(1,-1)
#                 for k in range(len(model.linears)-1):
#                     s = model.linears[k](s)
#                     s = D_mats[c][k] * s
#                 s = model.linears[-1](s).squeeze()
#                 s_vals.append(s)

#             s_vals = torch.stack(s_vals,dim=-1).softmax(dim=0).diag()
            
#         acc[i,y_test[j]] += (s_vals.argmax() == y_test[j])
        

# class_accs = acc / counts
# total_accs = torch.sum(acc,dim=-1)/torch.sum(counts)
# plt.figure()
# plt.plot(k_vals, total_accs)
# plt.axhline(taylor_acc, color='r',linestyle='--')
# plt.axhline(test_acc, color='k',linestyle='--')
# plt.xlabel('k')
# plt.ylabel('Accuracy')
# plt.savefig('./figures/acc_vs_kval_unkown.png')

# plot accuracy vs topk (assume class is unknown)
# print('\n\n Computing Accuracy by Hamming distance at each layer')
# acc = torch.zeros(len(train_feats))

# for i in range(len(train_feats)):

#     hamming_distances = torch.zeros(10, y_test.shape[0])
#     for c in range(10):

#         feats = train_feats[i][y_train==c,:]
#         feats[feats > 0] = 1
#         feats = feats.cuda()

#         samples = test_feats[i]
#         samples[samples > 0] = 1
#         samples=samples.cuda()

#         dist = torch.cdist(feats.unsqueeze(0), samples.unsqueeze(0), p=0).squeeze()
#         dist = dist.amin(dim=0)
#         hamming_distances[c] = dist

#     pred=hamming_distances.argmin(dim=0)
#     acc[i] = torch.sum(pred==y_test)/y_test.shape[0]

# print(acc)