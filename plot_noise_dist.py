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
from sklearn.metrics import roc_curve, auc

## Load model 
dataset = 'MNIST'
name = 'L4_H512_C10'
model = MLP.load_from_checkpoint(checkpoint_path="./logs/" + dataset + '/' + name + "/version_0/checkpoints/epoch=86-step=40803.ckpt").cpu().eval()
model.requires_grad_(False)
n_classes = model.linears[-1].weight.shape[0]


x_train, y_train, x_test, y_test = load_data(mnist, n_classes=n_classes)

## Get intermediate features for each class
with torch.no_grad():
    train_feats, _ = model(x_train)

with torch.no_grad():
    test_feats, test_preds = model(x_test)

test_acc = torch.sum(test_preds.argmax(dim=-1) == y_test)/y_test.numel()
print('Test Accuracy: ', test_acc.item())

## Plot the test features at each layer
for i in range(len(test_feats)):
    fig1, axes1 = plt.subplots(nrows=2, ncols=n_classes//2, figsize=(18, 8))
    for j in range(len(axes1.flat)):
        class_feats = test_feats[i][y_test==j,:]
        im1 = axes1.flat[j].imshow(class_feats, cmap='viridis', interpolation='nearest', aspect='auto')
        axes1.flat[j].set_xlabel('feature index')
        axes1.flat[j].set_ylabel('sample index')
        axes1.flat[j].set_title('Class ' + str(j))
    fig1.colorbar(im1, ax=axes1.ravel().tolist())
    plt.savefig('./figures/test_featsL'+str(i)+'.png')
    plt.close()


# Get D matrices for each class:
D_mats = {}
for i in range(n_classes):
    D_mats[i] = get_nonzero_feat_idx(train_feats, y_train, class_idx=i)


z = [x_test.clone() for i in range(n_classes)]
for l in range(model.num_layers-1):
    for c in range(n_classes):
        z[c] = model.linears[l](z[c])
        z[c] = z[c] * D_mats[c][l].unsqueeze(0)
        
for c in range(n_classes):
    z[c] = model.linears[-1](z[c])

preds = torch.stack(z, dim=0).permute(1,0,2)
preds = preds / torch.linalg.vector_norm(preds, dim=-1, keepdim=True)
preds = torch.diagonal(preds, dim1=1, dim2=2)
import pdb 
pdb.set_trace()
preds = torch.softmax(preds, dim=-1)
preds = torch.argmin(preds,dim=-1)
import pdb 
pdb.set_trace()



# Get Jaccard similarity index
J = []
for l in range(model.num_layers):
    mat = torch.zeros(n_classes, n_classes)
    for c1 in range(n_classes):
        d = D_mats[c1][l]
        for c2 in range(n_classes):
            mat[c1, c2] = jaccard_similarity(d, D_mats[c2][l])

    J.append(mat)
    save_tensor_as_csv(mat, './jaccard_L'+str(l+1)+'.csv')

print('HERE')
input()

# Get embeddings for each class by linear model



## Get the D matrices for each class
D0 = get_nonzero_feat_idx(train_feats, y_train, class_idx=0)
D1 = get_nonzero_feat_idx(train_feats, y_train, class_idx=1)

# Get embeddings for each class by linear model
z0 = x_test[y_test == 0]
z1 = x_test[y_test == 1]

for i in range(model.num_layers-1):
    z0, z1 = model.linears[i](z0), model.linears[i](z1)
    
    mask_z0 = torch.zeros_like(z0)
    mask_z0[:, D0[i]] = 1

    mask_z1 = torch.zeros_like(z1)
    mask_z1[:, D1[i]] = 1

    
    z0 = z0 * mask_z0
    z1 = z1 * mask_z1

z0, z1 = model.linears[-1](z0), model.linears[-1](z1)

# Get accuracy of linearized model
acc = (torch.sum(z0.argmax(dim=-1) == 0) + torch.sum(z1.argmax(dim=-1) == 1)) / y_test.numel()
print('Linearization Accuracy:', acc)


## Plot the test outputs
num_samples = 1000

z0 = torch.softmax(z0, dim=-1)
z1 = torch.softmax(z1, dim=-1)

test_preds = torch.softmax(test_preds, dim=-1)
# x0 = test_preds[y_test == 0]
# x1 = test_preds[y_test == 1]


# Plot ROC curve
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, test_preds[:,-1])
roc_auc_nn = auc(fpr_nn, tpr_nn)

labels = torch.cat((torch.zeros(z0.shape[0],1), torch.ones(z1.shape[0],1)))
fpr_lin, tpr_lin, thresholds_lin = roc_curve(labels, torch.cat((z0, z1), dim=0)[:,1])
roc_auc_lin = auc(fpr_lin, tpr_lin)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nn, tpr_nn, color='darkorange', lw=2, label=f'NN ROC curve (AUC = {roc_auc_nn:.2f})')
plt.plot(fpr_lin, tpr_lin, color='green', lw=2, label=f'LIN ROC curve (AUC = {roc_auc_lin:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('./figures/test_roc.png')

# plt.figure()
# plt.scatter(x1[0:num_samples,0], x1[0:num_samples,1], marker='o', c='blue', s=20)
# plt.scatter(z1[0:num_samples,0], z1[0:num_samples,1], marker='x', c='red', s=20)

# print(x0[0:10])
# print(z0[0:10])

# plt.savefig('./figures/test_outputs.png')
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.close()
