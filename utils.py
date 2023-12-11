import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics as tm 
import numpy as np
import csv 

class CNN(pl.LightningModule):
   
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()

        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.maxpool2 = nn.MaxPool2d(2)

        self.lin1 = nn.Linear(1024, 256)
        self.lin2 = nn.Linear(256, 10)

                     
    def forward(self, x):
        
        z = []

        x = self.conv1(x)
        x = torch.relu(x)
        z.append(x.detach().flatten(start_dim=1))

        x = self.conv2(x)
        x = torch.relu(x)
        z.append(x.detach().flatten(start_dim=1))

        x = self.conv3(x)
        x = torch.relu(x)
        z.append(x.detach().flatten(start_dim=1))

        x = self.maxpool1(x)

        x = self.conv4(x)
        x = torch.relu(x)
        z.append(x.detach().flatten(start_dim=1))

        x = self.conv5(x)
        x = torch.relu(x)
        z.append(x.detach().flatten(start_dim=1))

        x = self.conv6(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)

        x = self.lin1(x)
        x = torch.relu(x)
        z.append(x.detach())

        x = self.lin2(x)

        return z, x

    def training_step(self, batch, batch_idx):
        x, y = batch
        sparse_codes, pred = self.forward(x)
        loss = F.cross_entropy(pred, y)
        train_acc = tm.functional.accuracy(pred, y)

        self.log('train_loss', loss)
        self.log('train_acc', train_acc, on_epoch=True)

        # log sparsity
        for i in range(len(sparse_codes)):
            sparse_level = torch.mean(torch.count_nonzero(sparse_codes[i], dim=-1)/sparse_codes[i].shape[-1])
            self.log('train_sparsity_layer'+str(i+1), sparse_level)
             
        return loss
 
    def validation_step(self, batch, batch_idx):       
        x, y = batch
        
        sparse_codes, pred = self.forward(x)
        loss = F.cross_entropy(pred, y)
        val_acc = tm.functional.accuracy(pred, y)

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', val_acc, on_epoch=True)

        # log sparsity
        for i in range(len(sparse_codes)):
            sparse_level = torch.mean(torch.count_nonzero(sparse_codes[i], dim=-1)/sparse_codes[i].shape[-1])
            self.log('val_sparsity_layer'+str(i+1), sparse_level)

        return {'sparse_code': sparse_codes, 'label': y} 
    

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-2)
        # sch = torch.optim.lr_scheduler.CyclicLR(opt, 1e-7, 1e-3, step_size_up=10, cycle_momentum=False)
        # opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
        return {'optimizer':opt, 'lr_scheduler':scheduler, 'monitor':'val_loss'}
        # return {'optimizer': opt, 'lr_scheduler': sch} 
        

class MLP(pl.LightningModule):
   
    def __init__(self, input_size, num_layers, layers_size, output_size, loss_fn=F.cross_entropy, dropout=0):
        super().__init__()
        self.save_hyperparameters()

        self.num_layers = num_layers
        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size, bias=False)])
        self.linears.extend([nn.Linear(layers_size, layers_size, bias=False) for i in range(num_layers-1)])
        self.linears.append(nn.Linear(layers_size, output_size, bias=False))
        self.loss_fn = loss_fn
        self.accuracy = tm.Accuracy(task="multiclass", num_classes=output_size)
        self.dropout = nn.Dropout1d(p=dropout)

                     
    def forward(self, x):
        
        z = []
        for i in range(len(self.linears)-1):
            x = self.linears[i](x)
            x = self.dropout(x)
            x = torch.relu(x)
            z.append(x)

        y = self.linears[-1](x)
        y = y.squeeze()

        return z, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        sparse_codes, pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        train_acc = self.accuracy(pred, y)

        self.log('train_loss', loss)
        self.log('train_acc', train_acc, on_epoch=True)

        # log sparsity
        for i in range(len(sparse_codes)):
            num_nonzero = torch.count_nonzero(sparse_codes[i], dim=-1).float()
            percent_nonzero = torch.mean(num_nonzero/sparse_codes[i].shape[-1])
            self.log('train_num_nonzero_layer' + str(i+1), torch.mean(num_nonzero))
            self.log('train_percent_nonzero_layer'+str(i+1), percent_nonzero)
             
        return loss
 
    def validation_step(self, batch, batch_idx):       
        x, y = batch
        
        sparse_codes, pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        val_acc = self.accuracy(pred, y)

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', val_acc, on_epoch=True)

        # log sparsity
        for i in range(len(sparse_codes)):
            num_nonzero = torch.count_nonzero(sparse_codes[i], dim=-1).float()
            percent_nonzero = torch.mean(num_nonzero/sparse_codes[i].shape[-1])
            self.log('val_num_nonzero_layer' + str(i+1), torch.mean(num_nonzero))
            self.log('val_percent_nonzero_layer'+str(i+1), percent_nonzero)

        return {'sparse_code': sparse_codes, 'label': y} 
    

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        # sch = torch.optim.lr_scheduler.CyclicLR(opt, 1e-7, 1e-3, step_size_up=1, cycle_momentum=False)
        return opt #([opt], [sch])

        # opt_params = [layer.bias for layer in self.linears]
        # opt_params.append(self.linears[-1].weight)
        # opt = torch.optim.Adam(opt_params, lr=1e-2)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
        # {'optimizer':opt, 'lr_scheduler':scheduler, 'monitor':'val_loss'}
        # return torch.optim.Adam(self.parameters(), lr=1e-4)
        

def load_data(dataset, n_classes):

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train = torch.from_numpy(x_train).float().reshape(-1, 28 * 28) / 255
    y_train = torch.from_numpy(y_train).long().squeeze()
    x_test = torch.from_numpy(x_test).float().reshape(-1, 28 * 28) / 255
    y_test = torch.from_numpy(y_test).long().squeeze()

    x_train, y_train = x_train[y_train < n_classes], y_train[y_train < n_classes]
    x_test, y_test = x_test[y_test < n_classes], y_test[y_test < n_classes]

    return x_train, y_train, x_test, y_test


# train_feats is a list of length # layers, each element of list is tensor of size (train_samples, hidden_dim)
def get_nonzero_feat_idx(train_feats, y_train, class_idx):
    d = []
    for i in range(len(train_feats)):
        x = train_feats[i]
        x = x[y_train == class_idx]
        x[x > 0] = 1
        idx = torch.mean(x, dim=0)
        idx[idx < 0.5] = 0
        # idx[idx < 1e-2] = 0
        # idx = idx.nonzero().flatten()
        d.append(idx)
        
    return d

def jaccard_similarity(tensor1, tensor2):
    """
    Compute the Jaccard similarity between two tensors.

    Args:
    - tensor1 (torch.Tensor): First input tensor.
    - tensor2 (torch.Tensor): Second input tensor.

    Returns:
    - float: Jaccard similarity.
    """

    # Compute the intersection of non-zero elements
    intersection = np.intersect1d(tensor1, tensor2).shape[0]

    # Compute the union of non-zero elements
    union = np.union1d(tensor1, tensor2).shape[0]

    # Compute Jaccard similarity
    jaccard_similarity = intersection / union if union != 0 else 0.0

    return jaccard_similarity

def save_tensor_as_csv(tensor, file_path):
    """
    Save a 2D PyTorch tensor as a CSV file.

    Args:
    - tensor (torch.Tensor): Input 2D tensor.
    - file_path (str): File path to save the CSV file.

    Returns:
    - None
    """
    
    # Convert the PyTorch tensor to a NumPy array
    numpy_array = tensor.numpy()

    # Save the NumPy array to a CSV file
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(numpy_array)
