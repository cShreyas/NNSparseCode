from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from tensorflow.keras.datasets import fashion_mnist, mnist
from utils import *
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--classes', type=int)
parser.add_argument('--layers', type=int)
parser.add_argument('--hidden', type=int)

args = parser.parse_args()
name = args.name
n_classes = args.classes
n_layers = args.layers
hidden = args.hidden

x_train, y_train, x_test, y_test = load_data(mnist, n_classes)
train_ds, test_ds = TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
train_loader =  DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8)
test_loader =  DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=4)
model = MLP(input_size=28*28, num_layers=n_layers, layers_size=hidden, output_size=n_classes, dropout=0.2)

logger = TensorBoardLogger('logs/MNIST/', name=name)
lr_monitor = LearningRateMonitor(logging_interval='step') 
trainer = pl.Trainer(
    logger = logger,
    max_epochs=100,
    accelerator="gpu",
    devices=[2],
    log_every_n_steps=1,
    callbacks=[lr_monitor],
    enable_checkpointing=True
)
logger.log_hyperparams({"epochs":100, "optimizer":"Adam", "num_layers": n_layers, "layers_size": hidden, "num_classes":n_classes})
trainer.fit(model, train_loader, test_loader)