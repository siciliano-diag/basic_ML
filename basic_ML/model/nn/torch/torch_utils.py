import multiprocessing

import torch
import pytorch_lightning as pl

class BaseNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

       self.size_out = size_in, size_out

        self.in_activation = in_activation
        self.in_weights = torch.nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.in_bias = torch.nn.Parameter(torch.Tensor(self.size_in, self.size_out))

        # Initialization
        #torch.nn.init.xavier_uniform_(self.in_bias, gain=1.0)
        #torch.nn.init.xavier_uniform_(self.in_weights, gain=1.0)
        torch.nn.init.uniform_(self.in_bias)
        torch.nn.init.uniform_(self.in_weights)

        forward_func = lambda x: torch.sum(activations[self.in_activation](self.in_part(x)),dim=1)

        if out_activation is not None:
            self.out_weights = torch.nn.Parameter(torch.Tensor(self.size_out))
            self.out_bias = torch.nn.Parameter(torch.Tensor(self.size_out))
            self.out_activation = out_activation
            
            # Initialization
            torch.nn.init.uniform_(self.out_bias,b=self.size_in)
            torch.nn.init.uniform_(self.out_weights,a=-1,b=1)

            self.forward_func = lambda x: activations[self.out_activation](self.out_part(forward_func(x)))
        else:
            self.forward_func = forward_func
        
    def forward(self,x):
        return self.forward_func(x)

    def configure_optimizers(self):
      #lr = 1e-3
      optimizer = torch.optim.AdamW(self.parameters())#, lr=lr)
      return optimizer

    def step(self, batch, batch_idx, split):
        x,y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log(split+'_loss', loss)
        acc = 1-((y_hat>0.5).int() - y.int()).abs().float().mean()
        self.log(split+'_acc', acc)
        return loss, acc

    def training_step(self,batch,batch_idx):
        return self.step(batch,batch_idx,"train")[0]

    def validation_step(self,batch,batch_idx):
        return self.step(batch,batch_idx,"val")

    def test_step(self,batch,batch_idx):
        return self.step(batch,batch_idx,"test")


def prepare_data_loaders(data, loader_params):
    default_loader_params = {batch_size:128, num_workers: multiprocessing.cpu_count(), pin_memory: True, persistent_workers: True, drop_last: {"all": False, "train": True, "val": True, "test": False}}
    loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))

    loaders = {}
    for split in ["all", "train", "val", "test"]:
        if split+"_x" in data:
            td = TensorDataset(torch.Tensor(data[split+"_x"]),torch.Tensor(data[split+"_y"]))
            loaders[split] = DataLoader(td, **loader_params)
    return loaders


def prepare_callbacks(trainer_params):
    callbacks = []
    if "callbacks" in trainer_params:
        for callback_name,callback_params in trainer_params["callbacks"].items():
            callbacks.append(getattr(pl, callback_name)(**callbacks_params))
    return callbacks

def prepare_trainer(trainer_params):
    default_trainer_params = {enable_checkpointing: False, logger: False, accelerator: "auto", devices: "auto"}
    trainer_params = dict(list(default_trainer_params.items()) + list(trainer_params.items()))

    trainer = pl.Trainer(**trainer_params)

    return trainer

def train_nn(trainer, model, loaders):
    trainer.fit(model, loaders["train"], loaders["val"])
    
def test_nn(trainer, model, loaders):
    trainer.test(model, testloader)