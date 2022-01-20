from pytorch_lightning.loops import Loop
from utils import get_batch, transfomer_mask

class TrainingEpochLoop(Loop):

    def __init__(self, model, lossfunc, optimizer, data_iter, bptt, device="cpu"):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.data_iter = data_iter
        self.batch_idx = 0
        self.bptt = bptt
        self.device = device
        # Model
        self.model.train()
        # Loss function
        self.loss_func = lossfunc
        # transformer Mask

        self.transf_mask = transfomer_mask(self.bptt).to(device)

    @property
    def done(self):
        return self.batch_idx >= self.data_iter.size(0)-1

    def reset(self) -> None:
        self.data = self.data_iter

    def advance(self, *args, **kwargs) -> None:
        batch = get_batch(self.data, self.batch_idx)
        batch[0], batch[1] = batch[0].to(self.device), batch[1].to(self.device)
        batch_size = batch[0].size(0)
        if batch_size != self.bptt:  # only on last batch
          self.transf_mask = self.transf_mask[:batch_size, :batch_size]
        # Zero the gradients
        self.optimizer.zero_grad()
        # Call training step function: every call equals to a full loop over the dataset.
        loss = self.model.training_step(
             batch, self.transf_mask, self.loss_func)
        self.batch_idx+=self.bptt

        if self.batch_idx %10==0:
          print(loss.item())
        loss.backward()
        self.optimizer.step()