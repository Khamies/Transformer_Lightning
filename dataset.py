from pytorch_lightning.core import LightningDataModule

from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from utils import textTotensor, batchify_by_cutting

class PennTreeBank(LightningDataModule):

  def __init__(self, train_batch_size=32, test_batch_size=10):
    super().__init__()
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
  
  def prepare_data(self):

    # Build vocabulary
    train_iter_temp = PennTreebank(split='train')
    self.tokenizer = get_tokenizer('basic_english')
    self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter_temp), specials=['<unk>'])
    self.vocab.set_default_index(self.vocab['<unk>']) 

    # Tokenize the data
    train_iter, val_iter, test_iter = PennTreebank()
    self.train_data = textTotensor(train_iter, self.tokenizer, self.vocab)
    self.val_data = textTotensor(val_iter, self.tokenizer, self.vocab)
    self.test_data = textTotensor(test_iter, self.tokenizer, self.vocab)

    # Batchifying the data
    self.train_data = batchify_by_cutting(self.train_data, self.train_batch_size)  # shape [seq_len, batch_size]
    self.val_data = batchify_by_cutting(self.val_data, self.test_batch_size)
    self.test_data = batchify_by_cutting(self.test_data, self.test_batch_size)
  
  def setup(self):
    self.vocab_size = len(self.vocab)

  
  def train_dataloader(self):
    return self.train_data
  
  def test_dataloader(self):
    return self.test_data
  
  def val_dataloader(self):
    return self.val_data
  
  