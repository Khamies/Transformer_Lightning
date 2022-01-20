import math
import torch
from pytorch_lightning.core import LightningModule
from utils import PositionalEncoding


class TransformerLang_Lightning(LightningModule):
  
  def __init__(self, vocab_size, embed_size,n_att_heads, fnn_dim, n_layers, dropout):
    super().__init__()

    self.d_model = embed_size
    self.vocab_size = vocab_size
    # Embedding Layer
    self.embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
    #Positional Encoding
    self.positional_encoding = PositionalEncoding(d_model= embed_size)
    # Transform Layer
    self.transformerBlock = torch.nn.TransformerEncoderLayer(d_model= embed_size, nhead= n_att_heads,\
                                                             dim_feedforward=fnn_dim,dropout=dropout)
    
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer= self.transformerBlock,num_layers=n_layers)

    self.transformer_decoder = torch.nn.Linear(in_features= embed_size, out_features= vocab_size)

    self.log_softmax = torch.nn.LogSoftmax(dim=2)


  def forward(self, x, x_mask):
    """
    x: Batch of text data --> NxSxV
    """
    x = self.embed(x)  * math.sqrt(self.d_model)
    # positional encoding
    x = self.positional_encoding(x)
    # tranformer
    x = self.transformer_encoder(x, x_mask)
    # decoder
    output = self.transformer_decoder(x)


    return output
  
  def training_step(self, batch, transformer_mask, loss_func):
    data, targets = batch

    output = self(data, transformer_mask)
 
    mloss = loss_func(output.view(-1, self.vocab_size), targets)

    return mloss
  
  def test_step(self):
    pass
  
  def validation_step(self,batch, transformer_mask, loss_func):
    # with torch.no_grad():
    #   data, targets = batch
    #   output = self(data, transformer_mask)
    #   mloss = loss_func(output.view(-1, ntokens), targets)

    #   return mloss
    pass