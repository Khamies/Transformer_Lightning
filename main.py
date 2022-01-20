import math
import argparse
import torch


from dataset import PennTreeBank
from model import TransformerLang_Lightning
from train import TrainingEpochLoop

from settings import global_setting, model_setting, training_setting


# Global Settings

torch.manual_seed(global_setting["seed"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




parser =  argparse.ArgumentParser(description=" A parser for transformer based language model")
parser.add_argument("--bsz_train", type=str, default="32")
parser.add_argument("--bsz_test", type=str, default="32")
parser.add_argument("--bptt", type=str,default="35")
parser.add_argument("--embed_size", type=str, default="300") 
parser.add_argument("--ffnn_size", type=str, default="600")
parser.add_argument("--nhead", type=str, default="2")
parser.add_argument("--nlayers", type=str, default="2")
parser.add_argument("--lr", type=str, default="0.001")


# Extract commandline arguments   
args = parser.parse_args()

train_batch_size = int(args.bsz_train) if args.bsz_train!=None else  training_setting["bsz_train"]
test_batch_size = int(args.bsz_test) if args.bsz_test!=None else  training_setting["bsz_test"]
bptt = int(args.bptt) if args.bptt!=None else  training_setting["bptt"]
embed_size_transformer = int(args.embed_size) if args.embed_size!=None else  training_setting["embed_size"]
ffnn_size_transformer = int(args.ffnn_size) if args.ffnn_size!=None else  training_setting["ffnn_size"]
nhead_transformer = int(args.nhead) if args.nhead!=None else  training_setting["nhead"]
nlayers_transformer = int(args.nlayers) if args.nlayers!=None else  training_setting["nlayers"]
lr = float(args.lr) if args.lr!=None else  training_setting["lr"]
dropout = model_setting["dropout"]


dataset = PennTreeBank(train_batch_size, test_batch_size)
dataset.prepare_data()
dataset.setup()
train_data = dataset.train_dataloader()
test_data = dataset.test_dataloader()
val_data = dataset.val_dataloader()
vocab_size = dataset.vocab_size

model = TransformerLang_Lightning(vocab_size, embed_size_transformer, nhead_transformer, ffnn_size_transformer, nlayers_transformer, dropout).to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


if __name__ == "__main__":
    for epoch in range(10):
        TrainingEpochLoop(model,loss, optimizer, train_data, bptt, clip=training_setting["clip"]).run()