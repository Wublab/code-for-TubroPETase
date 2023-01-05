import torch
from torch import nn
from torch.optim import Adam
from mlm_pytorch import MLM
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tqdm import tqdm
# instantiate the language model

from x_transformers import TransformerWrapper, Encoder


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_file, vocab_file):
        self.sequences = self.read_fasta(fasta_file)
        self.read_vocab(vocab_file)
        super(SeqDataset, self).__init__()
    
    def __len__(self):
        return len(self.sequences)

    def read_vocab(self, vocab_file):
        self.vocab = {}
        with open(vocab_file, 'r') as vocab_file:
            for i, line in enumerate(vocab_file):
                word = line.replace("\n", "")
                self.vocab[word] = i
        return self.vocab

    def read_fasta(self, fasta_file):
        self.fasta_dict = {}
        with open(fasta_file, 'r') as fasta_file:
            for line in fasta_file:
                if line.startswith(">"):
                    name = line.replace("\n", "")
                    self.fasta_dict[name] = ""
                else:
                    self.fasta_dict[name] += line.replace("\n", "")
        return list(self.fasta_dict.values())

    def tokenizer(self, seq):
        x = torch.zeros(512)
        x[0] = 2
        seq = seq[:510]
        for i, aa in enumerate(seq):
            w = self.vocab.get(aa)
            if w:
                x[i] = w
            else:
                x[i] = 1

        x[len(seq)+1] = 3
        return x.long()

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        inp = self.tokenizer(seq)
        return inp

dataset = SeqDataset('../clean_90.fasta', 'vocab.txt')

validation_split = .1
shuffle_dataset = True
random_seed= 42

batch_size = 32

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


transformer = TransformerWrapper(
    num_tokens = 25,
    max_seq_len = 512,
    attn_layers = Encoder(
        dim = 512,
        depth = 3,
        heads = 8
    )
)

trainer = MLM(
    transformer,
    mask_token_id = 4,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    mask_ignore_token_ids = [2,3]  # other tokens to exclude from masking, include the [cls] and [sep] here
).to(device)

# optimizer

opt = Adam(trainer.parameters(), lr=3e-4)

# one training step (do this for many steps in a for loop, getting new `data` each time)

# data = torch.randint(0, 24, (8, 1024)).cuda()

total_epoch = 100

def eval(test_loader):
    val_loss = []
    with torch.no_grad():
        for test_data in test_loader:
            val_loss.append(trainer(test_data.to(device)).cpu())
    return np.mean(val_loss)


for epoch in range(total_epoch):
    with tqdm(train_loader, unit='epoch') as tepoch:
        val_loss = eval(validation_loader)
        for data in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            loss = trainer(data.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()
            # print(f"train loss: {loss}")
            # train_loss[i] = loss
            tepoch.set_postfix(loss=loss.item(),val_loss=val_loss)
        # check_heatmap()
        torch.save(transformer, f'./cubert_{epoch}.pt')
        

