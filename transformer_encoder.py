#%%
import torch as th
import math
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from models import MLP 
import torch.nn.functional as F
from torch.utils.data import DataLoader

#%%
class ScaledDotMultiHeadAttention(nn.Module):
    def __init__(self, token_dim, key_dim, value_dim, nheads):
        super().__init__()
        self.key_projector = nn.Linear(token_dim, key_dim*nheads)
        self.value_projector = nn.Linear(token_dim, value_dim*nheads)
        self.query_projector = nn.Linear(token_dim, key_dim*nheads)
        self.unify_heads = nn.Linear(value_dim*nheads, token_dim)
        self.nheads = nheads
        self.key_dim = key_dim
    def forward(self, x, mask=None):
        b = x.shape[0] #batch
        t = x.shape[1] #time/number in sequence
        h = self.nheads

        #project to batch x time x nheads x keyOrValueDim
        key   = self.key_projector(x).view(b, t, h, -1)
        query = self.query_projector(x).view(b, t, h, -1)
        value = self.value_projector(x).view(b, t, h, -1)

        #order the dimensions to use parallel matrix multiply implementations
        key = key.transpose(1, 2).reshape(b*h, t, -1)
        value = value.transpose(1, 2).reshape(b*h, t, -1)
        query = query.transpose(1, 2).reshape(b*h, t, -1)

        product_matrix = th.bmm(query, key.transpose(1, 2))/math.sqrt(self.key_dim)
        if mask is not None:
            product_matrix = product_matrix.masked_fill(mask, float('-inf'))
        attention_matrix = th.softmax(product_matrix, dim=-1)
        result = th.bmm(attention_matrix, value)

        catted_head_outputs = result.view(b, h, t, -1).transpose(1, 2).reshape(b, t, -1)
        return self.unify_heads(catted_head_outputs)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len):
        super().__init__()
        pe = th.zeros(1, max_seq_len, model_dim)
        for pos in range(max_seq_len):
            for i in range(model_dim):
                if i%2:
                    pe[0, pos, i] = math.cos(pos/(10000**(2*i/model_dim)))
                else:
                    pe[0, pos, i] = math.sin(pos/(10000**(2*i/model_dim)))
        self.pe = pe
    def forward(self, sequence_len):
        return self.pe[:, :sequence_len]

class MLP(nn.Module):
    def __init__(self, net_arch, last_activation = lambda x: x):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
        self.last_activation = last_activation
    def forward(self, h):
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        return self.last_activation(h)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, token_dim, key_dim, value_dim, nheads, mlp_hidden_dim):
        super().__init__()
        self.attention = ScaledDotMultiHeadAttention(token_dim, key_dim, value_dim, nheads)
        self.layer_norm_attn = nn.LayerNorm(token_dim)
        self.mlp = MLP([token_dim, mlp_hidden_dim, token_dim], last_activation=F.relu)
        self.layer_norm_mlp = nn.LayerNorm(token_dim)
    def forward(self, x, mask=None):
        h = self.attention(x, mask) + x
        h = self.layer_norm_attn(h)
        h = self.mlp(h) + h
        return self.layer_norm_mlp(h)

class TransformerEncoder(nn.Module):
    def __init__(self, token_dim, key_dim, value_dim, nheads, mlp_hidden_dim, max_seq_len, nblocks):
        super().__init__()
        self.positional_encoding = PositionalEncoding(token_dim, max_seq_len)
        l = [TransformerEncoderBlock(token_dim, key_dim, value_dim, nheads, mlp_hidden_dim) for _ in range(nblocks)]
        self.blocks = nn.ModuleList(l)
    def forward(self, x, mask=None):
        h = x + self.positional_encoding(x.shape[1])
        for block in self.blocks:
            h = block(h, mask)
        return h

class Model(nn.Module):
    def __init__(self, token_dim, key_dim, value_dim, nheads, mlp_hidden_dim, max_seq_len, nblocks, num_embeddings):
        super().__init__()
        self.encoder = TransformerEncoder(token_dim, key_dim, value_dim, nheads, mlp_hidden_dim, max_seq_len, nblocks)
        self.embedding = nn.Embedding(num_embeddings, token_dim)
        self.lin = nn.Linear(token_dim, num_embeddings)
    def forward(self, x, mask=None):
        h = self.embedding(x)
        h = self.encoder(h, mask)
        h = self.lin(h)
        h = th.softmax(h, dim=-1)
        return h

model = Model(32, 16, 16, 3, 64, 10, 4, 10)
opt = th.optim.Adam(model.parameters(), lr=3e-4)
#%%
seq_len = 6
e = 1e-8
mask = th.triu(th.ones(1, seq_len, seq_len)>0, diagonal=1)
for ep in range(30):
    for ibatch, (x, y) in enumerate(dataloader):
        preds = model(x, mask=mask)[:, -3:]
        yhot = F.one_hot(y, num_classes=10)[:, -3:]*1.0
        loss = yhot*th.log(preds+e) + (1-yhot)*th.log(1-preds+e)
        loss = -loss.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ibatch %10==0:
            print(ep, ibatch, loss.item())
#%%
for ibatch, (x, y) in enumerate(tdataloader):
    preds = model(x, mask=mask)[:, -3:]
    print(preds)
    img = preds[0].detach().numpy()
    plt.imshow(img)
    print(th.argmax(preds, dim=-1))
    print(y)
    print(x)
    break
# %%
from torch.utils.data import Dataset

class AdditionDataset(Dataset):
    """
    Returns addition problems of up to some number of digits in the inputs. Recall
    that all GPT cares about are sequences of integers, and completing them according to
    patterns in the data. Therefore, we have to somehow encode addition problems
    as a sequence of integers.
    
    The sum of two n-digit numbers gives a third up to (n+1)-digit number. So our
    encoding will simply be the n-digit first number, n-digit second number, 
    and (n+1)-digit result, all simply concatenated together. Because each addition
    problem is so structured, there is no need to bother the model with encoding
    +, =, or other tokens. Each possible sequence has the same length, and simply
    contains the raw digits of the addition problem.
    
    As a few examples, the 2-digit problems:
    - 85 + 50 = 135 becomes the sequence [8, 5, 5, 0, 1, 3, 5]
    - 6 + 39 = 45 becomes the sequence [0, 6, 3, 9, 0, 4, 5]
    etc.
    
    We will also only train GPT on the final (n+1)-digits because the first
    two n-digits are always assumed to be given. So when we give GPT an exam later,
    we will e.g. feed it the sequence [0, 6, 3, 9], which encodes that we'd like
    to add 6 + 39, and hope that the model completes the integer sequence with [0, 4, 5]
    in 3 sequential steps.
    
    fun exercise: does it help if the result is asked to be produced in reverse order?
    """

    def __init__(self, ndigit, split):
        self.split = split # train/test
        self.ndigit = ndigit
        self.vocab_size = 10 # 10 possible digits 0..9
        # +1 due to potential carry overflow, but then -1 because very last digit doesn't plug back
        self.block_size = ndigit + ndigit + ndigit + 1 - 1
        
        # split up all addition problems into either training data or test data
        num = (10**self.ndigit)**2 # total number of possible combinations
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def __len__(self):
        return self.ixes.size

    def __getitem__(self, idx):
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx]
        nd = 10**self.ndigit
        a = idx // nd
        b = idx %  nd
        c = a + b
        render = f'%0{self.ndigit}d%0{self.ndigit}d%0{self.ndigit+1}d' % (a,b,c) # e.g. 03+25=28 becomes "0325028" 
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        # y[:self.ndigit*2-1] = -100 # we will only train in the output locations. -100 will mask loss to zero
        return x, y
# %%
import numpy as np
# create a dataset for e.g. 2-digit addition

ndigit = 2
train_dataset = AdditionDataset(ndigit=ndigit, split='train')
test_dataset = AdditionDataset(ndigit=ndigit, split='test')
# %%
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
train_dataset[0] # sample a training instance just to see what one raw example looks like

# %%
mask.shape
# %%
a = th.randn(2, 1, 6, 6)
a.masked_fill(mask, float('-inf'))
# %%
tdataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, pin_memory=True)