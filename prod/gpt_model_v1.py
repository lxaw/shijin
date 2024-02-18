#####
# @lxaw
#
# This GPT model was heavily inspired from @karpathy's tutorial.
# In later iterations, we will make this better and more fine-tuned to Japanese.
#


import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttentionHead(nn.Module):
    # a head for self-attention
    #
    def __init__(self,head_size,n_embd,block_size,dropout):
        """
        Initialize a self attention head, taking in
        an embedding dimension (n_embd) and a head size (head_size).
        """
        super().__init__()
        # In self-attention mechanisms, using bias in the linear transformations for keys, queries, and values is generally avoided to prevent the model from learning a positional bias. Positional bias could lead to the model favoring certain positions or patterns in the input sequence, which could be detrimental to its ability to generalize across different sequences or tasks.
        # By not using bias in the linear transformations for keys, queries, and values, the model is encouraged to learn more generalized patterns in the data, allowing it to capture dependencies and relationships between tokens more effectively.

        # key
        self.k = nn.Linear(n_embd,head_size,bias=False)
        # query
        self.q = nn.Linear(n_embd,head_size,bias=False)
        # value
        self.v = nn.Linear(n_embd,head_size,bias=False)

        # In PyTorch, a buffer is a tensor that is meant to be part of the state of a module but is not a parameter (i.e., it's not going to be optimized during training). It's useful for maintaining state that shouldn't be trained, such as running statistics in batch normalization layers or pre-defined constants like in this case.
        # The register_buffer() method is used to register a tensor as a buffer within a PyTorch nn.Module. When you register a buffer, PyTorch manages this tensor as part of the module's state. It ensures that the buffer tensor is moved to the same device as the module's parameters when the module is moved to a GPU or another device. It also ensures that the buffer is included in the module's state dictionary when you save or load the module.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        """
        Forward x through.
        input dims: (batch, time-step, channels)
        output dims: (batch,time-step, head size)
        """
        batch_dim,time_dim,channel_dim = x.shape
        k = self.k(x) # dim: (batch,time-step,head size)
        q = self.query(x) # dim: (batch,time-step,head size)
        # get affinities (attention-scores)
        aff = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch,time,hs) @ (batch,time,hs) -> (batch,time,time)
        aff = aff.mask_fill(self.tril[:time_dim,:time_dim] == 0,float('-inf')) # (batch, time, time)
        aff = F.softmax(aff,dim=-1) # (batch,time,time)
        aff = self.dropout(aff)
        # weighted aggregation of the values
        v = self.v(x) # (batch,time,hs # (batch,time,hs) # (batch,time,hs) # (batch,time,hs))
        out = aff @ v # (batch,time,time) @ (batch,time,hs) -> (batch,time,hs)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multiple self-attention heads in parallel.
    """
    def __init__(self,num_heads,head_size,block_size,dropout,n_embd):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size=head_size,block_size=block_size,dropout=dropout,n_embd=n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads,n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """
    Linear layer followed by non-linearity
    """
    def __init__(self,n_embd,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer block.
    """
    def __init__(self,n_embd,head_size,num_heads,block_size,dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.self_attention = MultiHeadAttention(num_heads=num_heads,head_size=head_size,
                                     block_size=block_size,dropout=dropout,n_embd=n_embd)
        self.ffwd = FeedForward(n_embd=n_embd,dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self,vocab_size,n_embd,block_size,num_heads,head_size,dropout,device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd=n_embd,num_heads =num_heads,head_size=head_size,block_size=block_size,dropout=dropout)]
        )
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd,vocab_size)

        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        """
        Initialization of weights
        """
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module,nn.Embedding):
                torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            
    def forward(self,idx,targets = None):
        batch_dim,time_dim = idx.shape

        # idx and targets are (batch_dim,time_dim) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (batch_dim,time_dim,channel_dim)
        pos_emb = self.position_embedding_table(torch.arange(time_dim,device=self.device))
        x = tok_emb + pos_emb # (batch_dim,time_dim,channel_dim)
        x = self.blocks(x) # (batch_dim,time_dim,channel_dim)

        logits = self.lm_head(x) # (batch_dim,time_dim,vocab_size)

        if targets is None:
            loss = None
        else:
            batch_dim,time_dim,channel_dim = logits.shape
            logits = logits.view(batch_dim*time_dim,channel_dim)
            targets = targets.view(batch_dim*time_dim)
            loss = F.cross_entropy(logits,targets)
        
        return logits, loss
    
    def generate(self,idx,max_new_tokens):
        """
        Generate some text
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

        













