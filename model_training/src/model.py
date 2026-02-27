import torch
import torchtune.modules as Module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp as amp

class Attention_Head(nn.Module):
  def __init__(self, embed_dims, head_size, num_heads=8):
    super().__init__()
    self.embed_dims = embed_dims
    self.num_heads = num_heads
    self.head_size = head_size
    self.total_heads = head_size * num_heads

    self.q_proj = nn.Linear(embed_dims, self.total_heads)
    self.k_proj = nn.Linear(embed_dims, self.total_heads)
    self.v_proj = nn.Linear(embed_dims, self.total_heads)

    self.pe = Module.RotaryPositionalEmbeddings(self.head_size)

  def forward(self, logits, batch_size, seq_len):
    q = self.q_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)
    k = self.k_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)

    q_pe = self.pe(q)
    k_pe = self.pe(k)

    v = self.v_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)

    attention_out = F.scaled_dot_product_attention(q_pe, k_pe, v, is_causal=True)
    out = attention_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.total_heads)
    return out

class FFN(nn.Sequential):
  def __init__(self, embed_dims, hidden_dim):
    super().__init__(
        nn.Linear(embed_dims, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, embed_dims)
    )

class Block(nn.Module):
  def __init__(self, embed_dims, head_size):
    super().__init__()
    self.embed_dims = embed_dims
    self.head_size = head_size

    self.rms_Norm1 = nn.RMSNorm(embed_dims)
    self.rms_Norm2 = nn.RMSNorm(embed_dims)

    self.Attention_Head = Attention_Head(embed_dims, head_size)
    self.FFN = FFN(embed_dims, embed_dims*4)

  def forward(self, logits, batch_size, seq_len):

    x = self.Attention_Head(self.rms_Norm1(logits), batch_size, seq_len)
    x = x + logits
    out = self.FFN(self.rms_Norm2(x))
    out = out + x
    return out

class Model(nn.Module):
  def __init__(self, batch_size, seq_len, embed_dims, head_size, block_num, vocab_size, lr, iterations):
    super().__init__()
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.embed_dims = embed_dims
    self.head_size = head_size
    self.vocab_size = vocab_size

    self.token_embed = nn.Embedding(vocab_size, embed_dims)
    self.rms_Norm_embed = nn.RMSNorm(embed_dims)
    self.embed_proj = nn.Linear(embed_dims, vocab_size)
    self.block_list = nn.ModuleList([Block(embed_dims, head_size) for _ in range(block_num)])

  def forward(self, input, targets=None):
    batch_size, seq_len = input.shape
    logits = self.token_embed(input)

    for block in self.block_list:
      logits = block(logits, batch_size, seq_len)

    unembed_out = self.embed_proj(self.rms_Norm_embed(logits))

    if targets != None:
      preds = unembed_out.view(self.batch_size*self.seq_len, -1)
      targets = targets.view(-1)

      loss_fn = F.cross_entropy(preds, targets)
      return loss_fn

    return unembed_out

  def generate(self, input_tokens, max_tokens):
      for _ in range(max_tokens):
          last_seq = input_tokens[:, -self.seq_len:]
          logits = self(last_seq)
          logits = logits[:, -1, :]
          probs = F.softmax(logits, dim=-1)
          next_tok = torch.multinomial(probs, num_samples=1)
          input_tokens = torch.cat((input_tokens, next_tok), dim=1)
      return input_tokens
 