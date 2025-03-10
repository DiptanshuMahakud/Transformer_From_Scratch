# Implementing the Transformer Model from Scratch
# --------------------------------------------------
# This is a complete implementation of the Transformer model as introduced in the paper 
# "Attention Is All You Need" (Vaswani et al., 2017).
#
# References I used for understanding and implementing this model:
# 1. The original paper: "Attention Is All You Need"
# 2. CampusX YouTube Channel - for intuitive understanding of the architecture.
#
# This implementation closely follows the architecture described in the paper with a few 
# minor tweaks and optimizations for simplicity and clarity. 
# The project is divided into the following components:
# 1. Embedding Layer: Converts input tokens to vectors
# 2. Positional Encoding: Adds positional information to the input embeddings
# 3. Multi-Head Attention Block: The core building block of the Transformer model
# 4. Feed-Forward Block: A simple feed-forward neural network
# 5. Add & Norm: Residual connection followed by layer normalization
# 6. Encoder Block: A single block of the Transformer encoder
# 7. Encoder: The full Transformer encoder composed of multiple encoder blocks
# 8. Decoder Block: A single block of the Transformer decoder
# 9. Decoder: The full Transformer decoder composed of multiple decoder blocks
# 10. Transformer: The complete Transformer model composed of an encoder and decoder


import torch
import torch.nn as nn
import math


# Defining the embedding layer
class EmbeddingLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, padding_idx = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)))


# Defining the positional encoding
# Learned Positional Encoding Formula:
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# 
# However, for computational efficiency, we rewrite it using exponentials: 
# x = e^(log(x)). Just a neat math trick to simplify coding it.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # Trim div_term if odd
        
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]  # Auto-truncate to input sequence length
        return self.dropout(x)  

# Multi-Head Attention Block
# Designed to be versatile — can function as:
# 1. Standard Multi-Head Attention (Encoder Self-Attention)
# 2. Masked Multi-Head Attention (for Decoder Self-Attention)
# 3. Cross-Attention (for Encoder-Decoder Attention)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers to project input to Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask = None):
        # Scaled Dot-Product Attention
        scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Applying additive mask
        attention_scores = scores.softmax(dim=-1)  
        return self.dropout(attention_scores @ v), attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask = None):
        # Linear projections to create query , key and value
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        #(batch_size , seq_len , d_model) -> (batch_size , seq_len , num_heads , d_model) -> (batch_size , heads , seq_len , d_model)
        query = query.view(query.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        key = key.view(key.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()

        # Applying attention
        out, attention_scores = self.attention(query, key, value, mask)

        # Concatenate heads back
        #(batch_size , heads , seq_Len , d_model) -> (batch_size , seq_len , heads , d_k) -> (batch_size , seq_len, d_model)
        # contiguous means that after transposing the values are fragmented so we will make it a continuous memory block
        out = out.transpose(1, 2).contiguous().view(out.shape[0], -1, self.d_model)
        return self.w_o(out), attention_scores


# Feed Forward Block
# This block is applied after the attention mechanism to further transform the token embeddings.
# It consists of two linear layers with a GELU activation in between (instead of ReLU).
# GELU is a smoother non-linearity that tends to perform better than ReLU in practice.

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()  
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)

# Defining the Add & Norm block
# It takes the output of a previous sublayer as residual connection
# Adds it to our current flow and applies Layer Normalization.

class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout) 

    def forward(self, x: torch.Tensor, sublayer_outputs: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(sublayer_outputs)
        return self.norm(x)
    
# Defining the Encoder Block'
# This uses all our previously defined components to create a single encoder block.
# Just simple connections based on the paper.

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.multihead_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        attention_out, _ = self.multihead_attention(x, x, x, mask)
        x = self.add_norm1(x, attention_out)
        feed_forward_out = self.feed_forward(x)
        return self.add_norm2(x, feed_forward_out)

# Defining the Encoder
# This is the encoder of the Transformer model , composed of multiple encoder blocks stacked on top of each other.

class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, seq_len: int, n_layers: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.embeddings = EmbeddingLayer(d_model, vocab_size, dropout)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask= None) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Defining the Decoder Block
# This is quite similar to the encoder blocks in some way but here we make use of Masked Attention Mechanism and Cross Attention Mechanism
# Masked Attention Mechanism is used to prevent the model from peeking into the future during training.
# Cross Attention Mechanism is used to allow the decoder to focus on different parts of the input sequence.
 
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, encoder_outs: torch.Tensor, mask = None) -> torch.Tensor:
        # Masked self-attention + AddNorm
        masked_attention_out, _ = self.masked_attention(x, x, x, mask)
        x = self.add_norm1(x, masked_attention_out)
        # Encoder-decoder attention + AddNorm
        cross_attention_out, _ = self.cross_attention(x, encoder_outs, encoder_outs)
        x = self.add_norm2(x, cross_attention_out)
        # Feed-forward + AddNorm
        feed_forward_out = self.feed_forward(x)
        return self.add_norm3(x, feed_forward_out)

# Defining the Decoder
# This is quite similar to Encoder of the Transformer , simply stacking the decoder blocks
# And finally we add a final Linear layer and softmax to produce out logits.

class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, seq_len: int, n_layers: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.embeddings = EmbeddingLayer(d_model, vocab_size, dropout)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, encoder_outs: torch.Tensor, mask = None) -> torch.Tensor:
        # Embeddings + positional encoding
        x = self.embeddings(x)
        x = self.positional_encoding(x)
        # Decoder blocks
        for layer in self.layers:
            x = layer(x, encoder_outs, mask)
        # Final linear layer (no softmax, handled by loss function)
        return self.final_linear(x)

# Defining the Transformer
# We simply combine the Encoder and Decoder to create our very own Transformer model.
class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, seq_len: int, n_layers: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, d_ff, seq_len, n_layers, vocab_size, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, seq_len, n_layers, vocab_size, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask = None, tgt_mask = None) -> torch.Tensor:
        # Encode source sequence
        encoder_outs = self.encoder(src, src_mask)
        # Decode target sequence using encoder outputs
        logits = self.decoder(tgt, encoder_outs, tgt_mask)
        return logits