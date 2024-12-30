''' 
Ref: https://www.youtube.com/watch?v=hvaO120Qy0g&list=PL95p-eWIbW1Hyq30Td0cddmzVsWDCaPAu&index=7 
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import pandas as pd

# get the dataset
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
    with open('sales_textbook.txt', 'wb') as f:
        f.write(requests.get(url).content)

with open('sales_textbook.txt', 'r') as f:
    text = f.read()

#%%
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
max_token_value = tokenized_text.max().item() + 1 

# hyperparameters
batch_size = 4
context_length = 16
d_model = 64
num_heads = 4

# split train and validation
train_idex = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_idex]
valid_data = tokenized_text[train_idex:]

data = train_data
idxs = torch.randint(low=0, high=len(data)-context_length, size=(batch_size,))
x_batch = torch.stack([data[idx:idx+context_length] for idx in idxs])
y_batch = torch.stack([data[idx+1:idx+context_length+1] for idx in idxs])
print(x_batch.shape)
# print('X:', encoding.decode(x_batch[0].numpy()))
# print('Y:', encoding.decode(y_batch[0].numpy()))

#%%
input_embedding_lookup_table = nn.Embedding(max_token_value, d_model) # input_embedding_lookup_table.weight.data
x_batch_embedding = input_embedding_lookup_table(x_batch)
y_batch_embedding = input_embedding_lookup_table(y_batch)
print(x_batch_embedding.shape)

# get positional encoding
position_encoding_loopup_table = torch.zeros(context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
# apply the sine & cosine
import math
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
position_encoding_loopup_table[:, 0::2] = torch.sin(position * div_term)
position_encoding_loopup_table[:, 1::2] = torch.cos(position * div_term)
position_encoding_loopup_table = position_encoding_loopup_table.unsqueeze(0).expand(batch_size, -1, -1) # add batch to the first dimension

# add postional encoding to the input embedding
x = x_batch_embedding + position_encoding_loopup_table
y = y_batch_embedding + position_encoding_loopup_table
print(x.shape)

#%%
# get Q, K, V
Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

Q = Wq(x)
K = Wk(x)
V = Wv(x)
print(Q.shape)

# apply multi head
Q = Q.view(batch_size, context_length, num_heads, d_model//num_heads)
print(Q.shape)
Q = Q.permute(0, 2, 1, 3)
K = K.view(batch_size, context_length, num_heads, d_model//num_heads).permute(0, 2, 1, 3)
V = V.view(batch_size, context_length, num_heads, d_model//num_heads).permute(0, 2, 1, 3)
print(Q.shape)

output = Q @ K.transpose(-2, -1) / math.sqrt(d_model//num_heads) # Scale

# apply mask
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
output = output.masked_fill(mask, float('-inf'))
# print(pd.DataFrame(output[0, 0].detach().numpy()))

# apply softmax
attention_score = F.softmax(output, dim=-1)

# apply attention @ V
A = attention_score @ V
print(A.shape) # batch, head, context_len, feature_dimension

# apply concatenate
A = A.permute(0, 2, 1, 3).reshape(batch_size, context_length, d_model)
print(A.shape)

Wo = nn.Linear(d_model, d_model)
output = Wo(A)
print(output.shape)

# apply residual connection
output = output + x

# apply layer normalization
layer_norm = nn.LayerNorm(d_model)
layer_norm_output = layer_norm(output)

# apply feedforward network
output = nn.Linear(d_model, d_model*4)(layer_norm_output)
output = nn.ReLU()(output)
output = nn.Linear(d_model*4, d_model)(output)
output = output + layer_norm_output
print(output.shape)

output = layer_norm(output)

# apply final linear layer
output = nn.Linear(d_model, max_token_value)(output)
print(output.shape) # batch, context_len, max_token_value

logits = F.softmax(output, dim=-1)
predicted_index = torch.argmax(logits[0,0])
print(encoding.decode([predicted_index]))