# TinyStories-35M

A SLM trained the dataset roneneldan/TinyStories (https://arxiv.org/abs/2305.07759) as part of a school project. Trained on ~2 hours H100 SXM
Model Details: 
     Multiheaded Attention (Pytorch scaled_dot_product_attention)
     RoPE (torchtune implementation)
     FFN SiLU activation
     HF BPE Tokenizer
     
Model Config:
     dataset: roneneldan/TinyStories
     vocab_size: 2048 
     batch_size: 572 
     seq_len: 512
     head_size: 64 
     embed_dims: 512
     block_num: 12
     lr: 1e-3 
     steps: 10000 
