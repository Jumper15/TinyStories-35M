from modelConfig import ModelConfig
from model import Model

import torch
from transformers import PreTrainedTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = ModelConfig(
     dataset="roneneldan/TinyStories", # ds, or 
     vocab_size=2048, # vocab size
     batch_size=572, # batch size,
     seq_len=512, # seq len
     head_size=64, # head_size
     embed_dims=512, # embed dims
     block_num=12, # block nums
     lr=1e-3, # lr
     iterations=10000,
     cpu_cores=170
)

if __name__ == "__main__":
    model = Model(
        model_config.batch_size,
        model_config.seq_len,
        model_config.embed_dims,
        model_config.head_size,
        model_config.block_num,
        model_config.vocab_size,
        model_config.lr,
        model_config.iterations
    )
    
    model = model.to(device)
    model.load_state_dict(torch.load("model_state/TinyStories_4000.pt", map_location=torch.device('cpu')))
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained("TinyStories_BPE_8K")  
    # encoded_prompt = tokenizer("The brick building was in front of the boy. He then walked ", return_tensors='pt')
    encoded_prompt = tokenizer("Pip the small brown rabbit had never once left the meadow. Every morning she watched the golden sun rise over Bumble Hill and wondered what lay beyond. One Tuesday, a tiny rolled-up map appeared outside her burrow door, tied neatly with a piece of bright red string.", return_tensors='pt')
    input_toks = encoded_prompt['input_ids'].squeeze(0)
    input_toks = torch.unsqueeze(input_toks, dim=0).to(device)
    out_toks = model.generate(input_toks, 512)
    print(tokenizer.decode(out_toks[0].tolist()))
