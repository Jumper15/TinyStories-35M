from modelConfig import ModelConfig

from model import Model
from dataloader import DataLoader

from tokenizer import train_bpe_tokenizer
from training import model_pretraining

import wandb
import torch
import datasets
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# entrypoint
## example model config
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
     print("start...")
     torch.set_float32_matmul_precision('high')
     
     ds = load_dataset(model_config.dataset)

     # trained custom BPE tokenizer on dataset
     tokenizer = train_bpe_tokenizer(
          model_config.dataset, 
          model_config.vocab_size
     )

     # run = wandb.init(
     #     entity="chordataosteichthyes-personal",
     #     project="se-model-training",
     #     config={
     #         "learning_rate": model_config.lr,
     #         "architecture": "Transformer",
     #         "dataset": model_config.dataset,
     #         "epochs": model_config.iterations,
     #     }
     # )

     # tokenizer = PreTrainedTokenizerFast.from_pretrained("TinyStories_BPE_8K")

     print("initializing model")
     model = Model(
          model_config.batch_size,
          model_config.seq_len,
          model_config.embed_dims,
          model_config.head_size,
          model_config.block_num,
          model_config.vocab_size,
          model_config.lr,
          model_config.iterations,
     )
     # model.load_state_dict(torch.load("model_state/prev/TinyStories_1000.pt"))
     model = model.to(device)
     
     # initialize dataloader
     dataloader = DataLoader(
          tokenizer,
          ds['train'],
          model_config.cpu_cores
     )

     print("calling model training")
     model_pretraining(
          model, 
          dataloader, 
          model_config.iterations,
          model_config.batch_size,
          model_config.seq_len,
          model_config.lr
     )

     # run.finish()
     
     # torch.save(model.state_dict(), "model_state/TinyStories-20k.pt")
     