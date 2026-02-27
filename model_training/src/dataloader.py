import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DataLoader():
     def __init__(self, tokenizer, dataset, cpu_cores):
          self.tokenizer = tokenizer
          self.dataset = dataset

          def tokenize_function(examples):
              return tokenizer(examples['text'], add_special_tokens=False)
         
          tokenized_ds = dataset.map(
              tokenize_function,
              batched=True,
              num_proc=cpu_cores, 
              remove_columns=['text']
          )
          print("created mapping dataset")

          all_tensors = []
          for example in tokenized_ds:
              all_tensors.append(torch.tensor(example['input_ids'], dtype=torch.long))
          print("finished concatenating tensors")
          self.enc_ds = torch.cat(all_tensors)
          
     def load(self, batch_size, seq_len):
          B = batch_size
          T = seq_len
          ix = torch.randint(len(self.enc_ds) - T, (B,))
          indices = ix.view(-1, 1) + torch.arange(T)
          X = self.enc_ds[indices]
          y = self.enc_ds[indices+1]
          return X.to(device), y.to(device)
     