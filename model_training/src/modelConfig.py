from pydantic import BaseModel

class ModelConfig(BaseModel):
     dataset: str 
     vocab_size: int
     batch_size: int
     seq_len: int
     head_size: int
     embed_dims: int
     block_num: int
     lr: float
     iterations: int
     cpu_cores: int
     
     