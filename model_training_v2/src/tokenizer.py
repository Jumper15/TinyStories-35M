from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

def yield_training_dataset(dataset):
     for text in dataset['train']['text']:
          yield text

def train_bpe_tokenizer(dataset, vocab_size):
     ds = load_dataset(dataset)
     trainer = BpeTrainer(
          vocab_size=vocab_size,
          min_frequency=2,
          special_tokens=["[UNK]"],
          initial_alphabet=ByteLevel.alphabet()
     )
     
     tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
     tokenizer.normalizer = Sequence([NFKC()])
     tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
     
     # tokenizer training
     tokenizer.train_from_iterator(
          yield_training_dataset(ds), 
          trainer=trainer
     )
     
     tokenizer.decoder = ByteLevelDecoder()
     
     hf_tokenizer = PreTrainedTokenizerFast(
          tokenizer_object=tokenizer,
          unk_token="[UNK]"
     )
     
     hf_tokenizer.save_pretrained("./TinyStories_BPE_8K")
     
     return hf_tokenizer