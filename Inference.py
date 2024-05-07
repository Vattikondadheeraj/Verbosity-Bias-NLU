import torch
from tqdm import tqdm


from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm

# import the huggingface transformers libraries
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaForSequenceClassification

#### auto size stuff
import numpy as np
def factors(x):
    return [i for i in range(1,x+1) if x%i==0]

def auto_size(seq_len, topk):
    estimated = (28672/(seq_len*1.5)) -11.52605
    # hack
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)): return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]
###

def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))

# From huggingface
def rcache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past

def even_chunk(data, chunk_size=10):
    assert data.shape[0] % chunk_size == 0, "chunk_size must evenly divide the topk"
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:(i+chunk_size)]

# reward based search
class ARGS:
    def __init__(self, llm_path, rm_path, llm_dev="cuda:0", rm_dev="cuda:0", torch_dtype=torch.float16):
        self.llm_dev = llm_dev
        self.rm_dev = rm_dev
        print("Loading LLM...")
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype).to(self.llm_dev)
        self.LLM.eval()
        
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)

    def get_input_ids(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.llm_dev)
        return tokens
    
    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)



    def  generate(self, prompt, max_new_token=128, temperature=0.7, debug=False):
        tokens = self.get_input_ids(prompt)
        initial_len = tokens.shape[-1]
    
    # Check if the sequence length exceeds the LLM's maximum sequence length
        if tokens.shape[-1] > self.LLM.config.to_dict().get("max_sequence_length", 2048):
            print("The sequence of tokens is too long!!! Returning None!")
            return None

        cached = None  # Used for caching past key values for efficient generation
    
        iterator_obj = range(max_new_token)
        if debug: 
            iterator_obj = tqdm(iterator_obj)
    
        for _ in iterator_obj:
            with torch.no_grad():
                if cached is None:
                    inputs = self.LLM.prepare_inputs_for_generation(
                    input_ids=tokens, 
                    attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), 
                    past_key_values=None, 
                    use_cache=True
                )
                else:
                    inputs = self.LLM.prepare_inputs_for_generation(
                    input_ids=tokens, 
                    attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), 
                    past_key_values=cached, 
                    use_cache=True
                )

                mout = self.LLM(**inputs)
                cached = mout.past_key_values
                logits = mout.logits  # Assuming this is the attribute containing the logits
                probs = torch.nn.functional.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=-1)
                del mout  # Free memory
                
        return tokens

# Assuming `self.create_attention_mask` and `self.get_input_ids` are defined elsewhere in your class.
