import wandb
from argsearch import ARGS
from datasets import load_dataset
import tqdm

# Initialize wandb
wandb.init(project="Length_Bias-T_10-W_8-L_5", entity="vattikondadheeraj")

# Define the structure of the table
table = wandb.Table(columns=["Prompt", "Generated Response", "previous_tokens", "best_tokens", "prescreen_logits","new_scores", "rewards"])

dataset = load_dataset("Dahoas/rm-static")

LLM_PATH = "/home/vdhee/scratch/Dheeraj/ARGS/LM_PATH"
RM_PATH = "/home/vdhee/scratch/Dheeraj/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/training_scripts/opt/single_gpu/output"

searcher = ARGS(llm_path=LLM_PATH, rm_path=RM_PATH, llm_dev="cuda:0", rm_dev="cuda:0")

# Assuming you want to limit the number of processed prompts to keep the example concise
max_examples = None  # Set this to None to process all

for i, p in enumerate(tqdm.tqdm(dataset["test"]), start=1):
    prompt = "[INST] <<SYS>>You are a nice chatbot having a conversation with a human. I need a single very long consolidated response from the assistant<<SYS>>" + p["prompt"] + "[/INST]"
    output_tokens, previous_tokens, best_tokens, prescreen_logits, new_scores, rewards = searcher.generate(prompt, topk=10, weight=0, method="greedy", lookahead=0)
    tokens_text = searcher.tokens_to_text(output_tokens)[0]
    
    # Add the prompt and the generated response to the wandb table
    table.add_data(prompt, tokens_text, previous_tokens, best_tokens, prescreen_logits, new_scores, rewards)
    if i % 2 == 0:
        wandb.log({"Generated Responses": table})
        table = wandb.Table(columns=["Prompt", "Generated Response", "previous_tokens", "best_tokens", "prescreen_logits","new_scores", "rewards"])
wandb.finish()



