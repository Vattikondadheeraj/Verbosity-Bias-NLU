from Inference import ARGS
import csv
from datasets import load_dataset
import tqdm


dataset = load_dataset("Dahoas/rm-static")

LLM_PATH = "/home/vdhee/scratch/Dheeraj/ARGS/LM_PATH"
RM_PATH = "/home/vdhee/scratch/Dheeraj/DeepSpeed1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/training_scripts/opt/single_gpu/output"

searcher = ARGS(llm_path=LLM_PATH, rm_path=RM_PATH, llm_dev="cuda:0", rm_dev="cuda:0")

# args-greedy decoding with weight=1.0
with open('output-1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Prompt', 'Generated Response'])

    # Iterate over your dataset
    for p in tqdm.tqdm(dataset["train"]):
        prompt="[INST] <<SYS>>You are a nice chatbot having a conversation with a human. I need a single consolidated response from the assistant<<SYS>>" + p["prompt"] + "[/INST]"
     

        output_tokens = searcher.generate(prompt)
        tokens_text = searcher.tokens_to_text(output_tokens)[0]
        
        # Write the prompt and the generated response to the CSV file
        writer.writerow([p["prompt"], tokens_text])
        print(tokens_text, "***************************", "\n")