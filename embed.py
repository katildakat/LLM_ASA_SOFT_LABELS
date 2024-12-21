# IMPORTING LIBRARIES
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer 
import pandas as pd
import torch
import pickle
import utils_general, utils_prompts, utils_llama
#---------------------------------------------------------
if __name__ == "__main__":
    # READ CONFIG
    config_path = os.getenv('CONFIG_PATH')
    config = utils_general.load_config(config_path)
    print(config)
    #---------------------------------------------------------
    # SET UP PARAMETERS
    lang_dfs = {}
    for lang in config['languages']:
        dataset_path = config['dataset_paths'][lang]
        df = pd.read_csv(dataset_path)
        lang_dfs[lang] = df

    model_name = config['model']['name']
    model_name_short = config['model']['short_name']
    llama_n = config['model']['llama_n']
    transcript_column = config['transcript_column']
    batch_size = config['batch_size']
    #--------------------------------------------------------- 
    if torch.cuda.is_available():
        print("---------CUDA AVAILABLE---------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('---------START---------')
    #---------------------------------------------------------
    # LOADING THE MODEL AND THE TOKENIZER
    print("---------LOADING THE MODEL---------")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    tokenizer.add_eos_token = False
    tokenizer.add_bos_token = False
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto")
    model.eval()
    print("---------THE MODEL IS LOADED---------")
    print("---------MODEL'S DEVICE MAP---------")
    print(model.hf_device_map)
    print(model.device)
    #---------------------------------------------------------
    # GENERATE PROMPTS
    datasets = {}
    for lang in config['languages']:
        lang_df = lang_dfs[lang]
        demonstration_df = lang_df[lang_df['examples'] == True]
        samples_df = lang_df[lang_df['examples'] == False]

        sample_prompts = utils_prompts.make_ppl_prompts_chat(samples_df, transcript_column, tokenizer)
        demonstration_prompts = utils_prompts.make_ppl_prompts_chat(demonstration_df, "transcript_clean", tokenizer)

        # make the dataloaders
        sample_prompts_dataloader = utils_prompts.make_ppl_prompt_dataset(sample_prompts, batch_size)
        demonstration_prompts_dataloader = utils_prompts.make_ppl_prompt_dataset(demonstration_prompts, batch_size)

        datasets[lang] = [sample_prompts_dataloader, demonstration_prompts_dataloader]

    #---------------------------------------------------------
    # EMBEDDING
    print("---------EMBEDDING---------")
    for lang in config['languages']:
        print(f"---------{lang.upper()}---------")

        # make sample embeddings
        print("---------SAMPLE EMBEDDINGS---------\n")
        sample_to_embedding = utils_llama.get_embedding(datasets[lang][0], model, tokenizer, llama_n)
        # make demonstration embeddings
        print("---------DEMONSTRATION EMBEDDINGS---------\n")
        demo_to_embedding = utils_llama.get_embedding(datasets[lang][1], model, tokenizer, llama_n)

        # merge the dictionaries
        sample_to_embedding.update(demo_to_embedding)

        # save the embeddings
        name = f"{model_name_short}_{lang}"
        # check if the embeddings folder exists
        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")
        # pickle the embeddings
        with open(f"embeddings/{name}.pkl", 'wb') as f:
            pickle.dump(sample_to_embedding, f)   
         



        