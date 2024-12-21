# IMPORTING LIBRARIES
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer 
import pandas as pd
import torch
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
    # check if ppl_results folder exists
    if not os.path.exists("ppl_results"):
        os.makedirs("ppl_results")
    lang_dataset = {}
    for lang in config['languages']:
        lang_df = lang_dfs[lang]
        # LOAD ZERO-SHOT PROMPTS
        zero_shot_prompts = utils_prompts.load_prompts(f"prompts/sample_to_zero_shot_prompt_llama{llama_n}_{lang}_asr.json")
        lang_prompts_dataloader_zero = utils_prompts.make_ppl_prompt_dataset(zero_shot_prompts, batch_size)
        # LOAD ONE-SHOT PROMPTS
        one_shot_prompts = utils_prompts.load_prompts(f"prompts/sample_to_one_shot_prompt_random_llama{llama_n}_{lang}_asr.json")
        lang_prompts_dataloader_one = utils_prompts.make_ppl_prompt_dataset(one_shot_prompts, batch_size)
        # LOAD FEW-SHOT PROMPTS
        few_shot_prompts = utils_prompts.load_prompts(f"prompts/sample_to_few_shot_prompt_random_llama{llama_n}_{lang}_asr.json")
        lang_prompts_dataloader_few = utils_prompts.make_ppl_prompt_dataset(few_shot_prompts, batch_size)
        
        print("---------EVALUATING ZERO-SHOT SAMPLES---------")
        sample_to_ppl_zero = utils_llama.get_ppl(lang_prompts_dataloader_zero, model, tokenizer, llama_n)
        print("---------EVALUATING ONE-SHOT SAMPLES---------")
        sample_to_ppl_one = utils_llama.get_ppl(lang_prompts_dataloader_one, model, tokenizer, llama_n)
        print("---------EVALUATING FEW-SHOT SAMPLES---------")
        sample_to_ppl_few = utils_llama.get_ppl(lang_prompts_dataloader_few, model, tokenizer, llama_n)

        # check if the dataframe exists
        if os.path.exists(f"ppl_results/{lang}_ppl.csv"):
            ppl_df = pd.read_csv(f"ppl_results/{lang}_ppl.csv")
            ppl_df[f"{model_name_short}_zero"] = [sample_to_ppl_zero[sample] if sample in sample_to_ppl_zero else None for sample in ppl_df['sample']]
            ppl_df[f"{model_name_short}_one"] = [sample_to_ppl_one[sample] if sample in sample_to_ppl_one else None for sample in ppl_df['sample']]
            ppl_df[f"{model_name_short}_few"] = [sample_to_ppl_few[sample] if sample in sample_to_ppl_few else None for sample in ppl_df['sample']]
        else:
            # create a dataframe
            ppl_df = pd.DataFrame({'sample': list(sample_to_ppl_zero.keys()), 
                                   f"{model_name_short}_zero": list(sample_to_ppl_zero.values()),
                                   f"{model_name_short}_one": list(sample_to_ppl_one.values()), 
                                   f"{model_name_short}_few": list(sample_to_ppl_few.values())})

        
        print("---------EVALUATING SAMPLES WITH DUMMY SYSTEM PROMPTS---------")
        for transcript_column in config['transcript_columns']:
            lang_df_no_examples = lang_df[lang_df['examples'] == False].copy()
            prompts = utils_prompts.make_ppl_prompts_chat(lang_df_no_examples, transcript_column, tokenizer)
            print(f"{lang} + {transcript_column} prompt example:")
            print("|"+prompts[list(prompts.keys())[0]]+"|")
            lang_prompts_dataloader = utils_prompts.make_ppl_prompt_dataset(prompts, batch_size)
            sample_to_ppl_simple = utils_llama.get_ppl(lang_prompts_dataloader, model, tokenizer, llama_n)
            ppl_df[f"{model_name_short}_{transcript_column}"] = [sample_to_ppl_simple[sample] if sample in sample_to_ppl_simple else None for sample in ppl_df['sample']]

        ppl_df.to_csv(f"ppl_results/{lang}_ppl.csv", index=False)