# IMPORTING LIBRARIES
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer 
import pandas as pd
import torch
from utils_general import load_config
from utils_prompts import make_llama_prompt_dataset
from utils_llama import grade_shots
#---------------------------------------------------------
if __name__ == "__main__":
    # READ CONFIG
    config_path = os.getenv('CONFIG_PATH')
    config = load_config(config_path)
    print(config)
    #---------------------------------------------------------
    # SET UP PARAMETERS
    fin_df_path = config['fin_df_path']
    llama_n = config['llama_n'] # for the prompt file
    model_name = config['model_name']
    batch_size = config['batch_size']
    #---------------------------------------------------------
    if torch.cuda.is_available():
        print("---------CUDA AVAILABLE---------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('---------START---------')
    #--------------------------------------------------------- 
    # DATA PREP
    fin_df = pd.read_csv(fin_df_path)
    fin_df = fin_df[fin_df['lt']=='lt'].copy()
    #---------------------------------------------------------
    # LOADING THE MODEL AND THE TOKENIZER
    print("---------LOADING THE MODEL---------")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
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
    # GET LEVEL TOKEN IDS IN THE MODEL'S VOCABULARY
    levels = [str(x) for x in range(1,8)]
    level_to_token_id = {}
    for level in levels:
        level_to_token_id[level] = tokenizer.convert_tokens_to_ids(level)
    #---------------------------------------------------------
    # GRADING
    print("---------GRADING---------")

    langs = ['fin']
    lang_dfs = [fin_df]
    shot_types = ['zero_shot_prompt', 
                  'one_shot_prompt_random',
                  'few_shot_prompt_random']

    for i, lang in enumerate(langs):
        data_records = []
        for shot_type in shot_types:
            shot_path = f"prompts/sample_to_{shot_type}_llama{llama_n}_{lang}_asr.json"
            print(shot_path)
            # prepare the dataloader
            shot_dataloader = make_llama_prompt_dataset(shot_path, batch_size, lang_dfs[i])
            
            data_records = grade_shots(shot_dataloader, shot_type, model, tokenizer, level_to_token_id, data_records)
            df_predictions = pd.DataFrame(data_records)
            # save results by the config name
            name = config_path.split("/")[-1].split(".")[0]
            # check if the folder exists
            if not os.path.exists("grading_results"):
                os.makedirs("grading_results")
            df_predictions.to_csv(f"grading_results/{name}_{lang}.csv", index=False)