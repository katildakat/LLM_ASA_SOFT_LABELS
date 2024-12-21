# IMPORTING LIBRARIES
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json
from transformers import AutoTokenizer 
import pandas as pd
import utils_prompts, utils_general
#---------------------------------------------------------
if __name__ == "__main__":
    # READ CONFIG
    config_path = os.getenv('CONFIG_PATH')
    config = utils_general.load_config(config_path)
    print(config)
    #---------------------------------------------------------
    # SET UP PARAMETERS
    fin_df_path = config['fin_df_path']
    model_name = config['model_name']
    model_name_short = config['model_name_short']
    assistant_message_start = config['assistant_message_start']
    transcript_column = config['transcript_column']
    random_seed = config['random_seed']
    good_examples = config['good_examples']
    #--------------------------------------------------------- 
    # DATA PREP
    fin_df = pd.read_csv(fin_df_path)
    fin_df = fin_df[fin_df['lt']=='lt'].copy() # get only the task subset needed for the study
    fin_lt = fin_df[fin_df['lt']=='lt'].copy().reset_index() 

    fin_lt['num_raters'] = [len(x.split()) for x in fin_lt['cefr_all_scores']]
    fin_lt['transcript_clean'] = [utils_general.clean_transcripts(t) for t in fin_lt['transcript']]
    
    if good_examples:
        fin_samples = utils_prompts.sample_good_examples(fin_lt, random_seed=random_seed)
        print("---------FINNISH SAMPLES---------")
        for task in fin_samples['task'].unique():
            print(task)
            print(fin_samples[fin_samples['task']==task]['sample'].tolist())    
    
        fin_lt['examples'] = [True if x in fin_samples['sample'].tolist() else False for x in fin_lt['sample']]
        # saving the df
        fin_lt.to_csv("data/fin_average_df_examples_marked.csv", index=False)
    else:
        # set all examples to False
        fin_lt['examples'] = False
        # set samples to None
        fin_samples = None
    #--------------------------------------------------------- 
    # LOAD THE MATERIALS FOR THE PROMPTS
    # load templates and scales
    prompt_templates = utils_prompts.load_prompt_templates("data/prompt_templates.yaml")
    # load task descriptions
    with open('data/task_prompts.json') as f:
        task_prompts_all = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #---------------------------------------------------------
    def generate_prompts(df_all, transcript_column, prompt_templates, task_prompts, language_id, assistant_message_start, tokenizer, model_name_short, samples_df=None):
        if language_id == 'swe':
            language = 'Finland Swedish'
        else:
            language = 'Finnish'
        if transcript_column == 'transcript_clean':
            transcript_type = 'human'
        else:
            transcript_type = 'asr'
        if not good_examples:
            # add all
            pass


        df = df_all[df_all['examples']==False] # get only samples that are not used as demonstrations

        prompts_zero = {}
        prompts_one_random = {}
        sample_to_one_shot_demonstration = {}
        prompts_few_random = {}
        sample_to_few_shot_demonstration= {}

        for task in df['task'].unique():
            task_df = df[df['task']==task].copy()
            task_description = task_prompts[language_id][task]
            if samples_df is not None:
                task_examples_df = samples_df[samples_df['task']==task].copy()
            else:
                task_examples_df = None

            # zero-shot
            task_zero_shot_prompts = utils_prompts.make_zero_shot_prompt(task_df,
                                                                         transcript_column, 
                                                                         task_description, 
                                                                         prompt_templates, 
                                                                         language, 
                                                                         assistant_message_start,
                                                                         tokenizer)
            
            # one-shot
            task_one_shot_prompts, task_one_shot_sample_to_demonstration = utils_prompts.make_one_shot_prompt(task_df,
                                                                                                              transcript_column,
                                                                                                              task_description,
                                                                                                              prompt_templates,
                                                                                                              language,
                                                                                                              assistant_message_start,
                                                                                                              tokenizer,
                                                                                                              task_examples_df,
                                                                                                              random_seed=random_seed)
                                                                       

            # few-shot
            task_few_shot_prompts, task_few_shot_sample_to_demonstration = utils_prompts.make_few_shot_prompt(task_df, 
                                                                                                              transcript_column,
                                                                                                              task_description,
                                                                                                              prompt_templates, 
                                                                                                              language, 
                                                                                                              assistant_message_start,
                                                                                                              tokenizer,
                                                                                                              task_examples_df,
                                                                                                              random_seed=random_seed)
                                                                
            # add prompts
            prompts_zero = prompts_zero | task_zero_shot_prompts
            prompts_one_random = prompts_one_random | task_one_shot_prompts
            prompts_few_random = prompts_few_random | task_few_shot_prompts
            sample_to_one_shot_demonstration = sample_to_one_shot_demonstration | task_one_shot_sample_to_demonstration
            sample_to_few_shot_demonstration = sample_to_few_shot_demonstration | task_few_shot_sample_to_demonstration

        # check if folder prompts exists
        if not os.path.exists('prompts'):
            os.makedirs('prompts')
        # save the prompts
        with open(f"prompts/sample_to_zero_shot_prompt_{model_name_short}_{language_id}_{transcript_type}.json", 'w') as f:
            json.dump(prompts_zero, f)
        with open(f"prompts/sample_to_one_shot_prompt_random_{model_name_short}_{language_id}_{transcript_type}.json", 'w') as f:
            json.dump(prompts_one_random, f)
        with open(f"prompts/sample_to_few_shot_prompt_random_{model_name_short}_{language_id}_{transcript_type}.json", 'w') as f:
            json.dump(prompts_few_random, f)
        with open(f"prompts/sample_to_one_shot_demonstration_{model_name_short}_{language_id}_{transcript_type}.json", 'w') as f:
            json.dump(sample_to_one_shot_demonstration, f)
        
        print(sample_to_few_shot_demonstration["651_old_lukio"])
        with open(f"prompts/sample_to_few_shot_demonstration_{model_name_short}_{language_id}_{transcript_type}.json", 'w') as f:
            json.dump(sample_to_few_shot_demonstration, f)
    #---------------------------------------------------------
    # GENERATE PROMPTS
    print("---------GENERATING FINNIH PROMPTS---------")
    generate_prompts(fin_lt, transcript_column, prompt_templates, task_prompts_all, 'fin', assistant_message_start, tokenizer, model_name_short, fin_samples)
