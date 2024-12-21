import pandas as pd
import yaml
import json
import random
from datasets import Dataset
from torch.utils.data import DataLoader

def sample_good_examples(df, random_seed=None):
    sampled_examples = pd.DataFrame()
    
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        cefr_bins = task_df['cefr_bin'].unique()
        
        # First, try to sample from cefr_range == 0
        good_task_df = task_df[
            (task_df['num_raters'] > 1) & (task_df['cefr_range'] == 0)
        ]
        
        # Keep track of which cefr_bin labels have been sampled
        sampled_bins = []
        
        # Sample one example per cefr_bin where cefr_range == 0
        for bin_label in good_task_df['cefr_bin'].unique():
            bin_df = good_task_df[good_task_df['cefr_bin'] == bin_label]
            sample = bin_df.sample(n=1, random_state=random_seed)
            sampled_examples = pd.concat([sampled_examples, sample])
            sampled_bins.append(bin_label)
        
        # Identify any cefr_bin labels that were not sampled
        missing_bins = set(cefr_bins) - set(sampled_bins)
        
        if missing_bins:
            # Sample from cefr_range == 1 for missing bins
            range_one_df = task_df[
                (task_df['num_raters'] > 1) & (task_df['cefr_range'] == 1)
            ]
            for bin_label in missing_bins:
                bin_df = range_one_df[range_one_df['cefr_bin'] == bin_label]
                if not bin_df.empty:
                    sample = bin_df.sample(n=1, random_state=random_seed)
                    sampled_examples = pd.concat([sampled_examples, sample])
                    sampled_bins.append(bin_label)
                else:
                    print(f"No available samples for cefr_bin '{bin_label}' in task '{task}'")
    
    sampled_examples.reset_index(drop=True, inplace=True)
    return sampled_examples


def load_prompt_templates(prompts_path: str):
    with open(prompts_path, 'r') as file:
        return yaml.safe_load(file)['prompts']
    
def remove_eos(prompt, assistant_message_start):
    """
    Remove the end of string token from the prompt. So that the assistant can continue the conversation.
    """
    index = prompt.rfind(assistant_message_start)

    # slice the string up to the last occurrence of assistant start
    if index != -1:
        result_string = prompt[:index + len(assistant_message_start)] + " "
    else:
        result_string = prompt

    return result_string
    
def tokenize_chat_into_string(tokenizer, chat_dict, assistant_message_start):
    chat_string = tokenizer.apply_chat_template(chat_dict, tokenize=False)
    chat_string = remove_eos(chat_string, assistant_message_start)
    return chat_string
    
def make_zero_shot_prompt(task_df, transcript_column, task_description, prompt_templates, language, assistant_message_start, tokenizer):
    task_zero_shot_prompts = {}

    # fill in the system message template
    system_message = prompt_templates['grading_system_message_numerals'].format(language=language,
                                                                                proficiency_scale=prompt_templates['proficiency_scale_numerals'],
                                                                                task_description=task_description)
    zero_shot_chat_general = [
        {"role": "system", "content": system_message} 
    ]

    # finish the prompt for each sample
    for sample in task_df['sample']:
        sample_zero_shot_chat = zero_shot_chat_general.copy()
        
        transcript = task_df[task_df['sample']==sample][transcript_column].values[0]

        # add a transcript that needs grading to the chat
        sample_zero_shot_chat.append({"role": "user", "content": transcript})
        
        # add the start of the assistant message
        sample_zero_shot_chat.append({"role": "assistant", "content": assistant_message_start})
        
        # turn chat template into a string
        sample_zero_shot_string = tokenize_chat_into_string(tokenizer, sample_zero_shot_chat, assistant_message_start)
        
        task_zero_shot_prompts[sample] = sample_zero_shot_string

    return task_zero_shot_prompts

def make_one_shot_prompt(task_df, transcript_column, task_description, prompt_templates, language, assistant_message_start, tokenizer, task_examples_df=None, random_seed=42):
    """
    Select a random example from the same task and create a one-shot prompt for each sample.
    """
    sample_to_demonstrations = {}
    task_one_shot_prompts = {}

    # Fill in system message template
    system_message = prompt_templates['grading_system_message_numerals'].format(
        language=language,
        proficiency_scale=prompt_templates['proficiency_scale_numerals'],
        task_description=task_description
    )

    one_shot_chat_general = [
        {"role": "system", "content": system_message}
    ]

    # Finish the prompt for each sample
    for i, sample in enumerate(task_df['sample']):
        sample_one_shot_chat = one_shot_chat_general.copy()
        transcript = task_df[task_df['sample'] == sample][transcript_column].values[0]

        # Create a reproducible random seed based on the sample
        reproducible_seed = random_seed + i*10

        # Select a random example
        if task_examples_df is not None:
            example_row = task_examples_df.sample(1, random_state=reproducible_seed)
        else:
            # Ensure the selected sample is different from the current sample
            other_samples = task_df[task_df['sample'] != sample]
            example_row = other_samples.sample(1, random_state=reproducible_seed)

        example_id = example_row['sample'].values[0]
        example_transcript = example_row['transcript_clean'].values[0]
        example_bin = example_row['cefr_bin'].values[0]
        sample_to_demonstrations[sample] = example_id

        # Add example to the template
        sample_one_shot_chat.append({"role": "user", "content": example_transcript})
        sample_one_shot_chat.append({"role": "assistant", "content": assistant_message_start + " " + str(example_bin)})

        # Add transcript to grade
        sample_one_shot_chat.append({"role": "user", "content": transcript})
        # Add the start of the assistant message
        sample_one_shot_chat.append({"role": "assistant", "content": assistant_message_start})

        # Turn chat template into a string
        sample_one_shot_string = tokenize_chat_into_string(tokenizer, sample_one_shot_chat, assistant_message_start)

        task_one_shot_prompts[sample] = sample_one_shot_string

    return task_one_shot_prompts, sample_to_demonstrations


def make_few_shot_prompt(task_df, transcript_column, task_description, prompt_templates, language, assistant_message_start, tokenizer, task_examples_df=None, random_seed=42):
    """
    Create few-shot prompts for each sample by selecting examples and building a prompt string.
    """

    sample_to_demonstrations = {}
    task_few_shot_prompts = {}

    # Fill in system message template
    system_message = prompt_templates['grading_system_message_numerals'].format(
        language=language,
        proficiency_scale=prompt_templates['proficiency_scale_numerals'],
        task_description=task_description
    )

    few_shot_chat_general = [
        {"role": "system", "content": system_message}
    ]

    # If task_examples_df is provided, shuffle bins consistently
    if task_examples_df is not None:
        # Get unique bins, sort, and shuffle with the given seed
        unique_bins = sorted(task_examples_df['cefr_bin'].unique())
        random.seed(random_seed)
        random.shuffle(unique_bins)
        shuffled_bins = unique_bins

    # Generate prompts for each sample
    for sample in task_df['sample']:
        sample_few_shot_chat = few_shot_chat_general.copy()

        if task_examples_df is None:
            # if no task_examples_df, sample one example from each cefr_bin
            grouped_bins = task_df[task_df['sample'] != sample].groupby('cefr_bin')
            examples_list = []
            for _, group in grouped_bins:
                example = group.sample(1, random_state=random_seed)
                examples_list.append(example)
            examples_df = pd.concat(examples_list)
            # shuffle the examples
            examples_df = examples_df.sample(frac=1, random_state=random_seed)
        else:
            # filter and order the task_examples_df by the shuffled bins
            examples_list = []
            for score_bin in shuffled_bins:
                examples_list.append(task_examples_df[task_examples_df['cefr_bin'] == score_bin])
            examples_df = pd.concat(examples_list)

        # Track examples used for the current sample
        demonstration_ids = []

        for _, example_row in examples_df.iterrows():
            example_id = example_row['sample']
            example_transcript = example_row['transcript_clean']
            example_bin = example_row['cefr_bin']
            demonstration_ids.append(example_id)
            sample_few_shot_chat.append({"role": "user", "content": example_transcript})
            sample_few_shot_chat.append({"role": "assistant", "content": assistant_message_start + " " + str(example_bin)})

        # Save demonstrations for the sample
        sample_to_demonstrations[sample] = demonstration_ids

        # Get the transcript for the current sample
        transcript = task_df[task_df['sample'] == sample][transcript_column].values[0]

        # Add the sample transcript to grade
        sample_few_shot_chat.append({"role": "user", "content": transcript})
        # Add the start of the assistant message
        sample_few_shot_chat.append({"role": "assistant", "content": assistant_message_start})

        # Turn chat template into a string
        sample_few_shot_string = tokenize_chat_into_string(tokenizer, sample_few_shot_chat, assistant_message_start)

        # Save the prompt
        task_few_shot_prompts[sample] = sample_few_shot_string

    return task_few_shot_prompts, sample_to_demonstrations

def load_prompts(prompts_path):
    with open(prompts_path) as f:
        prompts = json.load(f)
    return prompts

def make_llama_prompt_dataset(path_to_prompts, batch_size, df):
    sample_to_prompt = load_prompts(path_to_prompts)

    samples = list(sample_to_prompt.keys())
    # make sure that the samples are in the dataframe[]
    sample_df = df[df['sample'].isin(samples)]
    # make sure that the order as in samples
    sample_df = sample_df.set_index('sample').loc[samples].reset_index()

    y_true = sample_df['cefr_bin'].tolist()
    tasks = sample_df['task'].tolist()
    y_true_means = sample_df['cefr_mean'].tolist()
    y_true_all_scores = sample_df['cefr_all_scores'].tolist()
    prompts = [sample_to_prompt[sample] for sample in samples]
    dataset = Dataset.from_dict({'sample':samples,
                                 'y_true':y_true,
                                 'task':tasks,
                                 'y_true_mean':y_true_means,
                                 'y_true_all_scores':y_true_all_scores,                
                                 'prompt':prompts})
    
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

def make_ppl_prompt_dataset(sample_to_prompt, batch_size):
    samples = list(sample_to_prompt.keys())
    dataset = Dataset.from_dict({'sample':samples,
                                 'prompt':list(sample_to_prompt.values())})
    
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

def make_ppl_prompts_chat(df, transcript_column, tokenizer):
    """"
    Create prompts as a dummy chat with the dummy system message and transcripts as user messags.
    """

    dummy_system_message = "Echo the user’s input exactly as provided."
    dummy_chat_general = [ 
        {"role": "system", "content": dummy_system_message} 
    ]

    sample_to_prompt = {}
    for sample in df['sample']:
        sample_chat = dummy_chat_general.copy()
        transcript = df[df['sample']==sample][transcript_column].values[0]
        sample_chat.append({"role": "user", "content": transcript})
        sample_chat_string = tokenizer.apply_chat_template(sample_chat, tokenize=False)
        sample_to_prompt[sample] = sample_chat_string
    
    return sample_to_prompt
