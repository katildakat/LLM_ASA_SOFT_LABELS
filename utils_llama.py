import torch
import numpy as np
from utils_general import get_hist_bin

def get_user_tokens(prompt, tokenizer, llama_n, print_user_prompt=False):

    # llama3 template
    """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

    What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Bonjour! The capital of France is Paris!<|eot_id|><|start_header_id|>user<|end_header_id|>
    """
    # llama2 template
    """
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
    <s>[INST] {{ user_message_2 }} [/INST]
    """

    llama3_user_start = '<|start_header_id|>user<|end_header_id|>\n\n'
    llama3_user_end = '<|eot_id|>'
    llama2_user_start = '[INST] '
    llama2_user_end =  ' [/INST]'
    llama2_system_end_marker = '<</SYS>>\n\n'
    
    # Find the start and end of the user message in the sample
    if llama_n == 3:
        start_user_idx = prompt.rfind(llama3_user_start) + len(llama3_user_start)
        end_user_idx = prompt.find(llama3_user_end, start_user_idx) # Find the end of the user message after the start
    elif llama_n == 2:
        # Count the number of [INST] occurrences in the prompt
        inst_count = prompt.count('[INST]')
        
        if inst_count == 1:
            # Zero-shot case: only one [INST], extract the content between system end and user message end
            start_user_idx = prompt.find(llama2_system_end_marker) + len(llama2_system_end_marker)
            end_user_idx = prompt.find(llama2_user_end, start_user_idx)
        else:
            # In-context case: more than one [INST], extract the content of the last user message
            start_user_idx = prompt.rfind(llama2_user_start) + len(llama2_user_start)
            end_user_idx = prompt.find(llama2_user_end, start_user_idx)

    # Extract the full context (system message +{}+ user message)
    full_context = prompt[:end_user_idx]
    
    # Extract the previous conversation
    prev_convo = prompt[:start_user_idx]

    # Tokenize the system_message to find the start of the user message
    num_to_mask = len(tokenizer.encode(prev_convo, add_special_tokens=False))

    user_prompt = prompt[start_user_idx:end_user_idx]
    user_prompt_len = len(tokenizer.encode(user_prompt, add_special_tokens=False))
    if num_to_mask + user_prompt_len != len(tokenizer.encode(full_context, add_special_tokens=False)):
        print("Warning: Mismatch in tokenization lengths")
        print(num_to_mask + user_prompt_len, len(tokenizer.encode(full_context, add_special_tokens=False)))
    if print_user_prompt:
        # print only the user prompt
        user_prompt = prompt[start_user_idx:end_user_idx]
        print("|"+user_prompt+"|")
        print('------')
    
    return full_context, num_to_mask, user_prompt_len

def get_embedding(dataset, model, tokenizer, llama_n):
    sample_to_embed = {}
    for i, batch in enumerate(dataset):
        print(f"Batch {i+1}/{len(dataset)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_contexts = []
        masks = []
        user_messages_len = []
        for prompt in batch['prompt']:
            if i == 0:
                print_user_prompt = True
                print("original prompt:\n", prompt)
            else:
                print_user_prompt = False
            full_context, mask, user_message_len = get_user_tokens(prompt, tokenizer, llama_n, print_user_prompt=print_user_prompt)
            full_contexts.append(full_context)
            masks.append(mask)
            user_messages_len.append(user_message_len)

        # Tokenize the full contexts (system + user message)
        encodings = tokenizer(full_contexts, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']  # 1 for real tokens, 0 for padding tokens

        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # last hidden state batch x seq_len x hidden_size
            for i, sample in enumerate(hidden_states):
                # Cut out the user message (between the mask and padding tokens)
                num_to_mask = masks[i]
                # Remove context (tokens before the user message start)
                user_message_embeddings = sample[num_to_mask:attention_mask[i].sum()]  # Remove padding
                # Compute the mean of the embeddings for the user message
                mean_embedding = user_message_embeddings.mean(dim=0)
                
                # Store the mean embedding for this sample
                sample_to_embed[batch['sample'][i]] = mean_embedding.cpu().numpy()

    return sample_to_embed


def get_ppl(dataset, model, tokenizer, llama_n):

    sample_to_ppl = {}    
    for i, batch in enumerate(dataset):
        print(f"Batch {i+1}/{len(dataset)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        masks = []
        full_contexts = []
        for prompt in batch['prompt']:
            if i == 0:
                print_user_prompt = True
            else:
                print_user_prompt = False
            # get the full context and the mask before the user message
            full_context, mask, _ = get_user_tokens(prompt, tokenizer, llama_n, print_user_prompt=print_user_prompt)
            full_contexts.append(full_context)
            masks.append(mask)
            
        # Tokenize the full contexts (system + user message)
        encodings = tokenizer(full_contexts, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']  # 1 for real tokens, 0 for padding tokens
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            # put -100 to the target_ids for the tokens that are not in the user message
            for i, mask in enumerate(masks):
                target_ids[i, :mask] = -100

            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits # batch x seq_len x vocab

            # Shift logits and labels for computing loss
            shift_logits = logits[..., :-1, :].contiguous() # ignore the last token, because there is no label for it
            shift_labels = target_ids[..., 1:].contiguous() # shift to +1 from the start (so the next is predicted)
            
            # zero out attention mask for -100 tokens
            for i, mask in enumerate(masks):
                attention_mask[i, :mask] = 0
            shift_attention_mask = attention_mask[..., 1:].contiguous() # shift to +1 from the start (so the next is predicted)
            
            # Compute number of valid tokens for each sample
            num_valid_tokens = shift_attention_mask.sum(dim=1)
            if (num_valid_tokens == 0).any():
                print("Warning: Some samples have no valid tokens for loss calculation")

            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) # for all tokens in a batch
            loss = loss.view(shift_labels.size())  # Reshape back to batch_size x seq_len
            # Apply attention mask to remove the loss for pad tokens
            masked_loss = loss * shift_attention_mask

            # Compute perplexity per response
            per_response_loss = masked_loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
            max_loss = 100  # Chosen to avoid large loss values
            # Log for excessive loss values
            if (per_response_loss > max_loss).any():
                print(f"Warning: Large loss detected in batch {i}")
            per_response_loss = per_response_loss.clamp(max=max_loss)

            perplexity_per_response = torch.exp(per_response_loss)
            perplexity_per_response = perplexity_per_response.cpu().numpy()
            
            # Add perplexity to the dictionary
            for j, sample in enumerate(batch['sample']):
                sample_to_ppl[sample] = perplexity_per_response[j]
                # check if the perplexity is NaN
                if np.isnan(perplexity_per_response[j]):
                    print("Warning: NaN perplexity for sample", sample)
                    # check the loss
                    print("Loss:", per_response_loss[j])
    return sample_to_ppl

def process_batch(batch, model, tokenizer, level_to_token_id):
    """
    Process a batch of prompts and return the predicted levels and the probabilities of each level.
    """
    levels = list(level_to_token_id.keys())
    levels.sort()

    inputs = tokenizer(batch['prompt'], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits[:, -1, :] # batch_size x vocab_size (last token logits)
    last_logits_levels = logits[:, [level_to_token_id[level] for level in levels]] # batch_size x n_levels
    last_logits_probs = torch.nn.functional.softmax(last_logits_levels, dim=-1).cpu()
    y_pred = torch.argmax(last_logits_probs, dim=-1) 
    y_pred_ids = [level_to_token_id[levels[i]] for i in y_pred] 

    return y_pred_ids, last_logits_probs.numpy()

def grade_shots(loader, shot_type, model, tokenizer, level_to_token_id, data_records):
    print(f"GRADING {shot_type.upper()}-SHOT")
    
    # create a lookup dictionary if data_records is not empty
    record_lookup = {r['sample']: r for r in data_records} if data_records else {}

    for i, batch in enumerate(loader):
        print(f"Batch {i+1}/{len(loader)}")
        y_pred, y_probs = process_batch(batch, model, tokenizer, level_to_token_id)
        
        for i, answer_token_id in enumerate(y_pred):
            answer_token = tokenizer.decode(answer_token_id)
            record = None

            # if it's the first round of grading
            if record_lookup.get(batch['sample'][i]) is None:
                record = {
                    'sample': batch['sample'][i],
                    'y_true': int(batch['y_true'][i]), 
                    'task': batch['task'][i],
                    'y_true_mean': float(batch['y_true_mean'][i]),
                    'y_true_all_scores': batch['y_true_all_scores'][i]
                }
                data_records.append(record)  # append here for the first evaluation
            else: # find existing record
                record = record_lookup.get(batch['sample'][i])

            if record is not None: # if record is found or created
                # add argmax prediction
                record[f'{shot_type}_prediction'] = int(answer_token)
                
                # Compute soft score
                levels = sorted(level_to_token_id.keys())
                soft_prediction = sum(y_probs[i, j] * int(level) for j, level in enumerate(levels))
                
                record[f'{shot_type}_soft_prediction'] = soft_prediction
                record[f'{shot_type}_soft_prediction_bin'] = get_hist_bin([soft_prediction])[0]

                # add probabilities
                for j, level in enumerate(levels):
                    record[f'{shot_type}_prob_{level}'] = y_probs[i, j]
    
    return data_records


