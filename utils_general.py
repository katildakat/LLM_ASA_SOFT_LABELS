import yaml
import re
import numpy as np
from scipy.stats import entropy
from sklearn import metrics
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def clean_transcripts(transcript, replace=False):
    transcript = transcript.lower()
    
    # a fix for one transcript:
    transcript = transcript.replace("<name*", "<name>")
    
    clean_t = re.sub('<.*?>', '', transcript)  
    clean_t = "".join(c for c in clean_t if c.isalpha() or c.isspace())
    clean_t = " ".join(clean_t.split())
    if replace:
        clean_t = clean_t.replace("w", "v").replace("é", "e").replace("ü", "u").replace("q","k").replace("z","s")
    return clean_t

def get_hist_bin(values, range_min=1, range_max=7):
    n_bins = range_max-range_min+1
    
    bin_labels=[]
    
    # get bin edges
    _, bin_edges = np.histogram([x for x in range(range_min,range_max+1)], bins=n_bins)
    for v in values:
        if v==9999:
            bin_labels.append(9999)
        else:
            i = 1
            while v > bin_edges[i]:
                i+=1
            b=i
            bin_labels.append(b)
    
    return bin_labels

def calculate_entropy(df, shot_prefix):
    df = df.copy()
    prob_columns = [f'{shot_prefix}_prob_{i}' for i in range(1, 8)]
    # Calculate entropy row-wise for the probability columns
    df[f'{shot_prefix}_entropy'] = df[prob_columns].apply(lambda x: entropy(x + 1e-9), axis=1)
    return df

def cosine_dist(embedding1, embedding2):
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return 1 - cosine_similarity

def add_sample_characteristics(results_df, df, sample_to_embedding, ppl_df, model_name_short, one_shot_sample_to_demonstration, few_shot_sample_to_demonstration):
    # get ppl for each sample
    results_df['zero_ppl'] = [ppl_df[ppl_df['sample']==sample][f'{model_name_short}_zero'].values[0] for sample in results_df['sample']]
    results_df['one_ppl'] = [ppl_df[ppl_df['sample']==sample][f'{model_name_short}_one'].values[0] for sample in results_df['sample']]
    results_df['few_ppl'] = [ppl_df[ppl_df['sample']==sample][f'{model_name_short}_few'].values[0] for sample in results_df['sample']]

    # add human variance
    human_var = []
    for sample in results_df['sample']:
        human_scores = [int(x) for x in df[df['sample']==sample]['cefr_all_scores'].values[0].split()]
        human_var.append(np.var(human_scores))
    results_df['human_var'] = human_var

    # add entropy per shot
    for shot in ['zero_shot_prompt', 'one_shot_prompt_random', 'few_shot_prompt_random']:
            results_df = calculate_entropy(results_df, shot)

    # add distances to demonstrations
    one_shot_dist = []
    few_shot_dist_max = []
    few_shot_dist_min = []
    few_shot_dist_avg = []
    for sample in results_df['sample']:
        one_shot_demonstration_sample = one_shot_sample_to_demonstration[sample]
        few_shot_demonstration_samples = few_shot_sample_to_demonstration[sample]
        
        sample_embedding = sample_to_embedding[sample]
        one_shot_demonstration_embedding = sample_to_embedding[one_shot_demonstration_sample]
        few_shot_demonstration_embeddings = [sample_to_embedding[demonstration_sample] for demonstration_sample in few_shot_demonstration_samples]
        # compute cosine distance
        one_shot_distance = cosine_dist(sample_embedding, one_shot_demonstration_embedding)
        one_shot_dist.append(one_shot_distance)
        few_shot_distances = [cosine_dist(sample_embedding, demo_emb) for demo_emb in few_shot_demonstration_embeddings]
        few_shot_dist_max.append(max(few_shot_distances))
        few_shot_dist_min.append(min(few_shot_distances))
        few_shot_dist_avg.append(np.mean(few_shot_distances))
    results_df['one_shot_dist'] = one_shot_dist
    results_df['max_dist'] = few_shot_dist_max
    results_df['min_dist'] = few_shot_dist_min
    results_df['av_dist'] = few_shot_dist_avg

    return results_df

def macro_mae(y_true, y_pred):
    """
    Calculate average MAE per class
    """
    levels = list(set(y_true)) # get all unique levels
    levels.sort()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = 0
    for level in levels:
        level_mask = y_true == level
        level_y_true = y_true[level_mask]
        level_y_pred = y_pred[level_mask]
        level_mae = metrics.mean_absolute_error(level_y_true, level_y_pred)
        mae += level_mae
    return mae/len(levels)

def count_s_win_rate(hard_labels, soft_labels, y_true):
    """"
    Count how many times soft labels are closer to y_true than hard labels
    """
    s_win = 0
    num_diff = 0 # how many times soft and hard labels are different
    for i in range(len(hard_labels)):
        if soft_labels[i] != hard_labels[i]:
            num_diff += 1
            if abs(soft_labels[i] - y_true[i]) < abs(hard_labels[i] - y_true[i]):
                s_win += 1
    return s_win, num_diff

def print_results(results):
    y_true = results["y_true"]
    # for each shot type
    # print s win rate
    # accuracy, macro f1, macro mae, kappa, correlation
    for shot_column in ['zero_shot_prompt', 'one_shot_prompt_random', 'few_shot_prompt_random']:
        y_pred = results[f"{shot_column}_prediction"]
        y_pred_soft_bin = results[f"{shot_column}_soft_prediction_bin"]
        s_win, out_of = count_s_win_rate(y_pred, y_pred_soft_bin, y_true)
        
        f1_hard = metrics.f1_score(y_true, y_pred, average='macro')
        acc_hard = metrics.accuracy_score(y_true, y_pred)
        mae_hard = macro_mae(y_true, y_pred)
        kappa_hard = metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        # print rounded values
        print(f"Shot type: {shot_column}")
        print(f"S win rate: {s_win}/{out_of}")
        print("Average entropy: ", np.mean(results[f"{shot_column}_entropy"]))
        print(f"Acc: {acc_hard:.2f}, F1: {f1_hard:.2f}, QWK: {kappa_hard:.2f}, MAE: {mae_hard:.2f}")

        f1_soft = metrics.f1_score(y_true, y_pred_soft_bin, average='macro')
        acc_soft = metrics.accuracy_score(y_true, y_pred_soft_bin)
        mae_soft = macro_mae(y_true, y_pred_soft_bin)
        kappa_soft = metrics.cohen_kappa_score(y_true, y_pred_soft_bin, weights='quadratic')
        
        # print rounded values
        print(f"Acc: {acc_soft:.2f}, F1: {f1_soft:.2f}, QWK: {kappa_soft:.2f}, MAE: {mae_soft:.2f}")

def plot_mean_distribution(data, prob_prefix='zero_shot_prompt_prob_', y_true_col='y_true', save_as_svg=False, cmap='viridis'):
    # Filter columns for probability distributions
    prob_cols = [col for col in data.columns if col.startswith(prob_prefix)]
    
    # Calculate the mean for each y_true value
    mean_distributions = data.groupby(y_true_col)[prob_cols].mean()
    
    # Normalize color range from 1 to 7
    norm = plt.Normalize(vmin=1, vmax=7)

    # Plotting each y_true's mean distribution with gradually changing colors
    fig, ax = plt.subplots(figsize=(6, 3))
    for y_val in mean_distributions.index:
        color = cmap(norm(y_val))
        ax.plot(mean_distributions.columns.str.replace(prob_prefix, ''), 
                mean_distributions.loc[y_val], 
                label=f'y_true = {y_val}', 
                color=color, 
                linewidth=2.5,  # Make the line thicker
                marker='o',     # Add points to the plot
                markersize=6)   # Size of the markers
    
    # Adding color bar to indicate y_true gradient from 1 to 7
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('y_true (1 to 7)')

    # Plot aesthetics
    ax.set_title(f'Mean Distribution of {prob_prefix} for Each y_true Label')
    ax.set_xlabel('Probability Index')
    ax.set_ylabel('Mean Probability')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid()
    
    # Save the plot as an SVG file if requested
    if save_as_svg:
        svg_filename=prob_prefix+'plot.svg'
        plt.savefig(svg_filename, format='svg')
    
    # Show the plot
    plt.show()

def compare_distributions(results_df, split_column_1, split_column_2, columns_to_test):
    """
    Perform Mann-Whitney U tests to compare distributions for specified columns 
    based on the difference between two split columns.
    
    Parameters:
        results_df (pd.DataFrame): The input DataFrame.
        split_column_1 (str): The first column used for comparison.
        split_column_2 (str): The second column used for comparison.
        columns_to_test (list): List of columns to test for distribution differences.
    
    Returns:
        dict: A dictionary containing the test statistic and p-value for each tested column.
    """
    # Split the data based on whether the two split columns match
    match_group = results_df[results_df[split_column_1] == results_df[split_column_2]]
    non_match_group = results_df[results_df[split_column_1] != results_df[split_column_2]]

    print(len(match_group), len(non_match_group))
    
    # Perform Mann-Whitney U tests for the specified columns
    results = {}
    for variable in columns_to_test:
        # Extract the variable values for each group
        match_group_var = match_group[variable]
        non_match_group_var = non_match_group[variable]
        print(variable)
        print(match_group[variable].mean(), non_match_group[variable].mean())
        
        # Perform the Mann-Whitney U test
        statistic, p_value = mannwhitneyu(match_group_var, non_match_group_var, alternative='two-sided')
        
        # Determine if the distributions are significantly different
        significance = "yes" if p_value < 0.05 else "no"
        
        # Determine if the median of split_column_1 is larger than split_column_2
        if significance == "yes":
            median_comparison = "split_column_1 > split_column_2" if match_group_var.median() > non_match_group_var.median() else "split_column_1 <= split_column_2"
        else:
            median_comparison = "N/A"
        
        # Store the results
        results[variable] = {'significance': significance, 'median_comparison': median_comparison}
    
    return results