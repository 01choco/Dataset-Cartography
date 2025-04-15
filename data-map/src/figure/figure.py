import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import ListedColormap


def get_figure(cfg, dataset):

    scores = [ item['scores'] for item in dataset]
    scores_avg = [np.mean(item) for item in scores]
    score_variance = [item['score_variance'] for item in dataset]
    correlation = [item['correlation'] for item in dataset]
    instruction = [item['instruction'] for item in dataset]
    gpt_inference = [item['gpt_inference'] for item in dataset]

    response = []
    for item in dataset:
        response_str = ""
        for i in range(len(item['completions'])):
            response_str += item['completions'][i]['response']
            if i != 3:
                response_str += ", "
        response.append(response_str)


    combined_df = pd.DataFrame({
        'score_avg': scores_avg,
        'scores': scores,
        'score_variance': score_variance,
        'correlation': correlation,
        'instruction': instruction,
        'response': response,
        'gpt_inference': gpt_inference
    })

    # Filter out negative cosine similarity values
    filtered_df = combined_df[combined_df['correlation'] >= cfg.min_cosine_similarity]
    filtered_df = filtered_df[filtered_df['correlation'] < cfg.max_cosine_similarity]
    min_cosine_similarity = filtered_df['correlation'].min()

    # Figure 
    # Define bins, colors, and shapes based on 'correlation'
    cosine_bins = [0.0, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95]

    # Generate color gradient: blue → dark blue → black → dark red → red
    cmap = ListedColormap(['blue', 'darkblue', 'black', 'darkred', 'red'])
    gradient_colors = [cmap(i / 4) for i in range(5)]  # 5 gradient colors
    gradient_colors.append('pink')  # Add red for the last bin
    colors = {i + 1: gradient_colors[i] for i in range(6)}

    # Define shapes for each bin
    shapes = {1: 'o', 2: 'x', 3: 's', 4: '^', 5: 'v', 6: 'd'}

    # Classify 'correlation' into bins
    filtered_df['cosine_bin'] = pd.cut(
        filtered_df['correlation'], 
        bins=cosine_bins,
        labels=[1, 2, 3, 4, 5, 6],  # Correspond to colors and shapes
        include_lowest=True
    )

    # Initialize plot
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot with varying colors and shapes
    for c in filtered_df['cosine_bin'].unique():
        if pd.isna(c):  # Skip NaN bins
            continue
        subset_df = filtered_df[filtered_df['cosine_bin'] == c]
        ax.scatter(
            subset_df['score_variance'], 
            subset_df['score_avg'], 
            c=[colors[int(c)]],  # Match the color to the cosine bin
            marker=shapes[int(c)],  # Match the shape to the cosine bin
            edgecolors='white',  # Add black border to points
            label=f'{cosine_bins[int(c)-1]:.1f} ~ {cosine_bins[int(c)]:.1f}', 
            alpha=1.0
        )

    # Sort the legend manually by the cosine_bin order
    handles, labels = ax.get_legend_handles_labels()  # Get current legend handles and labels

    # Sort handles and labels by the numeric range in labels
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: float(x[1].strip().split("~")[0]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    # LaTeX 스타일 텍스트 활성화
    plt.rcParams['font.family'] = 'cmr10'

    # Axis labels and legend
    plt.xlabel('Variance', fontsize=15)
    plt.ylabel('Average Score', fontsize=15)
    plt.ylim(0, 1)  # Set the y-axis limits based on cfg

    plt.title(cfg.title)
    ax.legend(
        sorted_handles,
        sorted_labels,
        title="correlation", 
        loc="center left",       # 범례를 왼쪽 중앙에 앵커링
        bbox_to_anchor=(1.0, 0.5),  # 그래프의 오른쪽에 위치하도록 설정
        frameon=True,            # 박스 활성화
        edgecolor='black',       # 테두리 색상
        framealpha=1.0,           # 테두리 투명도
        shadow=True
    )

    plt.grid(True)

    # Add colorbar for full cosine similarity range
    scatter = ax.scatter(
        filtered_df['score_variance'], 
        filtered_df['score_avg'], 
        c=filtered_df['correlation'], 
        cmap='viridis', 
        alpha=0
    )
    plt.savefig(cfg.fig_output_path)
    plt.show()
