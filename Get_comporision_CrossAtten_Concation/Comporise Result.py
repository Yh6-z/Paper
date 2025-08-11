
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
data_dir_Conc = os.path.join(parent_dir, "Cross Attention", "Gate Fusion", "Fusion_results")
data_dir_Cross = os.path.join(parent_dir, "Cross Attention", "Code_BertCrossAttentionRegressor")
data_dir_CrossMLP = os.path.join(parent_dir, "Cross Attention", "Code_BertCrossAttentionMLP")

# 读取数据集
try:
    regression_csv = os.path.join(data_dir_Conc, "regression_results_20250726_1852.csv")
    if not os.path.exists(regression_csv):
        raise FileNotFoundError(f"CSV file not found: {regression_csv}")
    regression_df = pd.read_csv(regression_csv)
    
    regression_df = regression_df.rename(columns={'Method': 'Property'})
    # regression_df['Property'] = regression_df['Property'].replace({
    #     'Numeric_Structure': 'Numeric',
    #     'Suspense_Question': 'Suspense'
    # })
    regression_df = regression_df[regression_df['Property'].isin(['Specificity', 'Numeracy', 'Pronouns', 'Suspense', 'Emotion', 'Readability'])]

    CA_csv = os.path.join(data_dir_Cross, "Attention_Linear_accuracy_results.csv")
    if not os.path.exists(CA_csv):
        raise FileNotFoundError(f"CSV file not found: {CA_csv}")
    CA_df = pd.read_csv(CA_csv)

    CA_M_csv = os.path.join(data_dir_CrossMLP, "Attention_MLP_accuracy_results.csv")
    CA_M_df = pd.read_csv(CA_M_csv)
    # CA_M_df['Property'] = CA_M_df['Property'].replace({
    #     'Numeric_Structure': 'Numeric',
    #     'Suspense_Question': 'Suspense'
    # })
    CA_M_df = CA_M_df[CA_M_df['Property'].isin(['Specificity', 'Numeracy', 'Pronouns', 'Suspense', 'Emotion', 'Readability'])]
    print("All datasets loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    raise

# 合并数据
merged_df = regression_df[['Property', 'Accuracy']].merge(
    CA_df[['Property', 'Accuracy']],
    on='Property',
    suffixes=('_Concatation', '_CrossAttention')
).merge(
    CA_M_df[['Property', 'Accuracy']],
    on='Property'
)
merged_df = merged_df.rename(columns={'Accuracy': 'Accuracy_CrossAttentionMLP'})

properties = ['Specificity', 'Numeracy', 'Pronouns', 'Suspense', 'Emotion', 'Readability']
merged_df = merged_df.set_index('Property').reindex(properties).reset_index()

# 创建第一个图形：Concatenation vs CrossAttention Linear with Baseline as bars
plt.figure(figsize=(14, 6), dpi=100)

bar_width = 0.3
x = np.arange(len(merged_df['Property']))
offset = [-bar_width, 0, bar_width]

datasets_1 = [
    {'name': 'LLM Only', 'column':None, 'color': '#7f7f7f'},  # 灰色 for baseline
    {'name': 'LLM+Cue(Concatation)', 'column': 'Accuracy_Concatation', 'color': '#1f77b4'},  # 蓝色
    {'name': 'LLM+Cue(CrossAttention)', 'column': 'Accuracy_CrossAttention', 'color': '#ff3e0e'}  # 橙色
]

all_accuracies_1 = []
for i, dataset in enumerate(datasets_1):
    color = dataset['color']
    label = dataset['name']
    
    if dataset['name'] == 'LLM Only':
        accuracies = [0.419] * len(merged_df)
        print(f"{label}:\n", accuracies)
    else:
        accuracies = merged_df[dataset['column']].tolist()
        print(f"{label}:\n", merged_df[dataset['column']])
    
    all_accuracies_1.extend(accuracies)
    
    x_positions = x + offset[i]
    bars = plt.bar(x_positions, accuracies, width=bar_width, color=color, label=label)
    
    for j, acc in enumerate(accuracies):
        plt.text(x_positions[j], acc + 0.001, f'{acc:.3f}', ha='center', va='bottom', fontsize=10, color='black')

if all_accuracies_1:
    y_min = min(all_accuracies_1)
    y_max = max(all_accuracies_1)
    y_range = y_max - y_min
    y_margin = y_range * 0.2 if y_range > 0 else 0.05
    plt.ylim(max(0.300, y_min - y_margin), y_max + y_margin + 0.021)
    print(f"Y-axis range for first plot: [{max(0.305, y_min - y_margin)}, {y_max + y_margin}]")
else:
    print("Warning: No accuracy values calculated for first plot, using default Y-axis range")
    plt.ylim(0.300, 0.500)

plt.xlabel('Cue', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
# plt.title('Accuracy Comparison: Concatenation vs CrossAttention Linear with Baseline', fontsize=14, pad=10)
plt.xticks(x, merged_df['Property'], rotation=0)       ## 掌控横坐标的文字倾斜教的
plt.legend(loc='best', fontsize=10)
plt.show()

# 创建第二个图形：CrossAttention Linear vs CrossAttentionMLP
plt.figure(figsize=(14, 6), dpi=100)

bar_width = 0.35
x = np.arange(len(merged_df['Property']))
offset = [-bar_width/2, bar_width/2]

datasets_2 = [
    {'name': 'LinearRegression', 'column': 'Accuracy_CrossAttention', 'color': '#ff3e0e'},  # 橙色
    {'name': 'MLP', 'column': 'Accuracy_CrossAttentionMLP', 'color': '#2ca02c'}  # 绿色
]

all_accuracies_2 = []
for i, dataset in enumerate(datasets_2):
    column = dataset['column']
    color = dataset['color']
    label = dataset['name']
    
    all_accuracies_2.extend(merged_df[column].tolist())
    print(f"{label}:\n", merged_df[column])
    
    x_positions = x + offset[i]
    bars = plt.bar(x_positions, merged_df[column], width=bar_width, color=color, label=label)
    
    for j, acc in enumerate(merged_df[column]):
        plt.text(x_positions[j], acc + 0.001, f'{acc:.3f}', ha='center', va='bottom', fontsize=10, color='black')

if all_accuracies_2:
    y_min = min(all_accuracies_2)
    y_max = max(all_accuracies_2)
    y_range = y_max - y_min
    y_margin = y_range * 0.2 if y_range > 0 else 0.05
    plt.ylim(max(0.3, y_min - y_margin), y_max + y_margin+0.01)
    print(f"Y-axis range for second plot: [{max(0.3, y_min - y_margin)}, {y_max + y_margin}]")
else:
    print("Warning: No accuracy values calculated for second plot, using default Y-axis range")
    plt.ylim(0.410, 0.465)

plt.xlabel('Cue', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
# plt.title('Accuracy Comparison: CrossAttention Linear vs CrossAttentionMLP', fontsize=14, pad=10)
plt.xticks(x, merged_df['Property'], rotation=0)
plt.legend(loc='best', fontsize=10)
plt.show()