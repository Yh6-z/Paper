### 这个脚本的作用是根据预测结果获得六个属性的准确率的csv文件及其可视化结果

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__)) # 获得当前脚本所在的目录的绝对路径


df_CONCRETENESS = pd.read_csv(os.path.join(script_dir,"Concreteness RESULT","Test_Cro_Atten_concreteness_pred_seed.csv"))
df_NUMERIC = pd.read_csv(os.path.join(script_dir,"Numeric RESULT","Test_Cro_Atten_numeric_pred_seed.csv"))
df_PRONOUN = pd.read_csv(os.path.join(script_dir,"Pronoun RESULT","Test_Cro_Atten_pronoun_pred_seed.csv"))
df_SUSPENSE = pd.read_csv(os.path.join(script_dir,"Suspense RESULT","Test_Cro_Atten_suspense_pred_seed.csv"))
df_SENTIMENT = pd.read_csv(os.path.join(script_dir,"Sentiment RESULT","Test_Cro_Atten_sentiment_pred_seed.csv"))
df_LENGTH = pd.read_csv(os.path.join(script_dir,"Length RESULT","Test_Cro_Atten_length_pred_seed.csv"))


def evaluate_accuracy(df,property_name):

    if df.empty or "pred_CTR" not in df.columns:
        print("错误：数据集为空或缺少 pred_CTR 列，跳过准确率计算")
        return 0.0, 0.0, 0, 0, 0
    
    correct = 0
    random_correct = 0
    total = 0
    
    for test_id, group in df.groupby("test_id"):
        if len(group) == 0:
            continue
        
        max_ctr = group["CTR"].max()
        true_max_indices = group[group["CTR"] == max_ctr].index.tolist()
        
        pred_max_idx = group["pred_CTR"].idxmax()
        valid_indices = group.index.tolist()
        
        total += 1
        if pred_max_idx in true_max_indices:
            correct += 1
        
        random_idx = random.choice(valid_indices)
        if random_idx in true_max_indices:
            random_correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    random_accuracy = random_correct / total if total > 0 else 0.0
    print(f"准确率计算：正确 {correct}/{total}，准确率 {accuracy:.4f}")
    print(f"随机准确率：正确 {random_correct}/{total}，随机准确率 {random_accuracy:.4f}")
    return accuracy, random_accuracy, correct, random_correct, total

def main():
    # 定义 DataFrame 和特征名称的映射
    dfs = {
        'Specificity': df_CONCRETENESS,
        'Numeracy': df_NUMERIC,
        'Pronouns': df_PRONOUN,
        'Suspense': df_SUSPENSE,
        'Emotion': df_SENTIMENT,
        'Readability': df_LENGTH
    }
    
    # 存储结果
    results = []
    
    # 对每个 DataFrame 评估准确率
    for property_name, df in dfs.items():
        print(f"\n评估 {property_name} 的准确率...")
        accuracy, random_accuracy, correct, random_correct, total = evaluate_accuracy(df, property_name)
        results.append({
            'Property': property_name,
            'Accuracy': accuracy,
            'Random_Accuracy': random_accuracy,
            'Correct': correct,
            'Total': total
        })
    
    # 创建结果 DataFrame
    results_df = pd.DataFrame(results)
    
    # 创建输出目录
    output_dir = os.path.join(script_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果到 CSV
    output_path = os.path.join(output_dir, "Attention_MLP_accuracy_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n准确率结果已保存到 {output_path}")
    
    # 打印结果表格
    print("\n准确率汇总：")
    print(results_df)




    # 可视化准确率（柱状图）
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.35
    x = np.arange(len(results_df['Property']))
    
    plt.bar(x, results_df['Accuracy'], bar_width, color='skyblue', label='Model Accuracy')
    plt.bar(x + bar_width, results_df['Random_Accuracy'], bar_width, color='lightcoral', label='Random Accuracy')
    
    for i, acc in enumerate(results_df['Accuracy']):
        plt.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom', fontsize=10, color='skyblue')
    for i, rand_acc in enumerate(results_df['Random_Accuracy']):
        plt.text(i + bar_width, rand_acc - 0.005, f'{rand_acc:.4f}', ha='center', va='top', fontsize=10, color='lightcoral')
    
    plt.xlabel('Property', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Comparison by Property', fontsize=14, pad=10)
    plt.ylim(0.3, 0.5)
    plt.xticks(x + bar_width / 2, results_df['Property'], rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # plot_path = os.path.join(output_dir, "accuracy_bar_plot.png")
    # plt.savefig(plot_path, dpi=300)
    # print(f"准确率柱状图已保存到 {plot_path}")
    
    plt.show()
if __name__ == "__main__":
    main()
