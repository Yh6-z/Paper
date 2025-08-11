## 使用Data process/ALL PROPERTY中merge_with_pron_one_hot.py的输出结果，位置为

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime
import random
import matplotlib.pyplot as plt

# 准确率评估函数
def evaluate_accuracy(df):
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
    print(f"准确率计算：正确 {correct}/{total}，准确率 {accuracy:.3f}")
    print(f"随机准确率：正确 {random_correct}/{total}，随机准确率 {random_accuracy:.3f}")
    return accuracy, random_accuracy, correct, random_correct, total

# 设置文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
train_file = os.path.join(parent_dir, "Data process", "output_data", "train_property.csv")
test_file = os.path.join(parent_dir, "Data process",  "output_data", "test_property.csv")

# 加载数据集
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print("训练集原始形状:", train_df.shape)
print("测试集原始形状:", test_df.shape)

# 处理 headline 列空值并打印处理数量
train_rows_before = train_df.shape[0]
train_df = train_df.dropna(subset=['headline'])
train_rows_after = train_df.shape[0]
print(f"训练集处理了 {train_rows_before - train_rows_after} 个 headline 列空值行，剩余形状: {train_df.shape}")

test_rows_before = test_df.shape[0]
test_df = test_df.dropna(subset=['headline'])
test_rows_after = test_df.shape[0]
print(f"测试集处理了 {test_rows_before - test_rows_after} 个 headline 列空值行，剩余形状: {test_df.shape}")

# 定义数值列（包括 one-hot 编码的 Pronoun 列）
numeric_cols = [
    'Concreteness_Score', 'Numeric_Structure_Score', 'Suspense_Question_Score',
    'Pronoun_First', 'Pronoun_Second', 'Pronoun_Third', 'Pronoun_Mixed', 'Pronoun_None',
    'prob_positive', 'prob_negative', 'prob_neutral', 'length_num'
]
required_cols = numeric_cols + ['CTR', 'test_id']

# 检查并处理其他列的空值
print("\n训练集其他列空值数量:")
print(train_df[required_cols].isna().sum())
print("测试集其他列空值数量:")
print(test_df[required_cols].isna().sum())

# 填充数值列和 CTR 的空值为 0，test_id 的空值为 -1
train_df[numeric_cols] = train_df[numeric_cols].fillna(0.0)
train_df['CTR'] = train_df['CTR'].fillna(0.0)
train_df['test_id'] = train_df['test_id'].fillna(-1)
test_df[numeric_cols] = test_df[numeric_cols].fillna(0.0)
test_df['CTR'] = test_df['CTR'].fillna(0.0)
test_df['test_id'] = test_df['test_id'].fillna(-1)

# 确保数值列是 float 类型
train_df[numeric_cols] = train_df[numeric_cols].astype(np.float32)
test_df[numeric_cols] = test_df[numeric_cols].astype(np.float32)
train_df['CTR'] = train_df['CTR'].astype(np.float32)

# 加载 bert-base-uncased 模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 移动模型到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def get_bert_embedding(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

# 生成训练集和测试集的嵌入
print("生成训练集嵌入...")
train_df['embedding'] = list(get_bert_embedding(train_df['headline'].tolist()))
print("生成测试集嵌入...")
test_df['embedding'] = list(get_bert_embedding(test_df['headline'].tolist()))

# 定义拼接逻辑，包括不拼接和全部拼接
attributes = {
    'Embedding': [],  # 0 维（仅嵌入）
    'Specificity': ['Concreteness_Score'],  # 1 维
    'Numeracy': ['Numeric_Structure_Score'],  # 1 维
    'Suspense': ['Suspense_Question_Score'],  # 1 维
    'Pronouns': ['Pronoun_First', 'Pronoun_Second', 'Pronoun_Third', 'Pronoun_Mixed', 'Pronoun_None'],  # 5 维
    'Emotion': ['prob_positive', 'prob_negative', 'prob_neutral'],  # 3 维
    'Readability': ['length_num'],  # 1 维
    'All_Attributes': numeric_cols  # 12 维
}

# 准备存储结果
results = {}
output_dir = os.path.join(script_dir, 'Fusion_results')
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# 对每个属性进行拼接和回归
for attr_name, attr_cols in attributes.items():
    print(f"\n处理属性: {attr_name}")
    
    # 拼接向量
    train_embeddings = np.stack(train_df['embedding'].values)  # (n_samples, 768)
    if attr_cols:  # 如果有属性列需要拼接
        attr_values = train_df[attr_cols].values  # (n_samples, n_cols)
        train_X = np.concatenate([train_embeddings, attr_values], axis=1)  # (n_samples, 768 + n_cols)
    else:  # 不拼接，仅使用嵌入
        train_X = train_embeddings
    
    test_embeddings = np.stack(test_df['embedding'].values)
    if attr_cols:
        test_attr_values = test_df[attr_cols].values
        test_X = np.concatenate([test_embeddings, test_attr_values], axis=1)
    else:
        test_X = test_embeddings
    
    # 目标变量
    train_y = train_df['CTR'].values
    test_y = test_df['CTR'].values
    
    # 训练线性回归模型
    model = LinearRegression()
    model.fit(train_X, train_y)
    
    # 预测
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)
    
    # 评估准确率
    result_df = test_df.copy()
    result_df['pred_CTR'] = test_pred
    accuracy, random_accuracy, correct, random_correct, total = evaluate_accuracy(result_df)
    
    # 存储结果
    results[attr_name] = {
        'accuracy': accuracy,
        'random_accuracy': random_accuracy,
        'correct': correct,
        'total': total,
        'vector_dim': train_X.shape[1]
    }
    
    # 保存预测结果
    # output_file = os.path.join(output_dir, f'{attr_name}_predictions_{timestamp}.csv')
    # result_df.to_csv(output_file, index=False)
    # print(f"预测结果已保存至: {output_file}")


print("\n回归结果汇总:")
for attr_name, metrics in results.items():
    print(f"{attr_name} (维度: {metrics['vector_dim']}):")
    print(f"  准确率: {metrics['accuracy']:.3f} (正确 {metrics['correct']}/{metrics['total']})")
    print(f"  随机准确率: {metrics['random_accuracy']:.3f}")

# 保存回归结果汇总为 CSV
summary_data = {
    'Method': [],
    'Dimension': [],
    'Accuracy': [],
    'Random_Accuracy': [],
    'Correct': [],
    'Total': []
}
for attr_name, metrics in results.items():
    summary_data['Method'].append(attr_name)
    summary_data['Dimension'].append(metrics['vector_dim'])
    summary_data['Accuracy'].append(metrics['accuracy'])
    summary_data['Random_Accuracy'].append(metrics['random_accuracy'])
    summary_data['Correct'].append(metrics['correct'])
    summary_data['Total'].append(metrics['total'])

summary_df = pd.DataFrame(summary_data)
summary_file = os.path.join(script_dir, "Fusion_results",f"regression_results_{timestamp}.csv")


## summary_file = 
summary_df.to_csv(summary_file, index=False)
print(f"\n回归结果汇总已保存至: {summary_file}")

# 可视化柱状图
print("生成准确率对比柱状图...")
df = pd.read_csv(summary_file)
x = np.arange(len(df))

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})

# 设置画布大小
plt.figure(figsize=(14, 6))

# 设置柱状图宽度和位置
bar_width = 0.35
x = np.arange(len(df['Method']))

# 绘制柱状图
plt.bar(x, df['Accuracy'], bar_width, color='skyblue', label='Model Accuracy')
plt.bar(x + bar_width, df['Random_Accuracy'], bar_width, color='lightcoral', label='Random Accuracy')

# 添加数值标签
for i, acc in enumerate(df['Accuracy']):
    plt.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom', fontsize=10, color='skyblue')
for i, rand_acc in enumerate(df['Random_Accuracy']):
    plt.text(i + bar_width, rand_acc + 0.002, f'{rand_acc:.4f}', ha='center', va='bottom', fontsize=10, color='lightcoral')

# 设置轴和标题
plt.xticks(x + bar_width / 2, df['Method'], rotation=20, ha='right')
plt.xlabel('Method', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy Comparison of Different Concatenation Methods', fontsize=14, pad=10)

# 设置纵轴范围
plt.ylim(0.3, 0.475)

# 添加图例和网格
plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)

# 优化布局并保存
plt.tight_layout()
os.makedirs(os.path.join(output_dir), exist_ok=True)
plot_path = os.path.join(output_dir, f"accuracy_comparison_{timestamp}.png")


plt.show()