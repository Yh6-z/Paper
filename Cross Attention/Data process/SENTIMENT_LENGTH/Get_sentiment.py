### 这个脚本的作用是获得数据集中对headline文本数据的情感极性分析结果。


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import os
from datetime import datetime
import logging

#  加载数据集
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

train_file = os.path.join(parent_dir, "original data","LoRA_CTR_train.csv")
test_file = os.path.join(parent_dir, "original data","LoRA_CTR_test.csv")
headline_file = os.path.join(parent_dir, "LIWC","Headline_liwc.csv")

# test_df = pd.read_csv(test_file, encoding='utf-8', encoding_errors='ignore') 
# train_df = pd.read_csv(train_file, encoding='utf-8', encoding_errors='ignore')
headline_df = pd.read_csv(headline_file)
# print("成功加载两个数据集", test_df.shape, train_df.shape)

#  设置输出位置
timestamp = datetime.now().strftime("%Y%m%d_%H%M")   
output_file_train = os.path.join(script_dir, f"Train_sentiment_{timestamp}.csv")
# logging.info(f"Output file will be saved to: {output_file_train}")
    
output_file_test = os.path.join(script_dir, f"Test_sentiment_{timestamp}.csv")
# logging.info(f"Output file will be saved to: {output_file_test}")

output_headline = os.path.join(script_dir, f"New_headline_sentiment.csv")

# # 在加载数据集后，检查空值
# print("训练集中 headline 列的空值数量:", train_df["headline"].isna().sum())
# # 锁定空值的位置（行索引）
# na_indices = train_df[train_df["headline"].isna()].index
# print("空值所在的行索引:", na_indices.tolist())
# print("测试集中的空值位置",test_df["headline"].isna().sum())  # 打印空值数量
# headlines_train = train_df["headline"].tolist() 
# headlines_test = test_df["headline"].tolist()
headlines_new = headline_df['headline'].tolist()


# # 验证输入格式
# if not all(isinstance(h, str) for h in headlines_train):
#     raise ValueError("Some headlines are not strings. Please check the data.")

# 加载模型和分词器
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 创建 pipeline 并启用 GPU
classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0,  # 使用 GPU
    return_all_scores=True  # 返回所有类别的概率
)

# 5. 提取概率分数
batch_size = 32
probs_list_train = []
probs_list_test = []

# for i in range(0, len(headlines_train), batch_size):
#     batch = headlines_train[i:i + batch_size]
#     try:
#         results = classifier(batch, batch_size=batch_size)
#         # 提取概率（results 是 List[List[Dict]]，每个 Dict 包含 label 和 score）
#         for result in results:
#             probs = [score["score"] for score in result]  # [negative, neutral, positive]
#             probs_list_train.append(probs)
#     except Exception as e:
#         print(f"Error processing batch {i // batch_size}: {e}")
#         probs_list_train.extend([[0.0, 0.0, 0.0]] * len(batch))  # 用占位符填充

# # 将概率添加到 train_df
# train_df["prob_negative"] = [probs[0] for probs in probs_list_train]
# train_df["prob_neutral"] = [probs[1] for probs in probs_list_train]
# train_df["prob_positive"] = [probs[2] for probs in probs_list_train]

# for j in range(0, len(headlines_test), batch_size):
#     batch = headlines_test[j:j + batch_size]
#     try:
#         results = classifier(batch, batch_size=batch_size)
#         # 提取概率（results 是 List[List[Dict]]，每个 Dict 包含 label 和 score）
#         for result in results:
#             probs = [score["score"] for score in result]  # [negative, neutral, positive]
#             probs_list_test.append(probs)
#     except Exception as e:
#         print(f"Error processing batch {j // batch_size}: {e}")
#         probs_list_test.extend([[0.0, 0.0, 0.0]] * len(batch))  # 用占位符填充

# # 将概率添加到 test_df
# test_df["prob_negative"] = [probs[0] for probs in probs_list_test]
# test_df["prob_neutral"] = [probs[1] for probs in probs_list_test]
# test_df["prob_positive"] = [probs[2] for probs in probs_list_test]

for k in range(0, len(headlines_new), batch_size):
    batch = headlines_new[k:k + batch_size]
    try:
        results = classifier(batch, batch_size=batch_size)
        # 提取概率（results 是 List[List[Dict]]，每个 Dict 包含 label 和 score）
        for result in results:
            probs = [score["score"] for score in result]  # [negative, neutral, positive]
            probs_list_test.append(probs)
    except Exception as e:
        print(f"Error processing batch {k // batch_size}: {e}")
        probs_list_test.extend([[0.0, 0.0, 0.0]] * len(batch))  # 用占位符填充

# 将概率添加到 test_df
headline_df["prob_negative"] = [probs[0] for probs in probs_list_test]
headline_df["prob_neutral"] = [probs[1] for probs in probs_list_test]
headline_df["prob_positive"] = [probs[2] for probs in probs_list_test]

# 6. 根据最大概率确定 sentiment 标签
def get_sentiment_label(row):
    probs = [row["prob_negative"], row["prob_neutral"], row["prob_positive"]]
    max_prob = max(probs)
    if max_prob == row["prob_negative"]:
        return "negative"
    elif max_prob == row["prob_neutral"]:
        return "neutral"
    else:
        return "positive"

#train_df["sentiment"] = train_df.apply(get_sentiment_label, axis=1)
#test_df["sentiment"] = test_df.apply(get_sentiment_label, axis=1)
headline_df["sentiment"] = headline_df.apply(get_sentiment_label, axis=1)

# 保存结果
# train_df.to_csv(output_file_train, index=False)
# test_df.to_csv(output_file_test, index=False)

headline_df.to_csv(output_headline, index = False)

