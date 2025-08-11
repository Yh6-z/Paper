### 这个脚本的作用是获得所有的六个属性的文本数据集。


import pandas as pd
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

data_dir_senti_length = os.path.join(script_dir, "SENTIMENT_LENGTH label")  
data_dir_liwc = os.path.join(script_dir, "LIWC")    

train_updated_file = os.path.join(data_dir_senti_length, "Merge RESULT","Train_updated.csv")
test_updated_file = os.path.join(data_dir_senti_length, "Merge RESULT","Test_updated.csv")
New_headline_file = os.path.join(data_dir_senti_length, "Merge RESULT","New_headline_sentiemnt_text.csv")

train_liwc_file = os.path.join(data_dir_liwc, "train_liwc_with_labels_20250723_1121.csv")
test_liwc_file = os.path.join(data_dir_liwc, "test_liwc_with_labels_20250723_1121.csv")
New_headline_liwc_file = os.path.join(data_dir_liwc, "Headline_liwc_with_labels.csv")
# 读取数据集
try:
    train_updated = pd.read_csv(train_updated_file, encoding='utf-8', encoding_errors="ignore")
    test_updated = pd.read_csv(test_updated_file, encoding='utf-8', encoding_errors="ignore")
    New_headline = pd.read_csv(New_headline_file, encoding='utf-8', encoding_errors="ignore")

    train_liwc = pd.read_csv(train_liwc_file, encoding='utf-8', encoding_errors="ignore")
    test_liwc = pd.read_csv(test_liwc_file, encoding='utf-8', encoding_errors="ignore")
    New_headline_liwc = pd.read_csv(New_headline_liwc_file, encoding='utf-8', encoding_errors="ignore")
    print("All datasets loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    raise


# 验证数据集行数一致
def check_row_count(df1, df2, dataset_name):
    if len(df1) != len(df2):
        raise ValueError(f"Row count mismatch in {dataset_name}: {len(df1)} vs {len(df2)}")
    print(f"Row count verified for {dataset_name}: {len(df1)} rows")

check_row_count(train_updated, train_liwc, "train datasets")
check_row_count(test_updated, test_liwc, "test datasets")

# 确保索引对齐
train_updated = train_updated.reset_index(drop=True)
test_updated = test_updated.reset_index(drop=True)
train_liwc = train_liwc.reset_index(drop=True)
test_liwc = test_liwc.reset_index(drop=True)

# 提取 sentiment 和 length 列
columns_to_extract = ['sentiment', 'length']
train_updated_subset = train_updated[columns_to_extract]
test_updated_subset = test_updated[columns_to_extract]
New_headline_subset = New_headline[columns_to_extract]


# 基于索引合并（直接拼接列）
train_merged = pd.concat([train_liwc, train_updated_subset], axis=1)
test_merged = pd.concat([test_updated_subset, test_liwc], axis=1)
New_headline = pd.concat([New_headline_subset, New_headline_liwc], axis=1)

# 重命名结果数据集
Train_text_sum = train_merged
Test_text_sum = test_merged
New_headline_text_sum = New_headline

# 保存新数据集
output_dir = os.path.join(script_dir, "output_data")  # 替换为输出路径
os.makedirs(output_dir, exist_ok=True)

train_output_file = os.path.join(output_dir, "Train_text_sum.csv")
test_output_file = os.path.join(output_dir, "Test_text_sum.csv")
New_headline_output_file = os.path.join(output_dir, "New_headline_text_sum.csv")

Train_text_sum.to_csv(train_output_file, index=False)
Test_text_sum.to_csv(test_output_file, index=False)
New_headline_text_sum.to_csv(New_headline_output_file, index=False)
print(f"New datasets saved as {train_output_file} and {test_output_file}")