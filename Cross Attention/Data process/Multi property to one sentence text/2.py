import pandas as pd
import os
from itertools import combinations

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# 定义数据文件路径
data_dir = os.path.join(parent_dir, "output_data")
train_file = os.path.join(data_dir, "Train_text_sum.csv")
test_file = os.path.join(data_dir, "Test_text_sum.csv")

# 提取函数
def extract_sentiment(sentiment_str):
    try:
        return str(sentiment_str).replace("The headline ", "").strip()      
        ## 将sentiment_dir中的"The headline "替换为空, .strip()移除首尾空字符串
    except (ValueError, AttributeError):
        return None

def extract_length(length_str):
    try:
        # value = str(length_str).replace("The headline ", "").strip()
        return str(length_str).replace("The headline ", "").strip()
    except (ValueError, AttributeError):
        return None

def extract_concreteness(concreteness_str):
    try:
        return str(concreteness_str).replace("The headline ", "").strip()
    except (ValueError, AttributeError):
        return None

def extract_numeric(numeric_str):
    try:
        return str(numeric_str).replace("The headline ", "").strip()
    except (ValueError, AttributeError):
        return None

def extract_suspense(suspense_str):
    try:
        return str(suspense_str).replace("The headline ", "").strip()
    except (ValueError, AttributeError):
        return None

def extract_pronoun(pronoun_str):
    try:
        return str(pronoun_str).replace("The headline ", "").strip()
    except (ValueError, AttributeError):
        return None

# 读取数据集
try:
    train_df = pd.read_csv(train_file, encoding="utf-8", encoding_errors="ignore")
    test_df = pd.read_csv(test_file, encoding="utf-8", encoding_errors="ignore")
    print("Datasets loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    raise

# 定义属性
attributes = ['sentiment', 'length', 'Concreteness_Label', 'Numeric_Label', 'Suspense_Label', 'Pronoun_Label']

# 提取值
extract_functions = {
    'sentiment': extract_sentiment,
    'length': extract_length,
    'Concreteness_Label': extract_concreteness,
    'Numeric_Label': extract_numeric,
    'Suspense_Label': extract_suspense,
    'Pronoun_Label': extract_pronoun
}

for df in [train_df, test_df]:
    for attr, func in extract_functions.items():
        df[f"{attr}_value"] = df[attr].apply(func)

# 生成二二组合
two_combinations = list(combinations(attributes, 2))

# 处理每个数据集
for df, file_name in [(train_df, 'Train_text_sum_two_combinations.csv'), 
                      (test_df, 'Test_text_sum_two_combinations.csv')]:
    # 为每个组合生成新列
    for combo in two_combinations:
        attr1, attr2 = combo
        col_name = f"{attr1}_{attr2}"
        df[col_name] = df.apply(
            lambda row: f"The headline {row[f'{attr1}_value']}, {row[f'{attr2}_value']}"
            if pd.notnull(row[f'{attr1}_value']) and pd.notnull(row[f'{attr2}_value'])
            else None, axis=1)

    # 删除临时列
    df = df.drop(columns=[f"{attr}_value" for attr in attributes], errors='ignore')

    # 保存结果
    output_dir = os.path.join(parent_dir, "output_data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name)
    df.to_csv(output_file, index=False)
    print(f"Saved {file_name} with two combinations")