#### 获得各个属性的数值型数据，并合并为一个完整的数据集，结果为train_property和test_property


import pandas as pd
import numpy as np
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"脚本目录: {script_dir}")
parent_dir = os.path.dirname(script_dir)  # e:\VS code project\Cross Attention\Data process

# 加载六个测试集和训练集预测结果的 CSV 文件
try:
    # 包含 Concreteness, Numeric, Suspense, Pronoun 的 score 和 text label 数据
    train_LIWC_path = os.path.join(parent_dir, "LIWC", "train_liwc_with_labels_20250714_1350.csv")
    test_LIWC_path = os.path.join(parent_dir, "LIWC", "test_liwc_with_labels_20250714_1350.csv")
    
    # 包含 sentiment, length 的 score 和 text label 数据
    train_sl_path = os.path.join(parent_dir, "SENTIMENT_LENGTH label", "Merge RESULT", "Train_updated.csv")
    test_sl_path = os.path.join(parent_dir, "SENTIMENT_LENGTH label", "Merge RESULT", "Test_updated.csv")

    # 验证文件存在
    for path in [train_LIWC_path, test_LIWC_path, train_sl_path, test_sl_path]:
        if not os.path.exists(path):
            print(f"错误：文件 {path} 不存在")
            exit(1)

    # 加载数据
    try:
        train_liwc_df = pd.read_csv(train_LIWC_path)
        test_liwc_df = pd.read_csv(test_LIWC_path)
        train_sl_df = pd.read_csv(train_sl_path)
        test_sl_df = pd.read_csv(test_sl_path)
    except UnicodeDecodeError:
        print("警告：默认编码失败，尝试使用 utf-8 编码加载")
        train_liwc_df = pd.read_csv(train_LIWC_path, encoding='utf-8', encoding_errors="ignore")
        test_liwc_df = pd.read_csv(test_LIWC_path, encoding='utf-8', encoding_errors="ignore")
        train_sl_df = pd.read_csv(train_sl_path, encoding='utf-8')
        test_sl_df = pd.read_csv(test_sl_path, encoding='utf-8')
except FileNotFoundError as e:
    print(f"错误：无法找到文件 {e.filename}")
    exit(1)

# 对 Pronoun_Usage 进行 one-hot 编码
def encode_pronoun_usage(df):
    """将 Pronoun_Usage 列进行 one-hot 编码，空值或 NaN 视为 None"""
    pronoun_categories = ['First', 'Second', 'Third', 'Mixed', 'None']
    one_hot_columns = ['Pronoun_First', 'Pronoun_Second', 'Pronoun_Third', 'Pronoun_Mixed', 'Pronoun_None']
    
    # 初始化 one-hot 列
    for col in one_hot_columns:
        df[col] = 0
    
    # 处理空值或 NaN
    df['Pronoun_Usage'] = df['Pronoun_Usage'].fillna('None')
    
    # 进行 one-hot 编码
    for idx, value in df['Pronoun_Usage'].items():
        if value in pronoun_categories:
            col_name = f'Pronoun_{value}'
            df.at[idx, col_name] = 1
    
    # 删除原始 Pronoun_Usage 列
    df = df.drop(columns=['Pronoun_Usage'])
    
    return df

# 在加载数据后立即对 Pronoun_Usage 进行编码
train_liwc_df = encode_pronoun_usage(train_liwc_df)
test_liwc_df = encode_pronoun_usage(test_liwc_df)

def verify_consistency(df1, df2, dataset_type="Train"):
    """检查两个数据集的 test_id, headline, CTR 一致性"""
    base_columns = ['test_id', 'headline', 'CTR']
    
    print(f"\n{dataset_type} 数据集一致性检查：")
    print(f"{dataset_type} LIWC 总行数: {len(df1)}, 唯一 test_id 数量: {df1['test_id'].nunique()}")
    print(f"{dataset_type} SL 总行数: {len(df2)}, 唯一 test_id 数量: {df2['test_id'].nunique()}")
    
    # 检查列
    for df, name in [(df1, f"{dataset_type} LIWC"), (df2, f"{dataset_type} SL")]:
        if not all(col in df.columns for col in base_columns):
            print(f"错误：{name} 数据集缺少必要列，现有列：{df.columns.tolist()}")
            return False
    
    # 标准化数据
    df1 = df1.copy()
    df2 = df2.copy()
    for df in [df1, df2]:
        if df['headline'].dtype == 'object':
            df['headline'] = df['headline'].str.strip().fillna("")
        df['CTR'] = df['CTR'].round(6).fillna(0)
    
    # 检查行数
    if len(df1) != len(df2):
        print(f"警告：{dataset_type} 数据集行数不一致，LIWC: {len(df1)}, SL: {len(df2)}")
        missing_ids = set(df1['test_id']) - set(df2['test_id'])
        if missing_ids:
            print(f"{dataset_type} SL 缺失的 test_id: {missing_ids}")
            print(f"缺失行数据:\n{df1[df1['test_id'].isin(missing_ids)][base_columns]}")
    
    # 检查 test_id 值一致性（在公共 test_id 范围内）
    common_ids = set(df1['test_id']) & set(df2['test_id'])
    df1_common = df1[df1['test_id'].isin(common_ids)][base_columns].sort_values('test_id').reset_index(drop=True)
    df2_common = df2[df2['test_id'].isin(common_ids)][base_columns].sort_values('test_id').reset_index(drop=True)
    
    # 确保列顺序一致
    df1_common = df1_common[base_columns]
    df2_common = df2_common[base_columns]
    
    # 打印数据类型
    print(f"{dataset_type} LIWC 数据类型:\n{df1_common.dtypes}")
    print(f"{dataset_type} SL 数据类型:\n{df2_common.dtypes}")
    
    # 比较值一致性
    try:
        diff_mask = (df1_common != df2_common) & ~(df1_common.isna() & df2_common.isna())
        diff_indices = diff_mask.any(axis=1)
        if diff_indices.any():
            print(f"警告：{dataset_type} 数据集的 test_id, headline, CTR 不一致")
            print(f"不一致的行（LIWC 数据）:\n{df1_common[diff_indices]}")
            print(f"不一致的行（SL 数据）:\n{df2_common[diff_indices]}")
        else:
            print(f"{dataset_type} 数据集的 test_id, headline, CTR 一致")
    except ValueError as e:
        print(f"错误：无法比较 DataFrame，原因：{e}")
        print(f"LIWC 列: {df1_common.columns.tolist()}")
        print(f"SL 列: {df2_common.columns.tolist()}")
        return True  # 继续拼接
    
    return True

def merge_predictions():
    # 验证必要列
    required_liwc_cols = ['test_id', 'headline', 'CTR', 'Concreteness_Score', 'Numeric_Structure_Score', 
                          'Suspense_Question_Score', 'Pronoun_First', 'Pronoun_Second', 'Pronoun_Third', 
                          'Pronoun_Mixed', 'Pronoun_None']
    required_sl_cols = ['test_id', 'headline', 'CTR', 'prob_positive', 'prob_negative', 'prob_neutral', 'length_num']
    
    for df, name in [(train_liwc_df, "Train LIWC"), (test_liwc_df, "Test LIWC")]:
        if not all(col in df.columns for col in required_liwc_cols):
            print(f"错误：{name} 数据集缺少必要列，现有列：{df.columns.tolist()}")
            return None, None
    for df, name in [(train_sl_df, "Train SL"), (test_sl_df, "Test SL")]:
        if not all(col in df.columns for col in required_sl_cols):
            print(f"错误：{name} 数据集缺少必要列，现有列：{df.columns.tolist()}")
            return None, None
    
    # 验证一致性
    if not verify_consistency(train_liwc_df, train_sl_df, dataset_type="Train"):
        print("警告：训练集数据验证失败，但将尝试拼接")
    if not verify_consistency(test_liwc_df, test_sl_df, dataset_type="Test"):
        print("警告：测试集数据验证失败，但将尝试拼接")
    
    # 优化内存：转换为 float32
    for col in ['Concreteness_Score', 'Numeric_Structure_Score', 'Suspense_Question_Score', 
                'prob_positive', 'prob_negative', 'prob_neutral']:
        if col in train_liwc_df.columns:
            train_liwc_df[col] = train_liwc_df[col].astype('float32')
            test_liwc_df[col] = test_liwc_df[col].astype('float32')
        if col in train_sl_df.columns:
            train_sl_df[col] = train_sl_df[col].astype('float32')
            test_sl_df[col] = test_sl_df[col].astype('float32')
    
    # 基于行索引拼接
    min_rows_train = min(len(train_liwc_df), len(train_sl_df))
    min_rows_test = min(len(test_liwc_df), len(test_sl_df))
    
    train_property = train_liwc_df[['test_id', 'headline', 'CTR', 'Concreteness_Score', 
                                    'Numeric_Structure_Score', 'Suspense_Question_Score', 
                                    'Pronoun_First', 'Pronoun_Second', 'Pronoun_Third', 
                                    'Pronoun_Mixed', 'Pronoun_None']].iloc[:min_rows_train].copy()
    train_property[['prob_positive', 'prob_negative', 'prob_neutral', 'length_num']] = \
        train_sl_df[['prob_positive', 'prob_negative', 'prob_neutral', 'length_num']].iloc[:min_rows_train]
    
    test_property = test_liwc_df[['test_id', 'headline', 'CTR', 'Concreteness_Score', 
                                  'Numeric_Structure_Score', 'Suspense_Question_Score', 
                                  'Pronoun_First', 'Pronoun_Second', 'Pronoun_Third', 
                                  'Pronoun_Mixed', 'Pronoun_None']].iloc[:min_rows_test].copy()
    test_property[['prob_positive', 'prob_negative', 'prob_neutral', 'length_num']] = \
        test_sl_df[['prob_positive', 'prob_negative', 'prob_neutral', 'length_num']].iloc[:min_rows_test]
    
    # 检查重复 test_id
    print(f"\n训练集重复 test_id 统计: {train_property['test_id'].value_counts().head()}")
    print(f"测试集重复 test_id 统计: {test_property['test_id'].value_counts().head()}")
    
    # 创建输出目录
    output_dir = os.path.join(parent_dir, "output_data")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    
    # 保存合并结果
    output_train_path = os.path.join(output_dir, "train_property.csv")
    output_test_path = os.path.join(output_dir, "test_property.csv")
    
    train_property.to_csv(output_train_path, index=False)
    test_property.to_csv(output_test_path, index=False)
    
    # 打印验证信息
    print(f"\n训练集合并数据集列：{train_property.columns.tolist()}")
    print(f"训练集合并数据集形状：{train_property.shape}")
    print("\n训练集前几行数据预览：")
    print(train_property.head())
    print(f"\n训练集合并结果已保存到 {output_train_path}")
    
    print(f"\n测试集合并数据集列：{test_property.columns.tolist()}")
    print(f"测试集合并数据集形状：{test_property.shape}")
    print("\n测试集前几行数据预览：")
    print(test_property.head())
    print(f"\n测试集合并结果已保存到 {output_test_path}")
    
    return train_property, test_property

if __name__ == "__main__":
    # 检查空值（针对 train_liwc_with_labels_20250625_1816.csv 的 53965 行问题）
    print("训练集中 headline 列的空值数量:", train_liwc_df["headline"].isna().sum())
    na_indices = train_liwc_df[train_liwc_df["headline"].isna()].index
    print("空值所在的行索引:", na_indices.tolist())  # 索引位置 [3544]，Excel 位置 [3546]
    
    train_property, test_property = merge_predictions()
    if train_property is not None and test_property is not None:
        print("合并完成！")
    else:
        print("合并失败，请检查错误信息。")