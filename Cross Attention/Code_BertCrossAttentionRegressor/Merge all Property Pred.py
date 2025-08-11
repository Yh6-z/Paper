### 用于将六个属性的预测结果合并到一个数据集中，主要用于计算Mean max min的预测


import pandas as pd
import numpy as np
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"脚本目录: {script_dir}")

# 加载六个测试集和训练集预测结果的 CSV 文件
try:
    # 训练集
    df_CONCRETENESS_test = pd.read_csv(os.path.join(script_dir, "Concreteness RESULT", "Test_Cro_Atten_concreteness_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_NUMERIC_test = pd.read_csv(os.path.join(script_dir, "Numeric RESULT", "Test_Cro_Atten_numeric_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_PRONOUN_test = pd.read_csv(os.path.join(script_dir, "Pronoun RESULT", "Test_Cro_Atten_pronoun_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_SUSPENSE_test = pd.read_csv(os.path.join(script_dir, "Suspense RESULT", "Test_Cro_Atten_suspense_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_SENTIMENT_test = pd.read_csv(os.path.join(script_dir, "Sentiment RESULT", "Test_Cro_Atten_sentiment_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_LENGTH_test = pd.read_csv(os.path.join(script_dir, "Length RESULT", "Test_Cro_Atten_length_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    
    # 训练集
    df_CONCRETENESS_train = pd.read_csv(os.path.join(script_dir, "Concreteness RESULT", "Train_Cro_Atten_concreteness_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_NUMERIC_train = pd.read_csv(os.path.join(script_dir, "Numeric RESULT", "Train_Cro_Atten_numeric_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_PRONOUN_train = pd.read_csv(os.path.join(script_dir, "Pronoun RESULT", "Train_Cro_Atten_pronoun_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_SUSPENSE_train = pd.read_csv(os.path.join(script_dir, "Suspense RESULT", "Train_Cro_Atten_suspense_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_SENTIMENT_train = pd.read_csv(os.path.join(script_dir, "Sentiment RESULT", "Train_Cro_Atten_sentiment_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")
    df_LENGTH_train = pd.read_csv(os.path.join(script_dir, "Length RESULT", "Train_Cro_Atten_length_pred_seed.csv"), encoding="utf-8",encoding_errors="ignore")

except FileNotFoundError as e:
    print(f"错误：无法找到文件 {e.filename}")
    exit(1)

def verify_consistency(dfs, property_names, dataset_type="Test"):
    """检查数据集列、总行数和 test_id 一致性"""
    base_df = dfs[property_names[0]]  # 以第一个数据集为基准
    base_columns = ['test_id', 'headline', 'CTR']
    min_rows = min(len(df) for df in dfs.values())  # 取最小行数
    
    print(f"\n{dataset_type} 数据集形状、总行数和列检查：")
    for prop, df in dfs.items():
        # 检查类型
        if not isinstance(df, pd.DataFrame):
            print(f"错误：{prop} 数据预期为 DataFrame 类型，实际得到 {type(df)}")
            return False
        # 检查列
        if not all(col in df.columns for col in base_columns + ['pred_CTR']):
            print(f"错误：{prop} 数据缺少必要列，现有列：{df.columns.tolist()}")
            return False
        # 打印总行数和形状
        print(f"{prop} 总行数: {len(df)}, 形状: {df.shape}, 唯一 test_id 数量: {df['test_id'].nunique()}")
        # 检查 test_id 重复（仅警告）

    
    # 检查 test_id, headline, CTR 一致性（在最小行数范围内）
    for prop, df in dfs.items():
        if len(df) > min_rows:
            print(f"警告：{prop} 行数 {len(df)} 超过最小行数 {min_rows}，将截取前 {min_rows} 行")
        df_subset = df.iloc[:min_rows][base_columns]
    return True

def merge_predictions():
    # 定义测试集和训练集 DataFrame 和特征名称的映射
    dfs_test = {
        'Concreteness': df_CONCRETENESS_test,
        'Numeric': df_NUMERIC_test,
        'Pronoun': df_PRONOUN_test,
        'Suspense': df_SUSPENSE_test,
        'Sentiment': df_SENTIMENT_test,
        'Length': df_LENGTH_test
    }
    
    dfs_train = {
        'Concreteness': df_CONCRETENESS_train,
        'Numeric': df_NUMERIC_train,
        'Pronoun': df_PRONOUN_train,
        'Suspense': df_SUSPENSE_train,
        'Sentiment': df_SENTIMENT_train,
        'Length': df_LENGTH_train
    }
    
    # 验证测试集和训练集数据集
    print("\n验证测试集数据...")
    if not verify_consistency(dfs_test, list(dfs_test.keys()), dataset_type="Test"):
        print("错误：测试集数据集验证失败，无法合并")
        return None, None
    
    print("\n验证训练集数据...")
    if not verify_consistency(dfs_train, list(dfs_train.keys()), dataset_type="Train"):
        print("错误：训练集数据集验证失败，无法合并")
        return None, None
    
    # 优化内存：将 pred_CTR 转换为 float32
    for prop, df in dfs_test.items():
        df['pred_CTR'] = df['pred_CTR'].astype('float32')
    for prop, df in dfs_train.items():
        df['pred_CTR'] = df['pred_CTR'].astype('float32')
    
    # 取最小行数对齐
    min_rows_test = min(len(df) for df in dfs_test.values())
    min_rows_train = min(len(df) for df in dfs_train.values())
    print(min_rows_test)
    print(min_rows_train)

    # 测试集：以 Concreteness 作为基础，提取 test_id, headline, CTR
    merged_df_test = dfs_test['Concreteness'][['test_id', 'headline', 'CTR']].iloc[:min_rows_test].copy()
    # 训练集：以 Concreteness 作为基础，提取 test_id, headline, CTR
    merged_df_train = dfs_train['Concreteness'][['test_id', 'headline', 'CTR']].iloc[:min_rows_train].copy()
    
    # 拼接每个属性的 pred_CTR 列
    for prop, df in dfs_test.items():
        merged_df_test[f'pred_CTR_{prop}'] = df['pred_CTR'].iloc[:min_rows_test]
    
    for prop, df in dfs_train.items():
        merged_df_train[f'pred_CTR_{prop}'] = df['pred_CTR'].iloc[:min_rows_train]
    
    # 创建输出目录
    output_dir = os.path.join(script_dir, "Merge RESULT")
    os.makedirs(output_dir, exist_ok=True)
    

    # 保存合并结果到 CSV
    output_test_path = os.path.join(output_dir, "merged_test_predictions_seed.csv")
    merged_df_test.to_csv(output_test_path, index=False)
    
    output_train_path = os.path.join(output_dir, "merged_train_predictions_seed.csv")
    merged_df_train.to_csv(output_train_path, index=False)

    
    # 打印验证信息
    print(f"\n测试集合并数据集列：{merged_df_test.columns.tolist()}")
    print(f"测试集合并数据集形状：{merged_df_test.shape}")
    print("\n测试集前几行数据预览：")
    print(merged_df_test.head())
    print(f"\n测试集合并结果已保存到 {output_test_path}")
    
    print(f"\n训练集合并数据集列：{merged_df_train.columns.tolist()}")
    print(f"训练集合并数据集形状：{merged_df_train.shape}")
    print("\n训练集前几行数据预览：")
    print(merged_df_train.head())
    print(f"\n训练集合并结果已保存到 {output_train_path}")
    
    # 打印 Sentiment 和 Length 的缺失行
    if len(dfs_train['Sentiment']) < len(dfs_train['Concreteness']):
        missing_indices = dfs_train['Concreteness'].index.difference(dfs_train['Sentiment'].index)
        print(f"\nSentiment 训练集缺失的索引: {missing_indices.tolist()}")
        print(f"Sentiment 缺失行的 test_id: {dfs_train['Concreteness'].loc[missing_indices, 'test_id'].tolist()}")
    if len(dfs_train['Length']) < len(dfs_train['Concreteness']):
        missing_indices = dfs_train['Concreteness'].index.difference(dfs_train['Length'].index)
        print(f"Length 训练集缺失的索引: {missing_indices.tolist()}")
        print(f"Length 缺失行的 test_id: {dfs_train['Concreteness'].loc[missing_indices, 'test_id'].tolist()}")
    
    return merged_df_test, merged_df_train

if __name__ == "__main__":
    merged_df_test, merged_df_train = merge_predictions()
    if merged_df_test is not None and merged_df_train is not None:
        print("合并完成！")
    else:
        print("合并失败，请检查错误信息。")