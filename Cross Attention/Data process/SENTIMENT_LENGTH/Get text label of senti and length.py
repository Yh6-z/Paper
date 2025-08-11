### 这个脚本用来得到长度和情感的文本数据

import pandas as pd
import numpy as np
import os 


def assign_sentiment(row):
    probs = [row['prob_negative'], row['prob_neutral'], row['prob_positive']]
    max_idx = np.argmax(probs)
    if max_idx == 0:
        return "The headline has a negtive tone"
    elif max_idx == 1:
        return "The headline has a neutral tone"
    else:
        return "The headline has a positive tone"

def compute_length(headline):
    """
    从 headline 计算词数（以空格分割），返回整数。
    处理空值或非字符串情况，返回 0。
    """
    if pd.isna(headline) or not isinstance(headline, str) or headline.strip() == "":
        return 0
    return len(headline.split())



# 示例使用
if __name__ == "__main__":
    ## 得到sentiment score的数据集
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)       ### e:\VS code project\Cross Attention\Data process\SENTIMENT_LENGTH label

    train_pred_path = os.path.join(script_dir, "Train_sentiment_20250625_1351.csv")
    test_pred_path = os.path.join(script_dir, "Test_sentiment_20250625_1351.csv")

    New_headline_path = os.path.join(script_dir, "New_headline_sentiment.csv")
    # 验证文件存在
    for path in [train_pred_path, test_pred_path]:
        if not os.path.exists(path):
            print(f"错误：文件 {path} 不存在")
            exit(1)

    # 加载数据
    try:
        train_df = pd.read_csv(train_pred_path)
        test_df = pd.read_csv(test_pred_path)
        headline_df = pd.read_csv(New_headline_path)
    except FileNotFoundError as e:
        print(f"错误：无法找到文件 {e.filename}")
        exit(1)

    # 验证必要列
    # required_cols = ['test_id', 'headline', 'CTR', 'prob_positive', 'prob_negative', 'prob_neutral']
    required_cols = ['test_id', 'prob_positive', 'prob_negative', 'prob_neutral']
    for df, name in [(train_df, "Train"), (test_df, "Test")]:
        if not all(col in df.columns for col in required_cols):
            print(f"错误：{name} 数据集缺少必要列，现有列：{df.columns.tolist()}")
            exit(1)

    # 打印数据集信息
    print(f"\n训练集总行数: {len(train_df)}, 唯一 test_id 数量: {train_df['test_id'].nunique()}")
    print(f"测试集总行数: {len(test_df)}, 唯一 test_id 数量: {test_df['test_id'].nunique()}")
    print(f"训练集 prob_negative NaN 数量: {train_df['prob_negative'].isna().sum()}")
    print(f"训练集 prob_neutral NaN 数量: {train_df['prob_neutral'].isna().sum()}")
    print(f"训练集 prob_positive NaN 数量: {train_df['prob_positive'].isna().sum()}")
    print(f"测试集 prob_negative NaN 数量: {test_df['prob_negative'].isna().sum()}")
    print(f"测试集 prob_neutral NaN 数量: {test_df['prob_neutral'].isna().sum()}")
    print(f"测试集 prob_positive NaN 数量: {test_df['prob_positive'].isna().sum()}")

    # 检查并删除 sentiment 列（如果存在）
    train_df = train_df.drop(columns='sentiment', errors='ignore')
    test_df = test_df.drop(columns='sentiment', errors='ignore')
    print("\n已删除 sentiment 列（若存在）")

    # 转换情感数据为文本数据
    print("\n生成 sentiment 列...")
    train_df['sentiment'] = train_df.apply(assign_sentiment, axis=1)
    test_df['sentiment'] = test_df.apply(assign_sentiment, axis=1)
    # 对New headline进行文本转换
    headline_df['sentiment'] = headline_df.apply(assign_sentiment, axis=1)

    # 打印 sentiment 列 NaN 统计
    print(f"训练集 sentiment NaN 数量: {train_df['sentiment'].isna().sum()}")
    print(f"测试集 sentiment NaN 数量: {test_df['sentiment'].isna().sum()}")

    # 计算 length 列
    print("\n计算 train_df 的 length 列...")
    train_df['length'] = train_df['headline'].apply(compute_length)
    train_df["length_num"] = train_df['length']
    train_df['length'] = train_df['length'].astype(int)
    train_df['length'] = train_df['length'].apply(lambda x: f"The headline contains {x} words")

    print("计算 test_df 的 length 列...")
    test_df['length'] = test_df['headline'].apply(compute_length)
    test_df["length_num"] = test_df['length']
    test_df['length'] = test_df['length'].astype(int)
    test_df['length'] = test_df['length'].apply(lambda x: f"The headline contains {x} words")

    print("计算 headline_df 的 length 列...")
    headline_df['length'] = headline_df['headline'].apply(compute_length)
    headline_df["length_num"] = headline_df['length']
    headline_df['length'] = headline_df['length'].astype(int)
    headline_df['length'] = headline_df['length'].apply(lambda x: f"The headline contains {x} words")

    # 验证 sentiment 和 length 列
    print("\n验证 train_df sentiment 和 length 列前几行：")
    print(train_df[['test_id', 'headline', 'sentiment', 'length']].head())
    print("\n验证 test_df sentiment 和 length 列前几行：")
    print(test_df[['test_id', 'headline', 'sentiment', 'length']].head())

    # 定义输出文件路径
    output_dir = os.path.join(script_dir, "Merge RESULT")
    os.makedirs(output_dir, exist_ok=True)
    output_path_train = os.path.join(output_dir, "Train_updated.csv")
    output_path_test = os.path.join(output_dir, "Test_updated.csv")

    output_path_new_headline = os.path.join(output_dir, "New_headline_sentiemnt_text.csv")
    # 保存更新后的数据集
    # train_df.to_csv(output_path_train, index=False)
    # test_df.to_csv(output_path_test, index=False)
    train_df.to_csv(output_path_train, index=False, encoding='utf-8')
    test_df.to_csv(output_path_test, index=False, encoding='utf-8')
    headline_df.to_csv(output_path_new_headline, index=False, encoding='utf-8')
    print(f"\n更新后的训练集文件已保存到 {output_path_train}")
    print(f"更新后的测试集文件已保存到 {output_path_test}")

    # 验证保存结果
    train_df_saved = pd.read_csv(output_path_train)
    test_df_saved = pd.read_csv(output_path_test)
    print("\n验证保存结果: ")
    print(train_df_saved[['test_id', 'headline', 'sentiment', 'length']].head())
    print(test_df_saved[['test_id', 'headline', 'sentiment', 'length']].head())
    
    
    


