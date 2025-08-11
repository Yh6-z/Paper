# Please install OpenAI SDK first: `pip3 install openai`

import pandas as pd
from openai import OpenAI
import json
import os
import time
import re

# 您的DeepSeek API密钥（从DeepSeek平台获取）
API_KEY = "sk-0ae45e6d369a447787da2e1f0ca82f91" 

# 创建DeepSeek客户端（兼容OpenAI格式）
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

# 定义6个类别
CATEGORIES = [
    'Social Issues',
    'Politics & International',
    'Culture & Lifestyle',
    'Environment & Safety',
    'Science & Technology',
    'Economy & Business'
]

def test_api():
    """
    测试API连通性，等价于原curl命令，但使用DeepSeek。
    如果测试通过，返回True；否则返回False。
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Testing. Just say hi and hello world and nothing else."}
            ],
            stream=False,
            temperature=1.0
        )
        reply = response.choices[0].message.content.strip().lower()
        print("API Test Response:", reply)
        expected = "hi and hello world"
        if expected in reply:
            print("API test passed!")
            return True
        else:
            print("API test failed: Unexpected response.")
            return False
    except Exception as e:
        print(f"API test error: {e}")
        return False

def classify_headline_via_api(headline):
    """
    使用DeepSeek API对headline进行分类。
    返回一个dict，key为类别，value为1或0。
    """
    prompt = f"""
Classify the following headline into the provided categories. 
For each category, return 1 if the headline belongs to it, 0 otherwise. 
Ensure the headline belongs to at least one category. 
Output ONLY a JSON object wrapped in triple backticks (```json\n...\n```), with no additional text.

Categories: {', '.join(CATEGORIES)}

Headline: {headline}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies text accurately and outputs only JSON as instructed."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=1.0,  # 保持温度为1.0
            max_tokens=1024
        )
        generated_text = response.choices[0].message.content
        print(f"Raw API response for '{headline}': {generated_text}")
        
        # 清理Markdown代码块
        json_content = generated_text.strip()
        json_content = re.sub(r'^```json\s*', '', json_content, flags=re.MULTILINE)
        json_content = re.sub(r'\s*```$', '', json_content, flags=re.MULTILINE)
        json_content = json_content.strip()
        
        # 解析JSON
        classification = json.loads(json_content)
        
        # 确保所有类别存在，并至少有一个1
        for cat in CATEGORIES:
            if cat not in classification:
                classification[cat] = 0
        if sum(classification.values()) == 0:
            print(f"Warning: No category assigned for '{headline}', forcing default.")
            classification[CATEGORIES[0]] = 1
        
        return classification
    
    except Exception as e:
        print(f"API error for headline '{headline}': {e}")
        return {cat: 0 for cat in CATEGORIES}

# 主函数
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("当前脚本的文件夹位置为: ", script_dir)
    parent_dir = os.path.dirname(script_dir)
    print("父目录文件夹位置为: ", parent_dir)
    
    input_file = os.path.join(parent_dir, "Merge_Combo_RESULT", "full_prior_predictions.csv")
    output_file = os.path.join(parent_dir, "Classification_ClassAccuracy", "classified_dataset.csv")
    content_output = os.path.join(parent_dir, "Classification_ClassAccuracy", "content_level_categories.csv")
    
    # 先运行API测试
    if not test_api():
        print("Stopping: API test failed. Please check your API key or endpoint.")
        return
    
    # 读取CSV
    df = pd.read_csv(input_file)
    
    # 添加类别列
    for cat in CATEGORIES:
        df[cat] = 0
    
    # 对每个headline调用API分类
    for idx, row in df.iterrows():
        print(f"Classifying headline: {row['headline']}")
        cls = classify_headline_via_api(row['headline'])
        for cat, val in cls.items():
            if cat in df.columns:
                df.at[idx, cat] = val
        time.sleep(1)  # 避免速率限制
    
    # 保存完整数据集
    df.to_csv(output_file, index=False)
    print(f"Saved classified dataset to {output_file}")
    
    # 内容级别聚合（union: max per test_id）
    content_df = df.groupby('test_id').agg({cat: 'max' for cat in CATEGORIES}).reset_index()
    content_df.to_csv(content_output, index=False)
    print(f"Saved content-level categories to {content_output}")

if __name__ == "__main__":
    main()