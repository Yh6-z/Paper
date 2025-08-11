### 将从liwc中提取的数值特征转换为文本标签
### 该脚本读取CSV文件，处理数值特征，并将其转换为文本标签
### 最终结果保存为新的CSV文件

import pandas as pd
import logging
import os
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_label_conversion.log'),
        logging.StreamHandler()
    ]
)

def convert_to_label(df, output_csv):
    """将数值特征转换为文本标签，并保存结果"""
    try:
        # 检查所需列
        required_columns = ['headline', 'Concreteness_Score', 'Numeric_Structure_Score', 
                           'Suspense_Question_Score', 'Pronoun_Usage']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 处理空值
        df['headline'] = df['headline'].fillna('').astype(str).str.strip()
        # numeric_cols = ['Concreteness_Score', 'Numeric_Structure_Score', 'Suspense_Question_Score', 'Sensory_Perception_Score']
        numeric_cols = ['Concreteness_Score', 'Numeric_Structure_Score', 'Suspense_Question_Score']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().sum() > 0:
                logging.warning(f"{col} contains {df[col].isna().sum()} NaN values; will be filled with 0.0")
                df[col] = df[col].fillna(0.0)
        df['Pronoun_Usage'] = df['Pronoun_Usage'].fillna('None').astype(str)
        
        # 输出最小值和最大值，并定义动态bins
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val == max_val:
                logging.warning(f"{col}: All values are {min_val}; assigning default label")
            else:
                third = (max_val - min_val) / 3
                logging.info(f"{col}: Min = {min_val}, Max = {max_val}")
                logging.info(f"Intervals: [{min_val}, {min_val + third:.3f}], ({min_val + third:.3f}, {min_val + 2*third:.3f}], ({min_val + 2*third:.3f}, {max_val}]")
        
        # 定义动态bins函数
        def get_bins(col):
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val == max_val:
                return [min_val - 1, min_val, max_val + 1]  # 虚拟bins以处理单一值
            return [min_val, min_val + (max_val - min_val)/3, 
                    min_val + 2*(max_val - min_val)/3, max_val]
        
        # Concreteness_Score 转换
        df['Concreteness_Label'] = pd.cut(
            df['Concreteness_Score'],
            bins=get_bins('Concreteness_Score'),
            labels=['The headline uses abstract language', 
                    'The headline uses semi-abstract language', 
                    'The headline uses specific language'],
            include_lowest=True
        )
        if df['Concreteness_Score'].min() == df['Concreteness_Score'].max():
            df['Concreteness_Label'] = 'The headline uses abstract language'  # 默认低标签，或根据需要调整
        
        
        # Numeric_Structure_Score 转换
        df['Numeric_Label'] = pd.cut(
            df['Numeric_Structure_Score'],
            bins=get_bins('Numeric_Structure_Score'),
            labels=['The headline contains no numerical content', 
                    'The headline contains some numerical content.', 
                    'The headline contains a lot of numerical content.'],
            include_lowest=True
        )
        if df['Numeric_Structure_Score'].min() == df['Numeric_Structure_Score'].max():
            df['Numeric_Label'] = 'The headline contains no numerical content'
        
        # Suspense_Question_Score 转换
        df['Suspense_Label'] = pd.cut(
            df['Suspense_Question_Score'],
            bins=get_bins('Suspense_Question_Score'),
            labels=['The headline do not use suspense', 
                    'The headline uses suspense', 
                    'The headline uses a lot of suspense'],
            include_lowest=True
        )
        if df['Suspense_Question_Score'].min() == df['Suspense_Question_Score'].max():
            df['Suspense_Label'] = 'This headline do not use suspense'
        
        # Pronoun_Usage 转换
        pronoun_map = {
            'None': 'The headline do not use pronouns',
            'First': 'The headline uses 1nd-person pronouns',
            'Second': 'The headline uses 2nd-person pronouns',
            'Third': 'The headline uses 3nd-person pronouns',
            'Mixed': 'The headline uses mixed pronouns'
        }
        df['Pronoun_Label'] = df['Pronoun_Usage'].map(pronoun_map)
        if df['Pronoun_Label'].isna().sum() > 0:
            logging.warning(f"Pronoun_Label contains {df['Pronoun_Label'].isna().sum()} NaN values; invalid Pronoun_Usage values detected")
            df['Pronoun_Label'] = df['Pronoun_Label'].fillna('This headline do not use pronouns')
        
        # 保存结果
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error in convert_to_label: {e}")
        raise



if __name__ == "__main__":
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # # 输入和输出文件
    # timestamp = "20250714_1339"  
    # input_file_train = os.path.join(script_dir, f"train_liwc_{timestamp}.csv")
    # input_file_test = os.path.join(script_dir, f"test_liwc_{timestamp}.csv")

    # output_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # output_file_train = os.path.join(script_dir, f"train_liwc_with_labels_{output_timestamp}.csv")
    # output_file_test = os.path.join(script_dir, f"test_liwc_with_labels_{output_timestamp}.csv")
    
    # # 处理训练和测试集
    # for input_file, output_file in [(input_file_train, output_file_train), (input_file_test, output_file_test)]:
    #     if os.path.exists(input_file):
    #         logging.info(f"Processing {input_file}")
    #         df = pd.read_csv(input_file)
    #         convert_to_label(df, output_file)
    #     else:
    #         logging.warning(f"Input file {input_file} does not exist")



    # 对另外设置的几个标题进行处理
    input_file = os.path.join(script_dir, f"Headline_liwc.csv")

    output_file = os.path.join(script_dir, f"Headline_liwc_with_labels.csv")
    
    
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        convert_to_label(df, output_file)
    else:
        logging.warning(f"Input file {input_file} does not exist")