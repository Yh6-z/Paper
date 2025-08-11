### 这个代码脚本的前提是使用 LIWC-22 对现有数据集进行处理，得到100多个特征后，摘取出一部分进行具体性、数字结构化、悬念提问等特征的分析
### 该脚本使用 BERT 分词器进行文本预处理，并计算具体性、数字结构化、悬念提问等特征得分
### 具体的得分是通过相应特征使用的词语出现的次数占总长度的比率进行度量的。

import pandas as pd
from transformers import BertTokenizer
import re
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.tag import pos_tag
import logging
import os
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('liwc_analysis.log'),
        logging.StreamHandler()
    ]
)

# 下载 NLTK 资源
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# 加载 BERT 分词器
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logging.info("BERT tokenizer loaded successfully")
except Exception as e:
    logging.error(f"Failed to load BERT tokenizer: {e}")
    raise

# 自定义 Suspense 词典
LIWC_SUSPENSE = {'unbelievable', 'guess', 'shocking', 'surprise', 'incredible', 'astonishing', 'unexpected'}

def preprocess_text(text):
    """使用 BERT 分词器预处理文本"""
    if not isinstance(text, str) or text.strip() == '':
        logging.warning(f"Empty or invalid text: {text}")
        return [], []
    try:
        tokens = tokenizer.tokenize(text.lower())       
        words = []      
        current_word = ""       

        for token in tokens:
            if token.startswith('##'):          
                current_word += token[2:]       
            else:
                if current_word:                
                    words.append(current_word)
                current_word = token
        if current_word:
            words.append(current_word)
        tagged = pos_tag(words)     
        return words, tagged
    except Exception as e:
        logging.error(f"Error in preprocessing text '{text}': {e}")
        return [], []

def concreteness_score(row):
    """计算具体性得分(0-1)，基于 LIWC-22 输出和 BERT 词性
    具体性得分越接近1, 表示标题越具体, 越接近0表示越抽象"""
    try:
        headline = row.get('headline', '')      
        wc_liwc = row.get('WC', 1)          

        if wc_liwc == 0 or not headline.strip():
            logging.debug(f"Zero word count or empty headline: {headline}")
            return 0.0
        
        # 预处理以获取 local_wc
        words, tagged = preprocess_text(headline)
        local_wc = len(words) if words else wc_liwc  # 使用 BERT 词数一致
        
        # 具体性（LIWC 百分比 / 100 获取比例）
        concrete_categories = ['article', 'space', 'time', 'motion', 'home', 'Physical', 'verb']
        concrete_score = sum(row.get(cat, 0) / 100 for cat in concrete_categories)
        
        # BERT 名词（使用 local_wc）
        noun_count = sum(1 for _, tag in tagged if tag in ['NN', 'NNS'])
        noun_score = noun_count / local_wc if local_wc > 0 else 0.0
        
        # 抽象性
        abstract_categories = ['Affect', 'cogproc', 'emo_pos', 'emo_neg', 'adj']
        abstract_score = sum(row.get(cat, 0) / 100 for cat in abstract_categories)
        analytic_score = row.get('Analytic', 0) / 100       
        
        # 综合得分
        concrete_total = (concrete_score + noun_score) / 2          
        abstract_total = (abstract_score + analytic_score) / 2      
        score = (concrete_total - abstract_total + 1) / 2           
        score = max(0.0, min(score, 1.0))
        
        logging.debug(
            f"Headline: {headline}, WC_LIWC: {wc_liwc}, local_wc: {local_wc}, "
            f"noun: {noun_score}, concrete: {concrete_score}, "
            f"abstract: {abstract_score}, analytic: {analytic_score}, score: {score}"
        )
        return score
    except Exception as e:
        logging.error(f"Error in concreteness_score for headline '{row.get('headline', 'N/A')}': {e}")
        return 0.0
    

    
def numeric_structure_score(row, headline):
    """计算数字与结构化信息得分（0-1）"""
    try:
        wc_liwc = row.get('WC', 1)      
        if wc_liwc == 0:
            return 0.0
        
        # LIWC 部分（/100）
        number_score = row.get('number', 0) / 100
        quantity_score = row.get('quantity', 0) / 100
        
        # 词级阿拉伯数字计数（匹配完整数字词，如 '123' 计1）
        arabic_digit_count = len(re.findall(r'\b\d+\b', str(headline).lower()))  
        # 使用 wc_liwc 作为分母（无预处理需求）
        arabic_score = arabic_digit_count / wc_liwc if wc_liwc > 0 else 0.0
        
        score = (number_score + quantity_score + arabic_score) / 3   
        score = max(0.0, min(score, 1.0))  # 严格归一化
        
        logging.debug(
            f"Headline: {headline}, WC: {wc_liwc}, "
            f"number: {number_score}, quantity: {quantity_score}, arabic: {arabic_score}, score: {score}"
        )
        return score
    except Exception as e:
        logging.error(f"Error in numeric_structure_score for '{headline}': {e}")
        return 0.0

def suspense_question_score(row, headline):
    """计算提问与悬念结构得分（0-1）"""
    if not isinstance(headline, str) or headline.strip() == '':
        return 0.0
    try:
        words, tagged = preprocess_text(headline)
        wc_liwc = row.get('WC', 1)
        if wc_liwc == 0:
            return 0.0
        local_wc = len(words) if words else wc_liwc  # 一致性
        
        # LIWC 部分（/100）
        qmark_score = row.get('QMark', 0) / 100         
        affect_score = row.get('Affect', 0) / 100       
        curiosity_score = row.get('curiosity', 0) / 100 
        discrep_score = row.get('discrep', 0) / 100     
        tentat_score = row.get('tentat', 0) / 100       

        # 自定义悬念（使用 local_wc）
        suspense_count = sum(1 for w in words if w in LIWC_SUSPENSE) 
        suspense_score = suspense_count / local_wc if local_wc > 0 else 0.0
        
        # 前向指代（使用 local_wc）
        forward_ref_count = sum(1 for word, tag in tagged if tag in ['DT', 'PRP'] and word in {'this', 'that', 'he', 'she', 'it'})  
        forward_ref_score = forward_ref_count / local_wc if local_wc > 0 else 0.0
        
        # 加权（总权重=1.0：0.25+0.2+0.2+0.15+0.2=1.0，原0.2*forward）
        score = (0.25 * qmark_score + 0.2 * affect_score + 0.2 * curiosity_score +
                 0.15 * ((discrep_score + tentat_score) / 2) + 0.2 * suspense_score +
                 0.2 * forward_ref_score)
        score = max(0.0, min(score, 1.0))
        
        logging.debug(
            f"Headline: {headline}, WC_LIWC: {wc_liwc}, local_wc: {local_wc}, "
            f"qmark: {qmark_score}, affect: {affect_score}, curiosity: {curiosity_score}, "
            f"discrep: {discrep_score}, tentat: {tentat_score}, suspense: {suspense_score}, "
            f"forward_ref: {forward_ref_score}, score: {score}"
        )
        return score
    except Exception as e:
        logging.error(f"Error in suspense_question_score for '{headline}': {e}")
        return 0.0

# pronoun_usage 函数未修改，保持原样
def pronoun_usage(row):
    """检测人称代词使用类别"""
    try:
        first_sing = row.get('i', 0)
        first_plur = row.get('we', 0) 
        second = row.get('you', 0)
        third_sing = row.get('shehe', 0)
        third_plur = row.get('they', 0)
        result = (
            'None' if first_sing + first_plur + second + third_sing + third_plur == 0 else
            'First' if (first_sing > 0 or first_plur > 0) and second == 0 and third_sing == 0 and third_plur == 0 else
            'Second' if second > 0 and first_sing == 0 and first_plur == 0 and third_sing == 0 and third_plur == 0 else
            'Third' if (third_sing > 0 or third_plur > 0) and first_sing == 0 and first_plur == 0 and second == 0 else
            'Mixed'
        )

        logging.debug(
            f"Headline: {row.get('headline', 'N/A')}, "
            f"i: {first_sing}, you: {second}, shehe: {third_sing}, they: {third_plur}, "
            f"result: {result}"
        )
        return result
    except Exception as e:
        logging.error(f"Error in pronoun_usage: {e}")
        return 'None'


def analyze_liwc_output(liwc_csv, output_csv):
    """主函数：在 LIWC-22 数据集上计算特征并保存结果"""
    try:
        # 读取数据集
        liwc_df = pd.read_csv(liwc_csv)
        logging.info(f"Loaded LIWC dataset with {len(liwc_df)} rows and {len(liwc_df.columns)} columns")
        
        # 处理空值和标准化
        if 'headline' in liwc_df.columns:
            liwc_df['headline'] = liwc_df['headline'].fillna('').astype(str).str.strip().str.lower().str.encode('utf-8').str.decode('utf-8')
        else:
            logging.warning("No 'headline' column found; assuming no title column is needed")
        
        # 清洗 LIWC 数据
        liwc_columns = ['WC', 'article', 'space', 'time', 'motion', 'home', 'Physical', 'verb',
                        'Affect', 'cogproc', 'emo_pos', 'emo_neg', 'adj', 'Analytic',
                        'number', 'quantity', 'QMark', 'curiosity', 'discrep', 'tentat',
                        'i','we', 'you', 'shehe', 'they','visual','auditory','feeling', 'Perception']
        existing_columns = [col for col in liwc_columns if col in liwc_df.columns]
        missing_columns = [col for col in liwc_columns if col not in liwc_df.columns]
        if missing_columns:
            logging.warning(f"Missing LIWC columns: {missing_columns}")
        
        for col in existing_columns:
            liwc_df[col] = pd.to_numeric(liwc_df[col], errors='coerce').fillna(0)
        
        # 检查空标题比例
        if 'headline' in liwc_df.columns:
            empty_titles = (liwc_df['headline'] == '') | (liwc_df['headline'].isna())
            logging.info(f"Number of empty titles: {empty_titles.sum()}")
        
        # 检查 LIWC 数据非零值
        non_zero_counts = {col: (liwc_df[col] > 0).sum() for col in existing_columns}
        logging.info(f"Non-zero counts: {non_zero_counts}")
        
        # 记录 LIWC 统计
        liwc_stats = liwc_df[existing_columns].describe()
        liwc_stats.to_csv('liwc_stats.csv')
        logging.info(f"LIWC data statistics saved to liwc_stats.csv")
        logging.info(f"LIWC stats summary:\n{liwc_stats}")
        
        # 计算特征
        liwc_df['Concreteness_Score'] = liwc_df.apply(
            lambda row: concreteness_score(row) if 'headline' in row else concreteness_score({**row, 'headline': ''}), axis=1
        )
        # liwc_df['Sensory_Perception_Score'] = liwc_df.apply(
        #     lambda row: sensory_perception_score(row), axis=1
        # )
        liwc_df['Numeric_Structure_Score'] = liwc_df.apply(
            lambda row: numeric_structure_score(row, row['headline'] if 'headline' in row else ''), axis=1
        )
        liwc_df['Suspense_Question_Score'] = liwc_df.apply(
            lambda row: suspense_question_score(row, row['headline'] if 'headline' in row else ''), axis=1
        )
        liwc_df['Pronoun_Usage'] = liwc_df.apply(pronoun_usage, axis=1)
        
        # 检查特征计算结果
        logging.info(f"Feature stats:\n{liwc_df[['Concreteness_Score', 'Numeric_Structure_Score', 'Suspense_Question_Score']].describe()}")
        
        need_columns = ["test_id",'headline', "CTR", 'Concreteness_Score', 'Numeric_Structure_Score',
            'Suspense_Question_Score', 'Pronoun_Usage'
        ]
        # need_columns = ['headline',  'Concreteness_Score', 'Numeric_Structure_Score',
        #     'Suspense_Question_Score', 'Pronoun_Usage'
        # ]
        liwc_df = liwc_df[need_columns]
        # 保存结果
        liwc_df.to_csv(output_csv, index=False)
        logging.info(f"Analysis complete. Results saved to {output_csv}")
    except Exception as e:
        logging.error(f"Error in analyze_liwc_output: {e}")
        raise

if __name__ == "__main__":
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入文件

    # input_file_train = os.path.join(script_dir, f"LIWC-22 Results - LoRA_CTR_train - LIWC Analysis.csv")
    # input_file_test = os.path.join(script_dir, f"LIWC-22 Results - LoRA_CTR_test - LIWC Analysis.csv")
    
    # # 输出文件
    # output_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # output_file_train = os.path.join(script_dir, f"Train_liwc_{output_timestamp}.csv")
    # output_file_test = os.path.join(script_dir, f"Test_liwc_{output_timestamp}.csv")
    # logging.info(f"Output files will be saved to: {output_file_train}, {output_file_test}")
    
    # # 分析训练和测试集
    # analyze_liwc_output(input_file_train, output_file_train)
    # analyze_liwc_output(input_file_test, output_file_test)

    input_file = os.path.join(script_dir, f"LIWC-22 Results - New_headline - LIWC Analysis.csv")
    output_file = os.path.join(script_dir, f"Headline_liwc.csv")
    analyze_liwc_output(input_file, output_file)

