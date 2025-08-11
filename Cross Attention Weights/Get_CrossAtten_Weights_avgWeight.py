import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 控制选择的属性
PROPERTY = "Concreteness"  # "Concreteness", "Sentiment", "Length", "Numeric", "Suspense", "Pronoun"

# 自定义数据集类
class CTRDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.headlines = df['headline'].astype(str).tolist()
        # 动态选择标签列
        label_column = PROPERTY + '_Label'
        # 确保标签列不为空
        self.property_labels = df[label_column].fillna('default').astype(str).tolist()
        self.ctrs = df['CTR'].astype(float).values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        headline = self.headlines[idx]
        property_label = self.property_labels[idx]
        
        # 如果标签太短，添加默认描述
        if len(property_label.strip()) < 3:
            property_label = f"{PROPERTY}: {property_label}"
        
        ctr = self.ctrs[idx]
        
        encoding_headline = self.tokenizer(
            headline,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding_property = self.tokenizer(
            property_label,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_headline': encoding_headline['input_ids'].squeeze(),
            'attention_mask_headline': encoding_headline['attention_mask'].squeeze(),
            'input_ids_property': encoding_property['input_ids'].squeeze(),
            'attention_mask_property': encoding_property['attention_mask'].squeeze(),
            'ctr': torch.tensor(ctr, dtype=torch.float32)
        }

class BertCrossAttentionRegressor(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=768, num_attention_heads=12, dropout_prob=0.1):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=config)
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none"
        )
        self.bert = get_peft_model(self.bert, lora_config)
        
        self.self_attention = torch.nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=dropout_prob)  # 未使用，但保留
        self.cross_attention_property = torch.nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=dropout_prob)
        
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.regressor = torch.nn.Linear(hidden_size, 1)
        # 输出注意力头的数量
        print(f"Self Attention heads: {self.self_attention.num_heads}")
        print(f"Cross Attention property heads: {self.cross_attention_property.num_heads}")
    
    def forward(self, input_ids_headline, attention_mask_headline, 
                input_ids_property, attention_mask_property, labels=None):
        outputs_headline = self.bert(input_ids_headline, attention_mask=attention_mask_headline)
        hidden_headline = outputs_headline.last_hidden_state
        
        outputs_property = self.bert(input_ids_property, attention_mask=attention_mask_property)
        hidden_property = outputs_property.last_hidden_state
        
        hidden_headline = hidden_headline.transpose(0, 1)
        hidden_property = hidden_property.transpose(0, 1)
        
        cls_headline = hidden_headline[0:1, :, :] 
        
        cross_attn_property, attn_weights = self.cross_attention_property(
            cls_headline, hidden_property, hidden_property,
            # average_attn_weights=False  # False 表示获取每个头的独立权重
            average_attn_weights=True # True 表示要获取12个头的平均权重
        )
        cross_attn_property = cross_attn_property.transpose(0, 1) 
        
        attn_output = self.norm(cross_attn_property + hidden_headline.transpose(0, 1)[:, :1, :]) 
        
        # 回归头
        cls_output = attn_output[:, 0, :]  
        cls_output = self.dropout(cls_output)
        preds = self.regressor(cls_output).squeeze(-1)  
        
        output = {"preds": preds}
        output["attn_weights"] = attn_weights  
        print("Attn_weights 的形状结构为: ", attn_weights.shape) 
        
        if labels is not None:
            loss_fct = nn.MSELoss()
            output["loss"] = loss_fct(preds, labels)
        
        return output
    

# 测试函数，提取多头 CLS 注意力权重
def test_model_with_weights(model, test_loader, test_df, device, tokenizer):
    model.eval()
    predictions = []
    all_cls_attn_weights = []  # 存储每个样本的权重，
    all_attn_masks_property = []  # 保存 Property_Label 的 attention_mask
    headlines = test_df['headline'].tolist()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Extracting Multi-Head CLS Attn Weights")):
            input_ids_headline = batch['input_ids_headline'].to(device)
            attention_mask_headline = batch['attention_mask_headline'].to(device)
            input_ids_property = batch['input_ids_property'].to(device)
            attention_mask_property = batch['attention_mask_property'].to(device)
            
            output = model(
                input_ids_headline, attention_mask_headline,
                input_ids_property, attention_mask_property
            )
            print(f"Batch {batch_idx}, attn_weights shape: {output['attn_weights'].shape}")
            predictions.extend(output['preds'].cpu().numpy())
            attn_weights = output['attn_weights'].cpu().numpy()  # [batch_size * num_heads, 1, src_len]
            
            batch_size = input_ids_headline.size(0)
            num_heads = model.cross_attention_property.num_heads  # 12
            src_len = attn_weights.shape[2]  # 128
            
            
            # 按样本处理
            for i in range(batch_size):
                # 获取当前样本的注意力权重
                cls_weights = attn_weights[i:i+1]  # [1, num_heads, 1, src_len]
                
                # 获取有效 token 的索引
                property_mask = attention_mask_property[i].cpu().numpy()
                valid_indices = np.where(property_mask == 1)[0]  # 有效 token 索引，例如 [0,1,2,3,4,5,6]
                
                # 切片有效 token 的权重
                valid_cls_weights = cls_weights[:, :, valid_indices]  # [1,  1, valid_len]

                # valid_cls_weights = valid_cls_weights.squeeze(2)  # [1, 1, valid_len]
                
                # 保存到列表
                all_cls_attn_weights.append(valid_cls_weights)
            
            all_attn_masks_property.extend(attention_mask_property.cpu().numpy())  # 保存 attention_mask
            torch.cuda.empty_cache()
    
    test_df = test_df.copy()
    test_df['pred_CTR'] = predictions
    test_df['cls_attn_weights'] = all_cls_attn_weights  # [1, num_heads, valid_len]
    test_df['attn_mask_property'] = all_attn_masks_property 
    return test_df

def main():
    # 数据集和模型路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    test_path = os.path.join(parent_dir, "Cross Attention", "Data process", "output_data", "Test_text_sum.csv")
    New_headline_path = os.path.join(parent_dir, "Cross Attention", "Data process", "output_data", "New_headline_text_sum.csv")

    model_result_dir = os.path.join(parent_dir, "Cross Attention", "Code_BertCrossAttentionRegressor", f"{PROPERTY} RESULT")
    model_save_path = os.path.join(model_result_dir, f"best_model_for_{PROPERTY.lower()}_Seed.pth")

    test_save_path = os.path.join(script_dir, f"{PROPERTY}",f"Test_Cro_Atten_{PROPERTY.lower()}_pred_seed_avgweights.csv")
    new_headline_save_path = os.path.join(script_dir, f"{PROPERTY}",f"New_Headline_Cro_Atten_{PROPERTY.lower()}_pred_seed_avgweights.csv")
    # 加载测试数据
    try:
        test_df = pd.read_csv(test_path, encoding="utf-8", encoding_errors="ignore")
        New_headline_df = pd.read_csv(New_headline_path, encoding="utf-8", encoding_errors="ignore")

        # 重命名 sentiment 和 length 列
        test_df.rename(columns={'sentiment': 'Sentiment_Label', 'length': 'Length_Label'}, inplace=True)
        New_headline_df.rename(columns={'sentiment': 'Sentiment_Label', 'length': 'Length_Label'}, inplace=True)

        test_df = test_df.dropna(subset=['CTR']).reset_index(drop=True)
        label_column = PROPERTY + '_Label'
        if label_column not in test_df.columns:
            print(f"错误：测试数据集中缺少 '{label_column}' 列")
            return
        test_df = test_df.drop(columns='pred_CTR', errors='ignore')
    except FileNotFoundError as e:
        print(f"错误：无法找到文件 {test_path}")
        return
    
    # 加载tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertCrossAttentionRegressor().to(device)
    
    # 加载保存的模型
    try:
        state_dict = torch.load(model_save_path, map_location=device) 
        # 重命名旧模型中的键以匹配当前模型
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # 处理所有可能的旧键名
            for old_prefix in ['cross_attention_concreteness', 'cross_attention_suspense', 
                          'cross_attention_sentiment', 'cross_attention_length', 
                          'cross_attention_numeric', 'cross_attention_pronoun']:
                if old_prefix in key:
                    new_key = key.replace(old_prefix, 'cross_attention_property')
                    break
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=True)
        print(f"已加载模型：{model_save_path}")
        print(f"加载模型的多头注意力头数 : {model.self_attention.num_heads}")
        print(f"加载模型的交叉多头注意力头数 : {model.cross_attention_property.num_heads}")
    except FileNotFoundError:
        print(f"错误：未找到模型文件 {model_save_path}")
        return
    except RuntimeError as e:
        print(f"错误：加载模型失败 - {e}")
        return
    
    # 创建测试数据集和DataLoader
    #test_dataset = CTRDataset(test_df, tokenizer)
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    New_headline_dataset = CTRDataset(New_headline_df, tokenizer)
    New_headline_loader = DataLoader(New_headline_dataset, batch_size=1, shuffle=False)
    # 运行测试并提取CLS注意力权重
    #test_df_with_preds = test_model_with_weights(model, test_loader, test_df, device, tokenizer)
    New_headline_df_with_preds = test_model_with_weights(model, New_headline_loader, New_headline_df, device, tokenizer)
    # 保存预测结果
    #test_df_with_preds.to_csv(test_save_path, index=False)
    New_headline_df_with_preds.to_csv(new_headline_save_path, index=False)
    print(f"测试数据预测结果已保存到 {test_save_path}")

if __name__ == "__main__":
    main()