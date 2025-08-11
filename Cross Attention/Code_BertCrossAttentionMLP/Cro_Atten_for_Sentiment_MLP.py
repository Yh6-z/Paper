import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, set_seed
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import os

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)

# 自定义数据集类
class CTRDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.headlines = df['headline'].astype(str).tolist()
        self.sentiments = df['sentiment'].astype(str).tolist()
        self.ctrs = df['CTR'].astype(float).values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        headline = self.headlines[idx]
        sentiment = self.sentiments[idx]
        ctr = self.ctrs[idx]
        
        encoding_headline = self.tokenizer(
            headline,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding_sentiment = self.tokenizer(
            sentiment,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_headline': encoding_headline['input_ids'].squeeze(),
            'attention_mask_headline': encoding_headline['attention_mask'].squeeze(),
            'input_ids_sentiment': encoding_sentiment['input_ids'].squeeze(),
            'attention_mask_sentiment': encoding_sentiment['attention_mask'].squeeze(),
            'ctr': torch.tensor(ctr, dtype=torch.float32)
        }

# Cross Attention 模型
class BertCrossAttentionRegressor(nn.Module):
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
        
        self.self_attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=dropout_prob)
        self.cross_attention_sentiment = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=dropout_prob)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids_headline, attention_mask_headline, 
                input_ids_sentiment, attention_mask_sentiment, labels=None):
        outputs_headline = self.bert(input_ids_headline, attention_mask=attention_mask_headline)
        hidden_headline = outputs_headline.last_hidden_state
        
        outputs_sentiment = self.bert(input_ids_sentiment, attention_mask=attention_mask_sentiment)
        hidden_sentiment = outputs_sentiment.last_hidden_state
        
        hidden_headline = hidden_headline.transpose(0, 1)
        hidden_sentiment = hidden_sentiment.transpose(0, 1)
        
        cross_attn_sentiment, _ = self.cross_attention_sentiment(hidden_headline, hidden_sentiment, hidden_sentiment)
        cross_attn_sentiment = cross_attn_sentiment.transpose(0, 1)
        
        attn_output = cross_attn_sentiment
        attn_output = self.norm(attn_output + hidden_headline.transpose(0, 1))
        
        cls_output = attn_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        preds = self.regressor(cls_output).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(preds, labels)
        
        return {"loss": loss, "preds": preds} if loss is not None else preds

# 准确率评估函数
def evaluate_accuracy(df):
    if df.empty or "pred_CTR" not in df.columns:
        print("错误：数据集为空或缺少 pred_CTR 列，跳过准确率计算")
        return 0.0, 0.0, 0, 0, 0
    
    correct = 0
    random_correct = 0
    total = 0
    
    for test_id, group in df.groupby("test_id"):
        if len(group) == 0:
            continue
        
        max_ctr = group["CTR"].max()
        true_max_indices = group[group["CTR"] == max_ctr].index.tolist()
        
        pred_max_idx = group["pred_CTR"].idxmax()
        valid_indices = group.index.tolist()
        
        total += 1
        if pred_max_idx in true_max_indices:
            correct += 1
        
        random_idx = random.choice(valid_indices)
        if random_idx in true_max_indices:
            random_correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    random_accuracy = random_correct / total if total > 0 else 0.0
    print(f"准确率计算：正确 {correct}/{total}，准确率 {accuracy:.3f}")
    print(f"随机准确率：正确 {random_correct}/{total}，随机准确率 {random_accuracy:.3f}")
    return accuracy, random_accuracy, correct, random_correct, total

# 训练函数
def train_model(model, train_loader, train_df, test_loader, test_df, optimizer, device, epochs=3, model_save_path="best_model.pth"):
    model.train()
    best_test_acc = 0.0
    best_epoch = 0
    best_train_df = None 
    best_test_df = None
    for epoch in range(epochs):
        total_loss = 0
        epoch_preds = []
        with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                input_ids_headline = batch['input_ids_headline'].to(device)
                attention_mask_headline = batch['attention_mask_headline'].to(device)
                input_ids_sentiment = batch['input_ids_sentiment'].to(device)
                attention_mask_sentiment = batch['attention_mask_sentiment'].to(device)
                labels = batch['ctr'].to(device)
                
                outputs = model(
                    input_ids_headline, attention_mask_headline,
                    input_ids_sentiment, attention_mask_sentiment,
                    labels
                )
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_preds.extend(outputs['preds'].detach().cpu().numpy())
                torch.cuda.empty_cache()
                
                pbar.set_postfix({'loss': f"{loss.item():.3f}"})
        
        avg_loss = total_loss / len(train_loader)
        train_df_temp = train_df.copy()
        train_df_temp['pred_CTR'] = epoch_preds
        acc, _, _, _, _ = evaluate_accuracy(train_df_temp)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.3f}, Training Accuracy: {acc:.3f}")
        
        test_df_temp = test_model(model, test_loader, test_df, device)
        test_acc, _, _, _, _ = evaluate_accuracy(test_df_temp)
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_acc:.3f}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_train_df = train_df_temp.copy()
            best_test_df = test_df_temp.copy()
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved at epoch {best_epoch} with test accuracy: {best_test_acc:.3f}")
    
    print(f"Best model from epoch {best_epoch} with test accuracy: {best_test_acc:.3f} saved to {model_save_path}")
    
    return best_train_df, best_test_df

# 测试函数
def test_model(model, test_loader, test_df, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids_headline = batch['input_ids_headline'].to(device)
            attention_mask_headline = batch['attention_mask_headline'].to(device)
            input_ids_sentiment = batch['input_ids_sentiment'].to(device)
            attention_mask_sentiment = batch['attention_mask_sentiment'].to(device)
            
            preds = model(
                input_ids_headline, attention_mask_headline,
                input_ids_sentiment, attention_mask_sentiment
            )
            predictions.extend(preds.cpu().numpy())
            torch.cuda.empty_cache()
    
    test_df = test_df.copy()
    test_df['pred_CTR'] = predictions
    return test_df

# 主函数
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    train_path = os.path.join(parent_dir, "Data process", "SENTIMENT_LENGTH label", "Merge RESULT", f"Train_updated.csv")
    test_path = os.path.join(parent_dir, "Data process", "SENTIMENT_LENGTH label", "Merge RESULT", f"Test_updated.csv")
    
    output_dir = os.path.join(script_dir, "Sentiment RESULT")
    os.makedirs(output_dir, exist_ok=True)
    train_save_path = os.path.join(output_dir, f"Train_Cro_Atten_sentiment_pred_seed.csv")
    test_save_path = os.path.join(output_dir, f"Test_Cro_Atten_sentiment_pred_seed.csv")
    
    model_save_path = os.path.join(script_dir, "Sentiment RESULT", f"best_model_for_Sentiment_seed.pth")
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        original_train_len = len(train_df)
        train_df = train_df.dropna(subset=['CTR']).reset_index(drop=True)
        print(f"Dropped {original_train_len - len(train_df)} rows with NaN CTR from train_df")
        original_test_len = len(test_df)
        test_df = test_df.dropna(subset=['CTR']).reset_index(drop=True)
        print(f"Dropped {original_test_len - len(test_df)} rows with NaN CTR from test_df")

        train_df = train_df.drop(columns='pred_CTR', errors='ignore')
        test_df = test_df.drop(columns='pred_CTR', errors='ignore')
        
        if 'sentiment' not in train_df.columns or 'sentiment' not in test_df.columns:
            print("错误：数据集中缺少 'sentiment' 列")
            exit(1)
        
        print("\n验证 train_df 列：", train_df.columns.tolist())
        print("验证 test_df 列：", test_df.columns.tolist())
    except FileNotFoundError as e:
        print(f"错误：无法找到文件 {e.filename}")
        exit(1)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = CTRDataset(train_df, tokenizer)
    test_dataset = CTRDataset(test_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用的设备是", device)
    
    model = BertCrossAttentionRegressor().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    
    train_df_with_preds, test_df_with_preds = train_model(model, train_loader, train_df, test_loader, test_df, optimizer, device, epochs=3, model_save_path=model_save_path)
    
    train_df_with_preds.to_csv(train_save_path, index=False)
    test_df_with_preds.to_csv(test_save_path, index=False)
    print(f"训练数据预测结果已保存到 {train_save_path}")
    print(f"测试数据预测结果已保存到 {test_save_path}")

if __name__ == "__main__":
    main()