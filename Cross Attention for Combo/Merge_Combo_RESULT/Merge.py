import pandas as pd
import os

def concatenate_full_prior_predictions():
    # 获取当前工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # e:\VS code project\Cross Attention for Combo
    output_dir = script_dir 

    # 定义组合数量和文件路径
    combos = [1, 2, 3, 4, 5, 6]
    combo_names = ['one', 'two', 'three', 'four', 'five', 'six']
    accuracy_files = [
        os.path.join(parent_dir, f"Combo_{n}_RESULT", f"accuracy_results_{n}_combinations.csv")
        for n in combo_names
    ]

    # 初始化组合列表
    all_combinations = []
    pred_files = []

    # 提取所有组合
    combo_idx = 1
    for combo, combo_name, file_path in zip(combos, combo_names, accuracy_files):
        try:
            # 读取准确率文件
            df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="ignore")
            df['Test_Accuracy'] = pd.to_numeric(df['Test_Accuracy'], errors='coerce')
            
            # 验证必要列
            if 'Combination' not in df.columns or 'Test_Accuracy' not in df.columns:
                print(f"Error: {file_path} missing 'Combination' or 'Test_Accuracy' column, skipping")
                continue
            
            # 获取组合及其准确率
            for _, row in df.iterrows():
                combination = row['Combination']
                test_accuracy = row['Test_Accuracy']
                pred_file = os.path.join(parent_dir, f"Combo_{combo_name}_RESULT", f"Test_Cro_Atten_{combination}_pred_seed.csv")
                pred_files.append((combo_idx, combo, combination, pred_file))
                all_combinations.append({
                    'Combo': combo,
                    'Combo_Name': combo_name,
                    'Combination': combination,
                    'Test_Accuracy': test_accuracy,
                    'Pred_File': pred_file,  # 新增：记录原始文件名
                    'New_Column_Name': f"pred_CTR_{combo_idx}"  # 新增：记录重命名列
                })
                combo_idx += 1
            print(f"Loaded {len(df)} combinations from Combo {combo} ({combo_name})")

        except FileNotFoundError:
            print(f"Error: File {file_path} not found, skipping")
            continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # 验证组合数量
    if len(all_combinations) != 63:
        print(f"Warning: Expected 63 combinations, found {len(all_combinations)}")

    # 初始化拼接数据集
    result_df = None
    pred_columns = {}

    # 读取和拼接预测数据
    for idx, combo, combination, pred_file in pred_files:
        try:
            # 读取预测文件
            df = pd.read_csv(pred_file, encoding="utf-8", encoding_errors="ignore")
            
            # 验证必要列
            if 'test_id' not in df.columns or 'headline' not in df.columns or 'CTR' not in df.columns or 'pred_CTR' not in df.columns:
                print(f"Error: {pred_file} missing required columns, skipping")
                continue
            
            # 转换为浮点数
            df['pred_CTR'] = pd.to_numeric(df['pred_CTR'], errors='coerce')
            
            # 以第一份有效数据为基准
            if result_df is None:
                result_df = df[['test_id', 'headline', 'CTR']].copy()
            
            # 提取 pred_CTR 并重命名
            pred_col_name = f"pred_CTR_{idx}"
            pred_columns[pred_col_name] = df['pred_CTR']
            print(f"Added pred_CTR for Combo {combo}, combination {combination} as {pred_col_name}")
        
        except FileNotFoundError:
            print(f"Error: File {pred_file} not found, skipping")
            continue
        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
            continue

    # 检查是否成功加载数据
    if result_df is None:
        print("Error: No valid prediction files were loaded")
        return

    # 横向拼接 pred_CTR 列
    for pred_col_name, pred_series in pred_columns.items():
        result_df[pred_col_name] = pred_series

    # 保存拼接结果
    output_file = os.path.join(output_dir, "full_prior_predictions.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Concatenated full prior predictions saved to {output_file}")

    # 保存组合信息（包含原始文件名和重命名列）
    combinations_df = pd.DataFrame(all_combinations)
    combinations_save_path = os.path.join(output_dir, "all_combinations.csv")
    combinations_df.to_csv(combinations_save_path, index=False)
    print(f"All combinations with pred file mappings saved to {combinations_save_path}")

if __name__ == "__main__":
    concatenate_full_prior_predictions()