# 这个脚本得到了使用先验信息的max min mean作为ucb算法的先验进行模拟实验后得到的准确率结果

import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool, Manager
from threading import Lock
import scipy.stats as stats

os.environ['PYTHONUNBUFFERED'] = '1'

def compute_confidence_interval(results):
    results = np.array(results)
    mean = np.mean(results)
    std_dev = np.std(results, ddof=1)
    std_error = std_dev / np.sqrt(len(results))
    z_score = stats.norm.ppf(0.975)
    margin_of_error = z_score * std_error
    return (mean - margin_of_error, mean + margin_of_error)

def get_UCB(num_trials, num_positive, z_value, t, ctr_llm, n_llm):
    mean_val = num_positive / num_trials
    ucb_llm = (ctr_llm * n_llm + num_positive) / (num_trials + n_llm) + z_value * np.sqrt(np.log(t) / (num_trials + n_llm))
    return ucb_llm

def simulate_bandit(df, T, n_value, z_value, ini=200, prior_type='max', output_dir='./results'):
    rng = np.random.RandomState(42)
    results = []
    selected_titles = []
    
    pred_columns = [f'pred_CTR_{i}' for i in range(1, 64)]
    
    try:
        sampled_ids = df['test_id'].drop_duplicates()
        if len(sampled_ids) < n_value:
            print(f"Warning: Only {len(sampled_ids)} test_ids available, requested {n_value}", flush=True)
            n_value = len(sampled_ids)
        sampled_ids = sampled_ids.sample(n=n_value, random_state=rng)
        print(f"Sampled {len(sampled_ids)} test_ids for T={T}, prior={prior_type}, PID={os.getpid()}", flush=True)
        sampled_df = df[df['test_id'].isin(sampled_ids)]
        
        for test_id, group in sampled_df.groupby('test_id'):
            if len(group) == 0:
                print(f"Warning: Empty group for test_id {test_id}, skipping", flush=True)
                continue
            trials = np.ones(len(group))
            successes = np.zeros(len(group))
            n_llm = ini * np.ones(len(group))
            total_T = int(len(group) * T)
            rewards = np.zeros(total_T)
            Correct = 0
            
            # 计算最优标题索引（相对索引）
            opt_arm = group['CTR'].idxmax()
            opt_arm_idx = group.index.get_loc(opt_arm)
            
            # 计算先验 CTR
            pred_ctrs = np.array(group[pred_columns].values)
            pred_ctrs = np.maximum(pred_ctrs, 0)
            if prior_type == 'max':
                ctr_llm = np.max(pred_ctrs, axis=1)
            elif prior_type == 'min':
                ctr_llm = np.min(pred_ctrs, axis=1)
            elif prior_type == 'mean':
                ctr_llm = np.mean(pred_ctrs, axis=1)
            else:
                raise ValueError(f"Invalid prior_type: {prior_type}")
            
            for t in range(total_T):
                ucbs = get_UCB(trials, successes, z_value, t + 1, ctr_llm, n_llm)
                chosen_arm = np.argmax(ucbs)
                
                ctr = group.iloc[chosen_arm]['CTR']
                reward = rng.binomial(1, ctr)
                rewards[t] = reward
                
                trials[chosen_arm] += 1
                successes[chosen_arm] += reward
                
                # 记录选中的标题
                selected_titles.append({
                    'test_id': test_id,
                    't': t,
                    'chosen_arm': chosen_arm
                })
                
                # 计算准确率
                if chosen_arm == opt_arm_idx:
                    Correct += 1
            
            accuracy = Correct / total_T if total_T > 0 else 0.0
            results.append({
                'test_id': test_id,
                'reward_sum': float(np.sum(rewards)),
                'accuracy': accuracy
            })
    
        
        print(f"Bandit simulation completed for T={T}, prior={prior_type}, results size: {len(results)}", flush=True)
    except Exception as e:
        print(f"Error in simulate_bandit for T={T}, prior={prior_type}: {e}", flush=True)
        raise
    return results

def run_simulation(args):
    T, test_df, fixed_n_llm, z_value, lock, output_csv = args
    try:
        print(f"Subprocess started for T={T}, PID={os.getpid()}", flush=True)
        print('----------------------------------', flush=True)
        print('Test for T = ', T, flush=True)
        n_test = test_df.groupby('test_id').ngroups
        if n_test == 0:
            raise ValueError("No valid test_ids found")
        print(f"Number of test_ids: {n_test}", flush=True)
        
        results_max = simulate_bandit(test_df, T, n_test, z_value, ini=fixed_n_llm, prior_type='max', output_dir=os.path.dirname(output_csv))
        results_min = simulate_bandit(test_df, T, n_test, z_value, ini=fixed_n_llm, prior_type='min', output_dir=os.path.dirname(output_csv))
        results_mean = simulate_bandit(test_df, T, n_test, z_value, ini=fixed_n_llm, prior_type='mean', output_dir=os.path.dirname(output_csv))
        
        return {
            'T': T,
            'z_ucb': z_value,
            'llm_ucb_max_result': [r['reward_sum'] for r in results_max],
            'llm_ucb_min_result': [r['reward_sum'] for r in results_min],
            'llm_ucb_mean_result': [r['reward_sum'] for r in results_mean],
            'accuracy_max': [r['accuracy'] for r in results_max],
            'accuracy_min': [r['accuracy'] for r in results_min],
            'accuracy_mean': [r['accuracy'] for r in results_mean],
            'mean_accuracy_max': np.mean([r['accuracy'] for r in results_max]) if results_max else 0.0,
            'mean_accuracy_min': np.mean([r['accuracy'] for r in results_min]) if results_min else 0.0,
            'mean_accuracy_mean': np.mean([r['accuracy'] for r in results_mean]) if results_mean else 0.0
        }
    except Exception as e:
        print(f"Error in run_simulation for T={T}: {e}", flush=True)
        raise

def init_worker():
    print(f"Worker initialized, PID={os.getpid()}", flush=True)

def run_simulation_for_method(data_path, output_csv, T_list, fixed_n_llm=200, z_value=0.08):
    try:
        print(f"Starting simulation for method: multi_prior", flush=True)
        print("Loading data...", flush=True)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        test_df = pd.read_csv(data_path)
        print("Data loaded", flush=True)
        print("Validating data...", flush=True)
        
        required_columns =  [f'pred_CTR_{i}' for i in range(1, 64)]

        if not all(col in test_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        if test_df[required_columns].isnull().any().any():
            raise ValueError("Data contains missing values in required columns")
        test_df = test_df[test_df['test_id'].notnull()]
        test_df = test_df.groupby('test_id').filter(lambda x: len(x) > 0)
        
        manager = Manager()
        lock = manager.Lock()
        print("Starting simulations...", flush=True)
        with Pool(processes=12, initializer=init_worker) as pool:
            args = [(t, test_df, fixed_n_llm, z_value, lock, output_csv) for t in T_list]
            results = list(pool.imap_unordered(run_simulation, args))
            print(f"Completed {len(results)} tasks for method multi_prior", flush=True)
        
        if results:
            csv_dir = os.path.dirname(output_csv) or '.'
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
                print(f"Created CSV output directory: {csv_dir}", flush=True)
            if not os.access(csv_dir, os.W_OK):
                raise PermissionError(f"CSV output directory {csv_dir} is not writable")
            print(f"CSV output directory {csv_dir} is writable", flush=True)
            ##############################################################################################################################
            # ### 追加一部分数据
            # if os.path.exists(output_csv):
            #     existing_df = pd.read_csv(output_csv)
            #     print(f"Loaded existing CSV with {len(existing_df)} rows", flush=True)
            # else:
            #     existing_df = pd.DataFrame()
            #     print("No existing CSV found, creating new one", flush=True)
            
            # for result in results:
            #     result['llm_ucb_max_result'] = str([float(x) for x in result['llm_ucb_max_result']])
            #     result['llm_ucb_min_result'] = str([float(x) for x in result['llm_ucb_min_result']])
            #     result['llm_ucb_mean_result'] = str([float(x) for x in result['llm_ucb_mean_result']])
            #     result['accuracy_max'] = str([float(x) for x in result['accuracy_max']])
            #     result['accuracy_min'] = str([float(x) for x in result['accuracy_min']])
            #     result['accuracy_mean'] = str([float(x) for x in result['accuracy_mean']])
            #     result['mean_accuracy_max'] = float(result['mean_accuracy_max'])
            #     result['mean_accuracy_min'] = float(result['mean_accuracy_min'])
            #     result['mean_accuracy_mean'] = float(result['mean_accuracy_mean'])
            
            # new_df = pd.DataFrame(results)
            # combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # combined_df.to_csv(output_csv, index=False)
            # print("Success save new data AP_UCB")
            ##############################################################################################################################
            for result in results:
                result['llm_ucb_max_result'] = str([float(x) for x in result['llm_ucb_max_result']])
                result['llm_ucb_min_result'] = str([float(x) for x in result['llm_ucb_min_result']])
                result['llm_ucb_mean_result'] = str([float(x) for x in result['llm_ucb_mean_result']])
                result['accuracy_max'] = str([float(x) for x in result['accuracy_max']])
                result['accuracy_min'] = str([float(x) for x in result['accuracy_min']])
                result['accuracy_mean'] = str([float(x) for x in result['accuracy_mean']])
                result['mean_accuracy_max'] = float(result['mean_accuracy_max'])
                result['mean_accuracy_min'] = float(result['mean_accuracy_min'])
                result['mean_accuracy_mean'] = float(result['mean_accuracy_mean'])
            
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"Combined CSV saved: {output_csv}, file size: {os.path.getsize(output_csv)} bytes", flush=True)
        else:
            print(f"Warning: No results generated for method multi_prior, CSV not created", flush=True)
        
    except Exception as e:
        print(f"Error in run_simulation_for_method (multi_prior): {e}", flush=True)
        raise

def main():
    try:
        print("Script started", flush=True)
        start_time = time.time()
        print(f"Starting at {time.ctime()}", flush=True)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_path = os.path.join(parent_dir, "Merge_Combo_RESULT", "full_prior_predictions.csv")
        
        output_csv = './UCB_results/AP_UCB_simulation_results.csv'
        
        T_list = [200, 400, 600, 800, 1000, 1200, 1500,1800]
        fixed_n_llm = 200
        z_value = 0.08
        
        run_simulation_for_method(
            data_path=data_path,
            output_csv=output_csv,
            T_list=T_list,
            fixed_n_llm=fixed_n_llm,
            z_value=z_value
        )
        
        print(f"Finished in {time.time() - start_time:.2f} seconds", flush=True)
        
    except Exception as e:
        print(f"Main error: {e}", flush=True)
        raise

if __name__ == '__main__':
    main()