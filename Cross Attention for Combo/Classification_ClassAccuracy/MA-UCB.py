import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool, Manager
from threading import Lock
import scipy.stats as stats
from collections import defaultdict

os.environ['PYTHONUNBUFFERED'] = '1'
CLASS = 'Culture & Lifestyle' ###  Culture & Lifestyle,  Economy & Business,  Environment & Safety,  Politics & International

## CLASS 用于控制我们选择哪一个分类的结果进行ucb实验

def compute_confidence_interval(results):
    results = np.array(results)
    mean = np.mean(results)
    std_dev = np.std(results, ddof=1)
    std_error = std_dev / np.sqrt(len(results))
    z_score = stats.norm.ppf(0.975)
    margin_of_error = z_score * std_error
    return (mean - margin_of_error, mean + margin_of_error)

def get_UCB(num_trials_1, num_trials_2, num_positive, z_value, t, pred_ctrs, n_llm):
    mean_val = (num_positive + pred_ctrs * n_llm) / (num_trials_1 + n_llm)
    ucb_llm = mean_val + z_value * np.sqrt(np.log(t) / (num_trials_2 + n_llm))
    return ucb_llm

def simulate_bandit(df, T, n_value, z_value, ini=200, output_dir='./results'):
    rng = np.random.RandomState(42)
    results = []
    selected_titles = []
    pred_columns = [f'pred_CTR_{i}' for i in range(1, 64)]  
    prior_types = [f'pred_CTR_{i}' for i in range(1, 64)]  # 63 个先验策略（每个 pred_CTR）
    num_priors = len(prior_types)  # 63
    
    try:
        sampled_ids = df['test_id'].drop_duplicates()
        if len(sampled_ids) < n_value:
            print(f"Warning: Only {len(sampled_ids)} test_ids available, requested {n_value}", flush=True)
            n_value = len(sampled_ids)
        sampled_ids = sampled_ids.sample(n=n_value, random_state=rng)
        print(f"Sampled {len(sampled_ids)} test_ids for T={T}, PID={os.getpid()}", flush=True)
        sampled_df = df[df['test_id'].isin(sampled_ids)]
        
        for test_id, group in sampled_df.groupby('test_id'):
            if len(group) == 0:
                print(f"Warning: Empty group for test_id {test_id}, skipping", flush=True)
                continue
            trials_1 = np.ones(len(group))
            trials_2 = np.ones((len(group), num_priors))  # 63 个先验
            successes = np.zeros(len(group))
            pred_ctrs_raw = np.array(group[pred_columns].values)
            pred_ctrs = np.maximum(pred_ctrs_raw, 0)  # 如果小于0则设为0
            n_llm = ini * np.ones(len(group))
            total_T = int(len(group) * T)
            rewards = np.zeros(total_T)
            Correct = 0
            
            # 计算最优标题索引（相对索引）
            opt_arm = group['CTR'].idxmax()
            opt_arm_idx = group.index.get_loc(opt_arm)
            
            headlines = group['headline'].values  # 获取headline列表
            
            for t in range(total_T):
                ucbs = np.zeros((len(group), num_priors))
                for k in range(num_priors):
                    ucbs[:, k] = get_UCB(trials_1, trials_2[:, k], successes, z_value, t + 1, pred_ctrs[:, k], n_llm)
                
                U_max = np.max(ucbs, axis=1)
                chosen_arm = np.argmax(U_max)
                chosen_attr = np.argmax(ucbs[chosen_arm])  # 0 to 62: corresponding to pred_CTR_1 to 63
                
                ctr = group.iloc[chosen_arm]['CTR']
                reward = rng.binomial(1, ctr)
                rewards[t] = reward
                
                trials_1[chosen_arm] += 1
                trials_2[chosen_arm, chosen_attr] += 1
                successes[chosen_arm] += reward
                
                # 记录选中的标题和先验属性
                selected_titles.append({
                    'test_id': test_id,
                    't': t,
                    'chosen_arm': chosen_arm,
                    'chosen_attr': prior_types[chosen_attr],  # 记录 pred_CTR_1 to pred_CTR_63
                    'headline': headlines[chosen_arm]  # 添加headline
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
        
        
        
        print(f"Bandit simulation completed for T={T}, results size: {len(results)}", flush=True)
    except Exception as e:
        print(f"Error in simulate_bandit for T={T}: {e}", flush=True)
        raise
    return results, selected_titles

def run_simulation(args):
    T, test_df, fixed_n_llm, best_z_ucb, lock, output_csv = args
    try:
        print(f"Subprocess started for T={T}, PID={os.getpid()}", flush=True)
        print('----------------------------------', flush=True)
        print('Test for T = ', T, flush=True)
        n_test = test_df.groupby('test_id').ngroups
        if n_test == 0:
            raise ValueError("No valid test_ids found")
        print(f"Number of test_ids: {n_test}", flush=True)
        
        results, selected_titles = simulate_bandit(test_df, T, n_test, best_z_ucb, ini=fixed_n_llm, output_dir=os.path.dirname(output_csv))
        print(f"LLM UCB result size: {len(results)}", flush=True)
        
        # 计算平均准确率
        mean_accuracy = np.mean([r['accuracy'] for r in results]) if results else 0.0
        
        return {
            'T': T,
            'z_ucb': best_z_ucb,
            'llm_ucb_result': [r['reward_sum'] for r in results],
            'accuracy': [r['accuracy'] for r in results],
            'mean_accuracy': mean_accuracy,
            'selected_titles': selected_titles
        }
    except Exception as e:
        print(f"Error in run_simulation for T={T}: {e}", flush=True)
        raise

def init_worker():
    print(f"Worker initialized, PID={os.getpid()}", flush=True)

def run_simulation_for_method(data_path, output_csv, T_list, fixed_n_llm=200, best_z_ucb=0.08):
    try:
        print(f"Starting simulation for method: random_select", flush=True)
        print("Loading data...", flush=True)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        test_df = pd.read_csv(data_path)
        print("Data loaded", flush=True)
        print("Validating data...", flush=True)
        
        required_columns = ['test_id', 'CTR', 'headline'] + [f'pred_CTR_{i}' for i in range(1, 64)]
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
            args = [(t, test_df, fixed_n_llm, best_z_ucb, lock, output_csv) for t in T_list]
            results = list(pool.imap_unordered(run_simulation, args))
            print(f"Completed {len(results)} tasks for method random_select", flush=True)
        
        if results:
            csv_dir = os.path.dirname(output_csv) or '.'
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
                print(f"Created CSV output directory: {csv_dir}", flush=True)
            if not os.access(csv_dir, os.W_OK):
                raise PermissionError(f"CSV output directory {csv_dir} is not writable")
            print(f"CSV output directory {csv_dir} is writable", flush=True)
            
            # 先进行聚合（使用selected_titles）
            aggregated = defaultdict(lambda: defaultdict(int))
            for result in results:
                T = result['T']
                for sel in result['selected_titles']:
                    key = (T, sel['test_id'], sel['chosen_arm'])
                    aggregated[key][sel['chosen_attr']] += 1
            
            rows = []
            for (T, test_id, chosen_arm), counts in aggregated.items():
                # 只包含有次数的先验
                filtered_counts = {k: v for k, v in counts.items() if v > 0}
                rows.append({
                    'T': T,
                    'test_id': test_id,
                    'headline': chosen_arm,
                    'chosen_attr': str(filtered_counts)  # 转换为字符串字典
                })
            agg_df = pd.DataFrame(rows)
            agg_csv = os.path.join(csv_dir, f"MA_UCB_aggregated_attrs_{CLASS}.csv")
            agg_df.to_csv(agg_csv, index=False)
            print(f"Aggregated attrs CSV saved: {agg_csv}, rows: {len(agg_df)}, file size: {os.path.getsize(agg_csv)} bytes", flush=True)
            
            # 然后处理主CSV，移除selected_titles
            for result in results:
                result['llm_ucb_result'] = str([float(x) for x in result['llm_ucb_result']])
                result['accuracy'] = str([float(x) for x in result['accuracy']])
                result['mean_accuracy'] = float(result['mean_accuracy'])
                result.pop('selected_titles', None)
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"Combined CSV saved: {output_csv}, file size: {os.path.getsize(output_csv)} bytes", flush=True)
            ############################################################################################################################
        #     # ### 追加一部分数据
        #     # if os.path.exists(output_csv):
        #     #     existing_df = pd.read_csv(output_csv)
        #     #     print(f"Loaded existing CSV with {len(existing_df)} rows", flush=True)
        #     # else:
        #     #     existing_df = pd.DataFrame()
        #     #     print("No existing CSV found, creating new one", flush=True)
            
        #     # for result in results:
        #     #     result['llm_ucb_result'] = str([float(x) for x in result['llm_ucb_result']])
        #     #     result['accuracy'] = str([float(x) for x in result['accuracy']])
        #     #     result['mean_accuracy'] = float(result['mean_accuracy'])
            
        #     # new_df = pd.DataFrame(results)
        #     # combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        #     # combined_df.to_csv(output_csv, index=False)
        #     # print(f"Success save new data MA_{CLASS}")
        #     ##############################################################################################################################

        else:
            print(f"Warning: No results generated for method random_select, CSV not created", flush=True)
        
    except Exception as e:
        print(f"Error in run_simulation_for_method (random_select): {e}", flush=True)
        raise

def main():
    try:
        print("Script started", flush=True)
        start_time = time.time()
        print(f"Starting at {time.ctime()}", flush=True)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(script_dir)
    
        data_path = os.path.join(script_dir, "category_datasets", f"{CLASS}_dataset.csv")
        
        out_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "UCB_results")
        output_csv = os.path.join(out_dir, f"MA_UCB_simulate_results_{CLASS}.csv")

        print(output_csv)
        
        T_list = [200, 400, 600, 800, 1000, 1200, 1500, 1800]

        fixed_n_llm = 200
        best_z_ucb = 0.08
        
        run_simulation_for_method(
            data_path=data_path,
            output_csv=output_csv,
            T_list=T_list,
            fixed_n_llm=fixed_n_llm,
            best_z_ucb=best_z_ucb
        )
        
        print(f"Finished in {time.time() - start_time:.2f} seconds", flush=True)
        
    except Exception as e:
        print(f"Main error: {e}", flush=True)
        raise

if __name__ == '__main__':
    main()