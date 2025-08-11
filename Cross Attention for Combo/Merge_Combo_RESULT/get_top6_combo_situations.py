import pandas as pd
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)        # ## e:\VS code project\Cross Attention for Combo
output_dir = script_dir 



combined_df = pd.read_csv(os.path.join(script_dir,"all_combinations.csv"))
top_6_df = combined_df.sort_values(by='Test_Accuracy', ascending=False).head(6)
print("\nTop 6 combinations by Test_Accuracy:")
for idx, row in top_6_df.iterrows():
    print(f"Combo {row['Combo']} ({row['Combo_Name']}): {row['Combination']}, Test_Accuracy: {row['Test_Accuracy']:.3f}")
    
# 保存结果
output_file = os.path.join(output_dir, "top_6_accuracy_combinations.csv")
top_6_df.to_csv(output_file, index=False)
print(f"\nTop 6 accuracy combinations saved to {output_file}")
