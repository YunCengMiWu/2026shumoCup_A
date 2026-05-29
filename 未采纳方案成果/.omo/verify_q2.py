import pandas as pd

df = pd.read_csv('results/q2_results.csv')

print('rows:', len(df))
print('slack present:', all(c in df.columns for c in ['s_self(MWh)', 's_green(MWh)', 's_feed(MWh)']))

# Find the target column (contains 吨) and status column
target_col = [c for c in df.columns if '吨' in c and '目标' in c][0]
status_col = [c for c in df.columns if '状态' in c][0]

print('target_col:', target_col)
print('status_col:', status_col)
print('D=36 feasible:', len(df[(df[target_col] == 36) & (df[status_col] == '最优')]))
print()
print('=== Columns ===')
for c in df.columns:
    print(c)
print()
print('=== Slack sample (first 5 feasible) ===')
feasible = df[df[status_col] == '最优']
print(feasible[['s_self(MWh)', 's_green(MWh)', 's_feed(MWh)']].head())
