import json

notebook_path = '/Users/sondoannam/vinuni/Day25-Lab-GPU-FinOps-Cost_Optimization/notebook/gpu_finops_lab.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 27
cell_27_code = """def analyze_multi_gpu_cost():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Synthetic data
    gpus = [1, 2, 4, 8]
    # Linear cost increase, diminishing returns on time
    # Time (hours) = 100 / (GPUs ** 0.8)
    time_hours = [100 / (g**0.8) for g in gpus]
    cost_per_hour = 3.0 # cost per GPU per hour
    total_cost = [t * g * cost_per_hour for t, g in zip(time_hours, gpus)]
    
    # Scaling Efficiency (%) = (Time_1 / (Time_N * N)) * 100
    efficiency = [(time_hours[0] / (time_hours[i] * gpus[i])) * 100 for i in range(len(gpus))]
    
    df = pd.DataFrame({
        'GPU Count': gpus,
        'Time (hours)': time_hours,
        'Total Cost ($)': total_cost,
        'Scaling Efficiency (%)': efficiency
    })
    
    print("Multi-GPU Scaling Analysis:")
    print(df.to_string(index=False))
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of GPUs')
    ax1.set_ylabel('Total Cost ($)', color=color)
    ax1.plot(df['GPU Count'], df['Total Cost ($)'], marker='o', color=color, linewidth=2, label='Total Cost')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Scaling Efficiency (%)', color=color)
    ax2.plot(df['GPU Count'], df['Scaling Efficiency (%)'], marker='s', color=color, linewidth=2, linestyle='--', label='Efficiency')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 110)
    
    plt.title('Multi-GPU Training: Cost vs. Scaling Efficiency')
    fig.tight_layout()
    plt.savefig('multi_gpu_scaling.png', bbox_inches='tight')
    plt.show()
    
    return df

scaling_df = analyze_multi_gpu_cost()"""

# Cell 28
cell_28_code = """def forecast_project_cost():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    # Synthetic historical data (past 30 days)
    np.random.seed(42)
    dates_past = [datetime.today() - timedelta(days=x) for x in range(30, 0, -1)]
    base_cost = 100
    trend = np.linspace(0, 20, 30)
    noise = np.random.normal(0, 10, 30)
    past_costs = base_cost + trend + noise
    
    # Forecast future data (next 30 days)
    dates_future = [datetime.today() + timedelta(days=x) for x in range(1, 31)]
    future_trend = np.linspace(20, 50, 30)
    forecast_costs = base_cost + future_trend
    
    # Confidence intervals
    std_dev = np.std(noise)
    upper_ci = forecast_costs + 1.96 * std_dev
    lower_ci = forecast_costs - 1.96 * std_dev
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates_past, past_costs, label='Historical Cost', color='blue', marker='.')
    plt.plot(dates_future, forecast_costs, label='Forecast Cost', color='orange', linestyle='--')
    plt.fill_between(dates_future, lower_ci, upper_ci, color='orange', alpha=0.2, label='95% Confidence Interval')
    
    plt.title('Project Cost Forecasting (30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Daily Cost ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('project_forecast.png', bbox_inches='tight')
    plt.show()
    
    return pd.DataFrame({'Date': dates_future, 'Forecast': forecast_costs, 'Lower_CI': lower_ci, 'Upper_CI': upper_ci})

forecast_df = forecast_project_cost()"""

# Cell 29
cell_29_code = """def analyze_optimization_opportunities():
    import pandas as pd
    
    # Mock data for instances
    data = {
        'Instance_ID': ['i-01', 'i-02', 'i-03', 'i-04', 'i-05'],
        'Instance_Type': ['p4d.24xlarge', 'p3.2xlarge', 'g4dn.xlarge', 'p3.8xlarge', 'g5.2xlarge'],
        'Avg_Utilization_%': [85, 25, 90, 15, 45],
        'Uptime_Hours': [720, 720, 120, 500, 720],
        'Monthly_Cost': [23000, 2200, 380, 8500, 750]
    }
    df = pd.DataFrame(data)
    
    recommendations = []
    
    for _, row in df.iterrows():
        if row['Avg_Utilization_%'] < 30 and row['Uptime_Hours'] > 300:
            rec = f"Downsize or terminate {row['Instance_ID']} ({row['Instance_Type']}). Low util ({row['Avg_Utilization_%']}%) but high uptime."
            priority = 'High'
        elif 30 <= row['Avg_Utilization_%'] < 60:
            rec = f"Consider spot instances or scheduling for {row['Instance_ID']} ({row['Instance_Type']}). Mod util ({row['Avg_Utilization_%']}%)."
            priority = 'Medium'
        else:
            rec = f"Monitor {row['Instance_ID']} ({row['Instance_Type']}). Good util ({row['Avg_Utilization_%']}%)."
            priority = 'Low'
            
        recommendations.append({'Instance': row['Instance_ID'], 'Priority': priority, 'Recommendation': rec})
        
    rec_df = pd.DataFrame(recommendations)
    
    print("Optimization Opportunities:")
    print("-" * 50)
    for p in ['High', 'Medium', 'Low']:
        subset = rec_df[rec_df['Priority'] == p]
        for _, row in subset.iterrows():
            print(f"[{row['Priority']}] {row['Recommendation']}")
            
    return rec_df

recommendations_df = analyze_optimization_opportunities()"""

# Cell 30
cell_30_code = """def create_advanced_finops_dashboard(scaling_df, forecast_df, rec_df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Scaling Efficiency
    ax1 = fig.add_subplot(gs[0, 0])
    color1 = 'tab:red'
    ax1.plot(scaling_df['GPU Count'], scaling_df['Total Cost ($)'], marker='o', color=color1)
    ax1.set_title('GPU Scaling: Cost vs Efficiency')
    ax1.set_xlabel('GPUs')
    ax1.set_ylabel('Cost ($)', color=color1)
    ax1_twin = ax1.twinx()
    color2 = 'tab:blue'
    ax1_twin.plot(scaling_df['GPU Count'], scaling_df['Scaling Efficiency (%)'], marker='s', linestyle='--', color=color2)
    ax1_twin.set_ylabel('Efficiency (%)', color=color2)
    ax1_twin.set_ylim(0, 110)
    
    # 2. Forecast
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(forecast_df['Date'], forecast_df['Forecast'], color='orange', label='Forecast')
    ax2.fill_between(forecast_df['Date'], forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='orange', alpha=0.2)
    ax2.set_title('30-Day Cost Forecast')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Cost ($)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Recommendations Priority Pie Chart
    ax3 = fig.add_subplot(gs[1, 0])
    priority_counts = rec_df['Priority'].value_counts()
    ax3.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', 
            colors=['#ff9999','#66b3ff','#99ff99'])
    ax3.set_title('Optimization Recommendations by Priority')
    
    # 4. Mock Cost Breakdown Bar Chart
    ax4 = fig.add_subplot(gs[1, 1])
    services = ['Compute', 'Storage', 'Network', 'Data Transfer']
    costs = [12000, 3500, 800, 1500]
    sns.barplot(x=services, y=costs, ax=ax4, palette='viridis')
    ax4.set_title('Monthly Cost Breakdown by Service')
    ax4.set_ylabel('Cost ($)')
    
    plt.tight_layout()
    plt.savefig('advanced_finops_dashboard.png', bbox_inches='tight')
    plt.show()

create_advanced_finops_dashboard(scaling_df, forecast_df, recommendations_df)"""

# Cell 31
cell_31_code = """def plot_optimization_roadmap():
    import matplotlib.pyplot as plt
    
    phases = [
        '1. Implement Tagging & Monitoring',
        '2. Right-size Underutilized Instances',
        '3. Transition to Spot Instances',
        '4. Implement Auto-scaling',
        '5. Architecture & AMP Tuning'
    ]
    start_days = [0, 7, 14, 21, 30]
    durations = [7, 10, 14, 20, 30]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, phase in enumerate(phases):
        ax.barh(phase, durations[i], left=start_days[i], color=colors[i], alpha=0.8)
        
    ax.set_xlabel('Days from Start')
    ax.set_title('FinOps Optimization Strategy Roadmap')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('optimization_roadmap.png', bbox_inches='tight')
    plt.show()

plot_optimization_roadmap()"""

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell.get('source', []))
    if '# TODO: Implement function to analyze costs for different GPU counts' in source:
        cell['source'] = [line + '\n' for line in cell_27_code.split('\n')]
    elif '# TODO: Implement project cost forecasting with confidence intervals' in source:
        cell['source'] = [line + '\n' for line in cell_28_code.split('\n')]
    elif '# TODO: Implement optimization opportunity analysis with prioritization' in source:
        cell['source'] = [line + '\n' for line in cell_29_code.split('\n')]
    elif '# TODO: Create comprehensive dashboard combining all Part 8.5 analyses' in source:
        cell['source'] = [line + '\n' for line in cell_30_code.split('\n')]
    elif '# TODO: Design a comprehensive cost optimization strategy for a given scenario' in source:
        cell['source'] = [line + '\n' for line in cell_31_code.split('\n')]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully!")
