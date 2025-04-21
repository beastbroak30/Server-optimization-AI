import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Number of hours for the dataset 
NUM_HOURS = 500000

# Function to generate a synthetic dataset
def generate_data_center_dataset(num_hours):
    # Initialize random seed for reproducibility
    np.random.seed(42)
    
    # Define base parameters
    start_time = datetime(2025, 4, 19, 0, 0)  # Starting at midnight
    hours = [start_time + timedelta(hours=i) for i in range(num_hours)]
    days = [h.strftime('%A') for h in hours]
    hour_of_day = [h.hour for h in hours]
    
    # Generate data with variations
    # Active users: Random between 500,000 and 2,000,000, with peaks during business hours
    active_users = np.random.randint(500000, 2000000, num_hours)
    active_users = [
        u * (1.3 if 8 <= h <= 17 else 0.7) for u, h in zip(active_users, hour_of_day)
    ]
    
    # CPU Usage: Influenced by active users and AI task load
    cpu_usage = [min(100, max(25, (u / 2000000) * 100 + np.random.normal(0, 6))) 
                 for u in active_users]
    
    # Internal Temperature: Affected by CPU usage, external temp, and cooling
    external_temp = np.random.uniform(18, 45, num_hours)  # Between 18°C and 45°C
    internal_temp = [min(42, max(27, c / 1.8 + e / 2.5 + np.random.normal(0, 2.5))) 
                     for c, e in zip(cpu_usage, external_temp)]
    
    # External Humidity: Random between 65% and 95%
    external_humidity = np.random.uniform(65, 95, num_hours)
    
    # Power Draw: Correlated with CPU usage and internal temp
    power_draw = [min(220, max(110, c * 1.6 + (t - 30) * 2 + np.random.normal(0, 12))) 
                  for c, t in zip(cpu_usage, internal_temp)]
    
    # Solar/Wind Contribution: Higher during day, affected by humidity
    solar_wind = [max(0, 35 * (1 - abs(12 - h) / 12) * (0.8 if e > 85 else 1) + np.random.normal(0, 6)) 
                  for h, e in zip(hour_of_day, external_humidity)]
    
    # Grid Price: Higher during peak hours (8 AM - 8 PM)
    grid_price = [0.20 if 8 <= h <= 20 else 0.10 + np.random.normal(0, 0.03) 
                  for h in hour_of_day]
    
    # Occupancy: Higher during business hours
    occupancy = [np.random.randint(12, 25) if 8 <= h <= 17 else np.random.randint(1, 10) 
                 for h in hour_of_day]
    
    # AI Task Load: Higher probability during business hours
    ai_task_load = [1 if (np.random.random() < 0.4 and 8 <= h <= 17) or np.random.random() < 0.2 else 0 
                    for h in hour_of_day]
    
    # Events: Include AI_Demand_High and other situations
    events = np.random.choice(
        ['Normal', 'Heatwave', 'Maintenance', 'Power_Spike', 'AI_Demand_High'], 
        num_hours, 
        p=[0.5, 0.15, 0.1, 0.1, 0.15]
    )
    
    # Adjust parameters based on events
    for i, event in enumerate(events):
        if event == 'Heatwave':
            external_temp[i] = min(38, external_temp[i] + 6)  # Hotter weather
            internal_temp[i] = min(42, internal_temp[i] + 4)  # Increased internal temp
            power_draw[i] = min(220, power_draw[i] + 15)      # More power for cooling
            external_humidity[i] = max(65, external_humidity[i] - 10)  # Drier conditions
        elif event == 'Maintenance':
            cpu_usage[i] = max(25, cpu_usage[i] * 0.7)        # Reduced CPU usage
            power_draw[i] = max(110, power_draw[i] * 0.8)     # Reduced power draw
            active_users[i] = max(500000, active_users[i] * 0.6)  # Fewer users
            ai_task_load[i] = 0                               # No AI tasks during maintenance
        elif event == 'Power_Spike':
            power_draw[i] = min(220, power_draw[i] + 25)      # Sudden power spike
            cpu_usage[i] = min(100, cpu_usage[i] + 15)        # Increased CPU usage
            grid_price[i] = min(0.25, grid_price[i] + 0.05)   # Higher grid price
        elif event == 'AI_Demand_High':
            cpu_usage[i] = min(100, cpu_usage[i] + 20)        # High CPU demand
            power_draw[i] = min(220, power_draw[i] + 20)      # Increased power usage
            active_users[i] = min(2000000, active_users[i] * 1.2)  # More users
            ai_task_load[i] = 1                               # AI tasks active
            internal_temp[i] = min(42, internal_temp[i] + 3)  # Higher internal temp
    
    # Create DataFrame
    data = {
        'CPU_Usage': cpu_usage,
        'Internal_Temp': internal_temp,
        'External_Temp': external_temp,
        'External_Humidity': external_humidity,
        'Power_Draw': power_draw,
        'Solar_Wind': solar_wind,
        'Grid_Price': grid_price,
        'Occupancy': occupancy,
        'Day': days,
        'Hour': hour_of_day,
        'Active_Users': active_users,
        'AI_Task_Load': ai_task_load,
        'Event': events
    }
    
    df = pd.DataFrame(data)
    
    # Round numerical columns to 1 decimal place
    numerical_cols = ['CPU_Usage', 'Internal_Temp', 'External_Temp', 
                      'External_Humidity', 'Power_Draw', 'Solar_Wind', 
                      'Grid_Price', 'Active_Users']
    df[numerical_cols] = df[numerical_cols].round(1)
    
    return df

# Function to visualize the dataset
def visualize_dataset(df):
    plt.figure(figsize=(12, 8))
    
    # Plot CPU Usage
    plt.subplot(3, 1, 1)
    plt.plot(df['Hour'], df['CPU_Usage'], label='CPU Usage (%)', color='blue')
    plt.title('CPU Usage Over Time')
    plt.xlabel('Hour')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.legend()
    
    # Plot Power Draw
    plt.subplot(3, 1, 2)
    plt.plot(df['Hour'], df['Power_Draw'], label='Power Draw (kW)', color='green')
    plt.title('Power Draw Over Time')# this is  being created to for showing the visualization of the dataset.
    plt.xlabel('Hour')
    plt.ylabel('Power Draw (kW)')
    plt.grid(True)
    plt.legend()
    
    # Plot Active Users
    plt.subplot(3, 1, 3)
    plt.plot(df['Hour'], df['Active_Users'], label='Active Users', color='orange')
    plt.title('Active Users Over Time')
    plt.xlabel('Hour')
    plt.ylabel('Active Users')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data_center_visualization.png')
    plt.show()

# Generate the dataset
dataset = generate_data_center_dataset(NUM_HOURS)

# Save to CSV
dataset.to_csv(f'data_center_dataset_{NUM_HOURS}.csv', index=False)

# Visualize the dataset


# Print the first few rows for verification
print(dataset.head())
