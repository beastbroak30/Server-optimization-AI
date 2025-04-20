# AI-Powered Server Efficiency Optimization

This project presents a **Python-based AI model** designed to enhance **server efficiency** for Microsoft data centers. The model is trained using **Google Colab** and considers various real-time environmental, operational, and human activity factors to make smart, energy-efficient decisions.

---

## 🧠 AI Case Study Overview

In large data centers, energy consumption and cooling costs are critical factors. We built an AI model that:

- Monitors **real-time metrics**
- Makes **dynamic decisions**
- Learns over time
- Controls server operations and environment adaptively

The model was trained on simulated and real-world-style datasets, taking into account various influencing factors.

---

## 📥 Input Features (Used for AI Training)

| Category           | Input Feature         | Description                                | Example Value       |
|-------------------|------------------------|--------------------------------------------|---------------------|
| Server Metrics     | CPU Usage             | Current server workload                    | 75%                 |
| Cooling Demand     | Internal Temperature  | Inside data center temperature             | 30°C               |
| Weather            | External Temperature  | Outside air temperature                    | 25°C               |
| Humidity           | External Humidity     | Affects cooling efficiency                 | 60%                 |
| Power Usage        | Current Power Draw    | Total energy usage now                     | 120 kW              |
| Renewable Energy   | Solar/Wind Availability | Renewable % available now               | 40% solar           |
| Electricity Cost   | Real-Time Grid Price  | Cost of electricity per kWh                | $0.15               |
| Human Activity     | Occupancy Count       | Number of people inside                    | 10                  |
| Time Context       | Day & Time            | Time of day and weekday/weekend            | Monday, 3 PM        |
| Server Requests    | Active Users/Requests | Number of active users on systems          | 2500                |
| AI Usage           | AI Task Load          | Is there a big AI training job coming?     | Yes                 |

---

## 🎯 AI Model Outputs (Decisions & Predictions)

| Output               | Description                                         | Example Decision                |
|---------------------|-----------------------------------------------------|---------------------------------|
| Workload Scheduling | Move tasks to energy-efficient time slots          | Move backups to 2 AM            |
| Cooling Adjustment  | Control AC based on server & room temperatures     | Reduce cooling at night         |
| Energy Source Choice| Pick between grid and solar/wind energy            | Use solar during daytime        |
| Power Distribution  | Smartly shift loads across zones/servers           | Shift AI load to cool zone      |
| Forecast Demand     | Predict usage spikes or drops                      | Spike at 6 PM → prep cooling    |
| Optimize Cost       | Reduce energy cost without hurting performance     | Turn off lights in empty zones  |

---

## 🧪 Model Training

- **Training Environment**: Google Colab  
- **Language**: Python  
- **Libraries Used**: Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn  
- **Model Type**: Multi-output Decision/Regression Model (e.g., Random Forest or Neural Network)  
- **Learning Strategy**: Continual learning support with incremental updates

---

## 📊 Dataset Creation

To train the model effectively, we generated a dataset combining real-world-inspired synthetic values, including:

- Server logs (CPU usage, power draw, etc.)
- Environmental sensors (temperature, humidity)
- Occupancy simulation
- Time-based patterns (weekdays vs weekends, peak hours)
- AI usage simulation (e.g., workload surges during training jobs)

### Example Dataset Generation Code (Python)
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

rows = []
start_time = datetime(2025, 1, 1, 0, 0)

for i in range(10000):
    current_time = start_time + timedelta(minutes=15*i)
    row = {
        "CPU_Usage": np.random.uniform(20, 95),
        "Internal_Temp": np.random.uniform(25, 35),
        "External_Temp": np.random.uniform(20, 40),
        "External_Humidity": np.random.uniform(30, 80),
        "Power_Draw": np.random.uniform(80, 150),
        "Solar_Wind": np.random.uniform(0, 100),
        "Grid_Price": np.random.uniform(0.1, 0.3),
        "Occupancy": np.random.randint(0, 50),
        "Day": current_time.strftime("%A"),
        "Hour": current_time.hour,
        "Active_Users": np.random.randint(1000, 5000),
        "AI_Task_Load": np.random.choice(["Yes", "No"])
    }
    rows.append(row)

pd.DataFrame(rows).to_csv("dataset.csv", index=False)
```

---

## 📂 Folder Structure (Example)

```
/
├── model_training/
│   ├── dataset.csv
│   ├── train_model.ipynb
│   └── model.pkl
└── README.md
```

---

## 🔮 Future Enhancements

- Predictive maintenance module
- Integration with computer vision for occupancy estimation
- Energy credits and billing optimization
- Visual dashboards for admin decision-making

---

## 📧 Contact

For contributions, questions or collaboration:  
**Antarip Kar**  
📧 antarip.dev@example.com *(Replace with actual)*

---

> ✨ This project showcases how AI can revolutionize energy and task management in modern data centers.
