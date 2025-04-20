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
---
## Model Overview
The DataCenterModel is a multi-output neural network designed for optimizing data center operations using deep learning. It utilizes TensorFlow/Keras to make multiple simultaneous predictions for various aspects of data center management.

## Model Architecture
- **Type**: Multi-output Neural Network
- **Framework**: TensorFlow/Keras
- **Architecture Type**: Feed-forward Neural Network with branching outputs

### Layer Structure
1. **Input Layer**
   - Dimensions: (None, 12) - 12 input features
   
2. **Shared Layers**
   - Dense Layer 1: 256 neurons, ReLU activation
   - Batch Normalization
   - Dropout (0.2)
   - Dense Layer 2: 128 neurons, ReLU activation
   - Batch Normalization
   - Dropout (0.2)

3. **Output Branches** (6 parallel branches)
   Each branch contains:
   - Dense Layer: 64 neurons, ReLU activation
   - Dense Layer: 32 neurons, ReLU activation
   - Output Layer: 1 neuron (linear activation)

## Training Parameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Batch Size**: 32
- **Epochs**: 100
- **Validation Split**: 20%
- **Early Stopping**: Yes (patience=10)

## Data Preprocessing
- Feature scaling using StandardScaler
- Label encoding for categorical 'Day' feature
- Target variable scaling using StandardScaler

## Model Features
1. **Continued Training Support**
   - Option to continue training from previous state
   - Model checkpointing capability

2. **Serialization**
   - Model saving/loading functionality
   - Scaler persistence for consistent predictions

3. **Custom Loss Functions**
   - custom_mse: Custom Mean Squared Error
   - custom_mae: Custom Mean Absolute Error

## Usage Guidelines
1. **Initialization**
   ```python
   model = DataCenterModel(load_previous=False)
   ```

2. **Training**
   ```python
   history = model.train(data_path='data_center_dataset.csv', 
                        epochs=100, 
                        batch_size=32)
   ```

3. **Prediction**
   ```python
   predictions = model.predict(input_data)
   model.interpret_predictions(predictions)
   ```

## Decision Thresholds
All outputs use a 0.5 threshold for binary decisions:
- Values > 0.5: Positive action recommended
- Values ≤ 0.5: No action needed

## Performance Considerations
- Uses batch normalization for training stability
- Implements dropout (0.2) for regularization
- Early stopping to prevent overfitting
- Scalable architecture for varying data sizes


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
📧 akantarip30@gmail.com

---

> ✨ This project is a case study given by TKS
