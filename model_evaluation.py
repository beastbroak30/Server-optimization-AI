import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

 

def evaluate_model(dataset):
    print("Loading model and data...")
    
    # Register all custom metrics and loss functions
    @tf.keras.utils.register_keras_serializable(package='Custom')
    def custom_mse(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    @tf.keras.utils.register_keras_serializable(package='Custom')
    def custom_mae(y_true, y_pred):
        return tf.keras.losses.mean_absolute_error(y_true, y_pred)

    # Create metrics instances
    @tf.keras.utils.register_keras_serializable(package='Custom')
    class CustomMSE(tf.keras.metrics.Mean):
        def __init__(self, name='custom_mse', **kwargs):
            super().__init__(name=name, **kwargs)

        def update_state(self, y_true, y_pred, sample_weight=None):
            error = tf.keras.losses.mean_squared_error(y_true, y_pred)
            return super().update_state(error, sample_weight=sample_weight)

    @tf.keras.utils.register_keras_serializable(package='Custom')
    class CustomMAE(tf.keras.metrics.Mean):
        def __init__(self, name='custom_mae', **kwargs):
            super().__init__(name=name, **kwargs)

        def update_state(self, y_true, y_pred, sample_weight=None):
            error = tf.keras.losses.mean_absolute_error(y_true, y_pred)
            return super().update_state(error, sample_weight=sample_weight)
    
    custom_objects = {
        'custom_mse': custom_mse,
        'custom_mae': custom_mae,
        'mse': custom_mse,  # Add standard name mapping
        'mae': custom_mae,  # Add standard name mapping
        'CustomMSE': CustomMSE,
        'CustomMAE': CustomMAE,
    }
    
    try:
        # Load the model with custom objects
        model = load_model('trained_datacenter_model.h5', custom_objects=custom_objects)
        print("Model loaded successfully")
        
        # Load scalers
        with open('trained_datacenter_model_scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        print("Scalers loaded successfully")
        
        # Load and prepare data
        data = pd.read_csv(dataset)
        print(f"Loaded {len(data)} data samples")
        
        # Prepare features
        features = [
            'CPU_Usage', 'Internal_Temp', 'External_Temp', 'External_Humidity',
            'Power_Draw', 'Solar_Wind', 'Grid_Price', 'Occupancy',
            'Hour', 'Active_Users', 'AI_Task_Load'
        ]
        
        # Convert Day to numeric
        day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                      'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        data['Day_Numeric'] = data['Day'].map(day_mapping)
        features.append('Day_Numeric')
        
        X = data[features]
        
        # Scale features using the scaler
        X_scaled = scalers['scaler_X'].transform(X)
        print("Features scaled successfully")
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.predict(X_scaled)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        create_decision_analysis(predictions, data)
        create_feature_analysis(X, predictions)
        create_temporal_analysis(data, predictions)
        
        print("\nEvaluation complete! Generated visualizations:")
        print("1. decision_distributions.png - Distribution of model decisions")
        print("2. decision_correlations.png - Correlation between different decisions")
        print("3. feature_importance_heatmap.png - Feature importance analysis")
        print("4. temporal_patterns.png - Decision patterns over time")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def create_decision_analysis(predictions, data):
    print("Generating decision analysis...")
    # Convert and reshape predictions correctly
    predictions = np.array(predictions)
    if len(predictions.shape) == 3:
        predictions = np.squeeze(predictions)  # Remove extra dimension if present
    predictions = predictions.T  # Transpose to get (samples, decisions) shape
    
    decision_names = [
        'Workload_Scheduling',
        'Cooling_Adjustment',
        'Energy_Source_Choice',
        'Power_Distribution',
        'Forecast_Demand',
        'Cost_Optimization'
    ]
    
    # Plot decision distributions
    plt.figure(figsize=(15, 8))
    for i, name in enumerate(decision_names):
        plt.subplot(2, 3, i+1)
        plt.hist(predictions[:, i], bins=30, alpha=0.7)
        plt.title(f'{name} Distribution')
        plt.xlabel('Decision Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('decision_distributions1.png')
    plt.close()
    
    # Plot decision correlation heatmap
    pred_df = pd.DataFrame(predictions, columns=decision_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pred_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Decision Correlation Matrix')
    plt.tight_layout()
    plt.savefig('decision_correlations1.png')
    plt.close()

def create_feature_analysis(features, predictions):
    print("Analyzing feature importance...")
    # Convert and reshape predictions correctly
    predictions = np.array(predictions)
    if len(predictions.shape) == 3:
        predictions = np.squeeze(predictions)  # Remove extra dimension if present
    predictions = predictions.T  # Transpose to get (samples, decisions) shape
    
    feature_names = features.columns
    
    # Calculate feature-decision correlations
    correlations = np.zeros((len(feature_names), predictions.shape[1]))
    for i, feature in enumerate(feature_names):
        for j in range(predictions.shape[1]):
            corr = np.corrcoef(features[feature], predictions[:, j])[0,1]
            correlations[i,j] = abs(corr)  # Use absolute correlation
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.imshow(correlations, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='|Correlation|')
    plt.xticks(range(predictions.shape[1]), ['WL', 'CL', 'EN', 'PW', 'FC', 'CT'], rotation=45)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title('Feature Importance Heatmap')
    plt.tight_layout()
    plt.savefig('feature_importance_heatmap1.png')
    plt.close()

def create_temporal_analysis(data, predictions):
    print("Creating temporal analysis...")
    # Convert and reshape predictions correctly
    predictions = np.array(predictions)
    if len(predictions.shape) == 3:
        predictions = np.squeeze(predictions)  # Remove extra dimension if present
    predictions = predictions.T  # Transpose to get (samples, decisions) shape
    
    # Analyze decisions by hour
    hourly_decisions = pd.DataFrame({
        'Hour': data['Hour'],
        'Workload': predictions[:, 0],
        'Cooling': predictions[:, 1],
        'Energy': predictions[:, 2],
        'Power': predictions[:, 3],
        'Forecast': predictions[:, 4],
        'Cost': predictions[:, 5]
    })
    
    # Create hourly pattern visualization
    plt.figure(figsize=(15, 8))
    hourly_means = hourly_decisions.groupby('Hour').mean()
    
    for column in hourly_means.columns:
        if column != 'Hour':  # Skip the Hour column when plotting
            plt.plot(hourly_means.index, hourly_means[column], 
                    label=column, marker='o')
    
    plt.title('Decision Patterns by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Decision Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('temporal_patterns1.png')
    plt.close()

if __name__ == "__main__":
    # Set TF_ENABLE_ONEDNN_OPTS=0 before importing TensorFlow
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    evaluate_model('data_center_dataset_120000.csv')