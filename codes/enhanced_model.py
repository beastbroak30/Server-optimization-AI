import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle

@tf.keras.utils.register_keras_serializable(package='Custom')
def custom_mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

@tf.keras.utils.register_keras_serializable(package='Custom')
def custom_mae(y_true, y_pred):
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)

class DataCenterModel:
    def __init__(self, load_previous=False, model_path='trained_datacenter_model.h5'):
        """Initialize the model with option to load previous training"""
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model_path = model_path
        
        # Load previous model if requested
        if load_previous and os.path.exists(model_path):
            print(f"Loading previously trained model from {model_path}")
            self.model = load_model(model_path, custom_objects={'custom_mse': custom_mse, 'custom_mae': custom_mae})
            
    def build_model(self, input_dim):
        # Input layer for 13 features
        input_layer = Input(shape=(input_dim,))
        
        # Shared layers
        x = Dense(256, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Get number of target features from the training data
        if not hasattr(self, 'target_names'):
            raise ValueError("Model must be trained first to determine number of targets")
        
        # Create dynamic output branches
        output_layers = []
        for i, target_name in enumerate(self.target_names):
            branch = Dense(64, activation='relu')(x)
            branch = Dense(32, activation='relu')(branch)
            output = Dense(1, name=f'target_{i}')(branch)
            output_layers.append(output)
        
        # Combine all outputs
        model = Model(
            inputs=input_layer,
            outputs=output_layers
        )
        
        return model
        
    def train(self, data_path, epochs=100, batch_size=32, continue_training=False):
        """Train the model with option to continue training from previous state"""
        print("\n=== Starting Model Training ===")
        print(f"Loading data from: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Dataset loaded with {len(df)} samples")

        # Define features based on your real CSV
        features = [
            'CPU_Usage', 'Internal_Temp', 'External_Temp', 'External_Humidity',
            'Power_Draw', 'Solar_Wind', 'Grid_Price', 'Occupancy',
            'Day', 'Hour', 'Active_Users', 'AI_Task_Load'
        ]
        print("\nFeatures used:", ", ".join(features))

        # Encode 'Day' (categorical) to numeric values
        le = LabelEncoder()
        df['Day'] = le.fit_transform(df['Day'])

        # Define target decision outputs
        targets = [
            'Workload_Scheduling',
            'Cooling_Adjustment',
            'Energy_Source_Choice',
            'Power_Distribution',
            'Forecast_Demand',
            'Optimize_Cost'
        ]
        print("Target variables:", ", ".join(targets))

        # Generate synthetic target values using logic
        y = np.zeros((len(df), len(targets)))

        y[:, 0] = (df['CPU_Usage'] > 80) | (df['Active_Users'] > df['Active_Users'].mean() + df['Active_Users'].std())
        y[:, 1] = (df['CPU_Usage'] > 70) & (df['External_Temp'] > df['External_Temp'].mean())
        y[:, 2] = (df['Solar_Wind'] > df['Solar_Wind'].mean()) & (df['Grid_Price'] > df['Grid_Price'].mean())
        y[:, 3] = (df['Occupancy'] > df['Occupancy'].mean()) | (df['AI_Task_Load'] > df['AI_Task_Load'].mean() + df['AI_Task_Load'].std())

        peak_hours = (df['Hour'] >= 9) & (df['Hour'] <= 17)
        y[:, 4] = peak_hours & (df['CPU_Usage'] > df['CPU_Usage'].mean())

        y[:, 5] = (df['Grid_Price'] > df['Grid_Price'].mean()) & (df['CPU_Usage'] > df['CPU_Usage'].mean())

        y = y.astype(float)

        # Store target names for later
        self.target_names = targets

        X = df[features].values

        print("\n=== Data Preprocessing ===")
        # Scale numerical features
        X_scaled = self.scaler_X.fit_transform(X)

        # Encode categorical 'Day' feature
        le = LabelEncoder()
        df['Day'] = le.fit_transform(df['Day'])

        y_scaled = self.scaler_y.fit_transform(y)
        print("Data scaling completed")

        print("\n=== Model Building ===")
        
        # If continuing training, load previous model first
        if continue_training and self.model is not None:
            print("Continuing training from previous model state")
        else:
            print("Starting fresh training")
            self.model = self.build_model(X_scaled.shape[1])
            
        print("Model built with", len(targets), "outputs")

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['mse'] * len(targets),
            metrics=[['mae'] for _ in range(len(targets))]  # Fixed: Now provides metrics for each output
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        print("\n=== Training Start ===")
        history = self.model.fit(
            X_scaled,
            [y_scaled[:, i] for i in range(len(targets))],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        print("\n=== Training Complete ===")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return history

    
    def predict(self, data):
        print("\n=== Making Predictions ===")
        
        # Ensure all required features are present
        required_features = [
            'CPU_Usage', 'Internal_Temp', 'External_Temp', 'External_Humidity',
            'Power_Draw', 'Solar_Wind', 'Grid_Price', 'Occupancy',
            'Day', 'Hour', 'Active_Users', 'AI_Task_Load'
        ]
        
        # Add any missing features with default values
        for feature in required_features:
            if feature not in data.columns:
                if feature == 'Internal_Temp':
                    data[feature] = data['External_Temp'] - 5  # Estimate internal temp
                elif feature == 'Power_Draw':
                    data[feature] = data['CPU_Usage'] * 1.5  # Estimate power draw
                else:
                    data[feature] = 0  # Default value for other missing features

        # Encode 'Day' feature
        le = LabelEncoder()
        le.fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        data['Day'] = le.transform(data['Day'])
        
        # Ensure columns are in the correct order
        data = data[required_features]
        
        # Scale input data
        data_scaled = self.scaler_X.transform(data.values)
        print("Input data scaled")
        
        # Make predictions
        print("Running model predictions...")
        predictions_scaled = self.model.predict(data_scaled)
        
        # Inverse transform predictions
        predictions = np.column_stack([pred.flatten() for pred in predictions_scaled])
        predictions = self.scaler_y.inverse_transform(predictions)
        print("Predictions processed and scaled back to original range")
        
        return predictions
    
    def interpret_predictions(self, predictions):
        workload, cooling, energy, power, forecast, cost = predictions[0]
        
        print("=== AI Model Decisions ===\n")
        print(f"Workload Scheduling: {workload:.2f} (>0.5 suggests task rescheduling)")
        print(f"Cooling Adjustment: {cooling:.2f} (>0.5 suggests cooling increase)")
        print(f"Energy Source Choice: {energy:.2f} (>0.5 favors renewable sources)")
        print(f"Power Distribution: {power:.2f} (>0.5 suggests load redistribution)")
        print(f"Forecast Demand: {forecast:.2f} (>0.5 indicates expected spike)")
        print(f"Cost Optimization: {cost:.2f} (>0.5 suggests cost-saving actions)\n")
        
        # Provide specific recommendations based on predictions
        print("=== Recommendations ===\n")
        if workload > 0.5:
            print("→ Move backups and non-critical tasks to off-peak hours")
        if cooling > 0.5:
            print("→ Increase cooling capacity in anticipation of load")
        if energy > 0.5:
            print("→ Switch to solar/wind power during peak generation")
        if power > 0.5:
            print("→ Redistribute AI workload to cooler zones")
        if forecast > 0.5:
            print("→ Prepare for usage spike - adjust cooling and power")
        if cost > 0.5:
            print("→ Implement power-saving measures in low-usage areas")
    
    def save_trained_model(self, path=None):
        """Save the trained model and scalers"""
        if path is None:
            path = self.model_path
        
        # Save the model
        self.model.save(path)
        print(f"\nModel saved successfully to '{path}'")
        
        # Save scalers
        scaler_path = path.replace('.h5', '_scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y
            }, f)
        print(f"Scalers saved to '{scaler_path}'")
    
    def load_trained_model(self, path=None, custom_objects=None):
        """Load a previously trained model and its scalers"""
        if path is None:
            path = self.model_path
            
        # Load the model with custom objects if provided
        self.model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        
        # Load scalers
        scaler_path = path.replace('.h5', '_scalers.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.scaler_X = scalers['scaler_X']
                self.scaler_y = scalers['scaler_y']
                
        print(f"\nLoaded model and scalers from '{path}'")
        
        # Store target names
        self.target_names = [
            'Workload_Scheduling',
            'Cooling_Adjustment',
            'Energy_Source_Choice',
            'Power_Distribution',
            'Forecast_Demand',
            'Optimize_Cost'
        ]
        
def main():
    try:
        print("Initializing Data Center Model...")
        model = DataCenterModel()
        
        # Train the model with synthetic data
        data_path = 'data_center_dataset_100000.csv'
        history = model.train(data_path, epochs=100, batch_size=32,continue_training=True)
        
        # Save the trained model
        model.save_trained_model('trained_datacenter_model.h5')
        print("\nModel saved successfully to 'trained_datacenter_model.h5'")
        
        # Make a test prediction with complete feature set
        test_data = pd.DataFrame({
            'CPU_Usage': [90],
            'Internal_Temp': [62],
            'External_Temp': [38],
            'External_Humidity': [60],
            'Power_Draw': [200],
            'Solar_Wind': [85],
            'Grid_Price': [0.92],
            'Occupancy': [8],
            'Day': ['Monday'],
            'Hour': [4],
            'Active_Users': [1500000],
            'AI_Task_Load': [1]
        })
        
        predictions = model.predict(test_data)
        print("\nTest Prediction Results:")
        model.interpret_predictions(predictions)
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{data_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()