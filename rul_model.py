import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load and prepare the dataset
file_path = "Battery_RUL.csv"  # Update the file path if necessary
battery_data = pd.read_csv(file_path)

# Data Cleaning: Remove invalid rows
battery_data = battery_data[
    (battery_data['Decrement 3.6-3.4V (s)'] >= 0) &
    (battery_data['Time at 4.15V (s)'] >= 0) &
    (battery_data['RUL'] >= 0)
]

# Features and target (exclude 'Cycle_Index')
feature_columns = [
    'Discharge Time (s)', 
    'Decrement 3.6-3.4V (s)', 
    'Max. Voltage Dischar. (V)', 
    'Min. Voltage Charg. (V)', 
    'Time at 4.15V (s)', 
    'Time constant current (s)', 
    'Charging time (s)'
]
X = battery_data[feature_columns]  # Features
y = battery_data['RUL']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor and train it
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit App Interface
st.title("Remaining Useful Life (RUL) Prediction")
st.write("""
### Enter the battery parameters below to predict the Remaining Useful Life (RUL).
""")

# Create input fields for user inputs
def user_input_features():
    discharge_time = st.number_input("Discharge Time (s)", min_value=0.0, step=1.0)
    decrement = st.number_input("Decrement 3.6-3.4V (s)", min_value=0.0, step=1.0)
    max_voltage = st.number_input("Max. Voltage Discharge (V)", min_value=0.0, step=0.01)
    min_voltage = st.number_input("Min. Voltage Charge (V)", min_value=0.0, step=0.01)
    time_at_4_15v = st.number_input("Time at 4.15V (s)", min_value=0.0, step=1.0)
    time_constant_current = st.number_input("Time Constant Current (s)", min_value=0.0, step=1.0)
    charging_time = st.number_input("Charging Time (s)", min_value=0.0, step=1.0)
    
    data = {
        'Discharge Time (s)': discharge_time,
        'Decrement 3.6-3.4V (s)': decrement,
        'Max. Voltage Dischar. (V)': max_voltage,
        'Min. Voltage Charg. (V)': min_voltage,
        'Time at 4.15V (s)': time_at_4_15v,
        'Time constant current (s)': time_constant_current,
        'Charging time (s)': charging_time
    }
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# Ensure all inputs are provided before predicting
if st.button("Predict RUL"):
    if input_df.isnull().any(axis=None):
        st.error("Please fill in all the fields before predicting.")
    else:
        predicted_rul = model.predict(input_df)[0]
        st.write(f"### Predicted Remaining Useful Life (RUL): {predicted_rul:.2f} cycles")
