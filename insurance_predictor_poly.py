import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the data
x_data = pd.read_csv('x.csv')
y_data = pd.read_csv('y.csv')

# Features used for prediction
X = x_data[['age', 'sex', 'bmi', 'smoker', 'children']]
y = y_data['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply polynomial transformation to the features (degree = 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Scale the features after polynomial transformation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Train a Ridge regression model (you can change this to LinearRegression if desired)
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Check model coefficients (to understand the learning behavior)
print("Model coefficients:", model.coef_)

# Check predicted vs actual values for train and test sets
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Print Predicted vs Actual for the first 10 samples in the train set
print(f"Predicted vs Actual (Train Set):")
for actual, pred in zip(y_train[:10], y_train_pred[:10]):
    print(f"Actual: {actual}, Predicted: {pred}")

# Tkinter-based GUI for prediction
def predict_charges():
    try:
        # Get user inputs from the GUI
        age = float(age_entry.get())
        sex = float(sex_var.get())
        bmi = float(bmi_entry.get())
        smoker = float(smoker_var.get())
        children = float(children_entry.get())

        # Prepare the input for prediction (scaling and polynomial transformation)
        inputs = [[age, sex, bmi, smoker, children]]
        inputs_poly = poly.transform(inputs)
        inputs_scaled = scaler.transform(inputs_poly)

        # Debugging: Check the transformed and scaled input features
        print("Transformed and scaled input features:", inputs_scaled)

        # Make the prediction
        #prediction = model.predict(inputs_scaled)[0]

        # Clamp the prediction to avoid zero or negative charges
        #if prediction < 0:
            ##prediction = 0

        # Display the result
        result_label.config(text=f"Predicted Charges: ${pred:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Create the Tkinter window
root = tk.Tk()
root.title("Insurance Charges Predictor (Polynomial Regression with Regularization)")
root.geometry("400x400")

# Labels and entry widgets for user input
ttk.Label(root, text="Age").grid(row=0, column=0, pady=5, padx=5)
age_entry = ttk.Entry(root)
age_entry.grid(row=0, column=1, pady=5, padx=5)

ttk.Label(root, text="Sex (1: Male, 2: Female)").grid(row=1, column=0, pady=5, padx=5)
sex_var = tk.StringVar(value="1")
sex_combobox = ttk.Combobox(root, textvariable=sex_var, values=["1", "2"])
sex_combobox.grid(row=1, column=1, pady=5, padx=5)

ttk.Label(root, text="BMI").grid(row=2, column=0, pady=5, padx=5)
bmi_entry = ttk.Entry(root)
bmi_entry.grid(row=2, column=1, pady=5, padx=5)

ttk.Label(root, text="Smoker (1: Yes, 2: No)").grid(row=3, column=0, pady=5, padx=5)
smoker_var = tk.StringVar(value="2")
smoker_combobox = ttk.Combobox(root, textvariable=smoker_var, values=["1", "2"])
smoker_combobox.grid(row=3, column=1, pady=5, padx=5)

ttk.Label(root, text="Children").grid(row=4, column=0, pady=5, padx=5)
children_entry = ttk.Entry(root)
children_entry.grid(row=4, column=1, pady=5, padx=5)

# Predict button to trigger the prediction
predict_button = ttk.Button(root, text="Predict Charges", command=predict_charges)
predict_button.grid(row=5, column=0, columnspan=2, pady=10)

# Result label to display the predicted charges
result_label = ttk.Label(root, text="Predicted Charges: ", font=("Arial", 12))
result_label.grid(row=6, column=0, columnspan=2, pady=10)

# Start the Tkinter main loop
root.mainloop()
