import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("Occupancy_Estimation.csv")

# ✅ Print available columns
print("📄 Available columns in dataset:")
print(df.columns.tolist())

# ✅ Feature columns
features = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
            'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
            'S6_PIR', 'S7_PIR']  # S5_CO2 वगळले आहे

X = df[features]

# ✅ Target column
y = df['Room_Occupancy_Count']  # ← हे योग्य नाव आहे

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model_logistic.pkl")
print("✅ Model trained and saved as model_logistic.pkl")
