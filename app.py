import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page Setup
st.set_page_config(page_title="Occupancy Estimation App", layout="wide")

# Load dataset
df = pd.read_csv("Occupancy_Estimation.csv")
model = joblib.load("model.pkl")

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
selected = st.sidebar.radio("Choose Page", ["Dataset View", "Summary", "EDA", "Predict Occupancy"])

# Main Title
st.markdown("<h1 style='text-align: center;'>🏢 Occupancy Estimation App</h1>", unsafe_allow_html=True)

# -------------------------------------
# Dataset View Page
# -------------------------------------
if selected == "Dataset View":
    st.markdown("### 📄 Raw Dataset View")
    st.dataframe(df.head(20))

# -------------------------------------
# Summary Page
# -------------------------------------
elif selected == "Summary":
    st.markdown("### 📋 Statistical Summary")
    st.dataframe(df.describe())

# -------------------------------------
# EDA Page
# -------------------------------------
elif selected == "EDA":
    st.markdown("### 📈 Occupancy Count")
    if "Room_Occupancy_Count" in df.columns:
        counts = df["Room_Occupancy_Count"].value_counts().sort_index()

        # Barplot
        fig1, ax1 = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax1)
        ax1.set_title("Room Occupancy Count Distribution")
        ax1.set_xlabel("Occupancy Count")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        # Heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

        # Boxplot
        st.markdown("### 📦 Boxplot of S1_Temp by Occupancy Count")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="Room_Occupancy_Count", y="S1_Temp", ax=ax3)
        ax3.set_title("S1_Temp Distribution by Occupancy Count")
        st.pyplot(fig3)

    else:
        st.error("❌ 'Room_Occupancy_Count' column missing.")

# -------------------------------------
# Prediction Page
# -------------------------------------
elif selected == "Predict Occupancy":
    st.markdown("### 🧮 Enter Sensor Inputs")

    cols = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
            'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
            'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']
    
    inputs = {}
    col1, col2 = st.columns(2)
    for i, col in enumerate(cols):
        with col1 if i % 2 == 0 else col2:
            inputs[col] = st.number_input(f"{col}", value=25.0 if 'Temp' in col else 0.0)

    if st.button("🔍 Predict"):
        try:
            input_df = pd.DataFrame([inputs])
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Predicted Room Occupancy Count: {int(prediction)}")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
