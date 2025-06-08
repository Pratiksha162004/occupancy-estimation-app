import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Title
st.title("🏢 Occupancy Estimation App")

# Load dataset
df = pd.read_csv("Occupancy_Estimation.csv")

# Load model
model = joblib.load("model.pkl")

# Tabs
tab1, tab2 = st.tabs(["📊 EDA", "🔮 Prediction"])

# ------------------------ EDA TAB ------------------------
with tab1:
    st.subheader("📈 Occupancy Count")
    if "Room_Occupancy_Count" in df.columns:
        occupancy_counts = df["Room_Occupancy_Count"].value_counts()
        fig1, ax1 = plt.subplots()
        sns.barplot(x=occupancy_counts.index, y=occupancy_counts.values, ax=ax1)
        ax1.set_xlabel("Occupancy Count")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Room Occupancy Count Distribution")
        st.pyplot(fig1)
    else:
        st.error("❌ 'Room_Occupancy_Count' column not found in the dataset.")

    st.subheader("🌡️ Temperature Distribution (S1 - S4)")
    temp_cols = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']
    fig2, ax2 = plt.subplots()
    df[temp_cols].plot(kind='box', ax=ax2)
    ax2.set_title("Temperature Sensor Distribution")
    st.pyplot(fig2)

    st.subheader("🟤 CO2 Level vs Room Occupancy")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Room_Occupancy_Count', y='S5_CO2', data=df, ax=ax3)
    ax3.set_title("CO2 Level by Room Occupancy")
    st.pyplot(fig3)

# ------------------------ Prediction TAB ------------------------
with tab2:
    st.subheader("Enter Sensor Values for Prediction")

    S1_Temp = st.number_input("S1_Temp", value=25.00)
    S2_Temp = st.number_input("S2_Temp", value=25.00)
    S3_Temp = st.number_input("S3_Temp", value=25.00)
    S4_Temp = st.number_input("S4_Temp", value=25.00)

    S1_Light = st.number_input("S1_Light", value=100.00)
    S2_Light = st.number_input("S2_Light", value=100.00)
    S3_Light = st.number_input("S3_Light", value=100.00)
    S4_Light = st.number_input("S4_Light", value=100.00)

    S1_Sound = st.number_input("S1_Sound", value=0.10)
    S2_Sound = st.number_input("S2_Sound", value=0.10)
    S3_Sound = st.number_input("S3_Sound", value=0.10)
    S4_Sound = st.number_input("S4_Sound", value=0.10)

    S5_CO2 = st.number_input("S5_CO2", value=400.00)
    S5_CO2_Slope = st.number_input("S5_CO2_Slope", value=0.00)

    S6_PIR = st.number_input("S6_PIR", value=0.00)
    S7_PIR = st.number_input("S7_PIR", value=0.00)

    if st.button("Predict Occupancy"):
        try:
            input_data = pd.DataFrame([[
                S1_Temp, S2_Temp, S3_Temp, S4_Temp,
                S1_Light, S2_Light, S3_Light, S4_Light,
                S1_Sound, S2_Sound, S3_Sound, S4_Sound,
                S5_CO2, S5_CO2_Slope,
                S6_PIR, S7_PIR
            ]], columns=[
                'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
                'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
                'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
                'S5_CO2', 'S5_CO2_Slope',
                'S6_PIR', 'S7_PIR'
            ])

            prediction = model.predict(input_data)[0]
            st.success(f"🏠 Predicted Occupancy Count: {int(prediction)}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
