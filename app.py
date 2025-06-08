import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Streamlit page setup
st.set_page_config(page_title="Occupancy Estimation App", layout="wide")
st.title("🏢 Occupancy Estimation App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Occupancy_Estimation.csv")
    return df

df = load_data()

# Load model
model = joblib.load("model_logistic.pkl")

# Navigation
pages = ["Dataset View", "Summary", "EDA", "Predict Occupancy"]
page = st.sidebar.radio("📊 Navigation\nChoose Page", pages)

# --- 1. Dataset View ---
if page == "Dataset View":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

# --- 2. Summary Statistics ---
elif page == "Summary":
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

# --- 3. EDA ---
elif page == "EDA":
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("### Room Occupancy Count Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x="Room_Occupancy_Count", ax=ax2)
        st.pyplot(fig2)

    st.write("### S1_Temp vs Room Occupancy Count (Boxplot)")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="Room_Occupancy_Count", y="S1_Temp", ax=ax3)
    st.pyplot(fig3)

# --- 4. Predict Occupancy ---
elif page == "Predict Occupancy":
    st.subheader("🔍 Predict Occupancy")
    st.write("Enter Sensor Readings:")

    # Input fields
    s1_temp = st.number_input("S1_Temp", value=25.0)
    s2_temp = st.number_input("S2_Temp", value=24.8)
    s3_temp = st.number_input("S3_Temp", value=24.6)
    s4_temp = st.number_input("S4_Temp", value=25.4)

    s1_light = st.number_input("S1_Light", value=120)
    s2_light = st.number_input("S2_Light", value=35)
    s3_light = st.number_input("S3_Light", value=54)
    s4_light = st.number_input("S4_Light", value=43)

    s1_sound = st.number_input("S1_Sound", value=0.10)
    s2_sound = st.number_input("S2_Sound", value=0.12)
    s3_sound = st.number_input("S3_Sound", value=0.15)
    s4_sound = st.number_input("S4_Sound", value=0.18)

    s6_pir = st.selectbox("S6_PIR", [0, 1])
    s7_pir = st.selectbox("S7_PIR", [0, 1])

    # Prepare input
    input_df = pd.DataFrame([[
        s1_temp, s2_temp, s3_temp, s4_temp,
        s1_light, s2_light, s3_light, s4_light,
        s1_sound, s2_sound, s3_sound, s4_sound,
        s6_pir, s7_pir
    ]], columns=[
        'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
        'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
        'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
        'S6_PIR', 'S7_PIR'
    ])

    # Predict
    if st.button("🔮 Predict"):
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"📌 Predicted Room Occupancy Count: **{prediction}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
