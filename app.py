import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Machinery Demand Predictor", page_icon="📊", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        }
        .css-1v0mbdj {padding: 2rem !important;}
        h1 {
            color: #1f77b4;
            text-align: center;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .stDownloadButton>button {
            background-color: #28a745;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
        }
    </style>
""", unsafe_allow_html=True)
model = joblib.load("model.pkl")
model_columns = joblib.load("columns.pkl")

st.title("📊 Machinery Demand Predictor")
st.markdown("Upload your dataset and get AI-powered demand predictions in seconds!")

uploaded_file = st.file_uploader("📁 Upload your CSV file below", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded File Preview")
    st.dataframe(input_df.head(), use_container_width=True)
    input_df_processed = input_df.drop(columns=["Date", "Predicted_Demand"], errors='ignore')
    input_encoded = pd.get_dummies(input_df_processed)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    try:
        predictions = model.predict(input_encoded)
        input_df["Predicted_Demand_New"] = predictions

        st.subheader("📈 Predicted Demand Output")
        st.success("✅ Predictions generated successfully!")
        st.dataframe(input_df[["Predicted_Demand_New"]].head(), use_container_width=True)

        st.download_button("⬇️ Download Full Predictions", input_df.to_csv(index=False), file_name="predictions.csv")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
