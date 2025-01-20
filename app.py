import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI
from streamlit_chat import message

# Load the trained model and preprocessing tools
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

# Feature names and their descriptions
feature_descriptions = {
    'XAU BGNL': 'Gold Price in Bulgarian Lev',
    'ECSURPUS': 'Eurozone Surplus in USD',
    'BDIY': 'Baltic Dry Index Yearly',
    'CRY': 'Cryptocurrency Index',
    'DXY': 'US Dollar Index',
    'JPY': 'Japanese Yen Exchange Rate',
    'GBP': 'British Pound Exchange Rate',
    'Cl1': 'Crude Oil Futures (Front Month)',
    'VIX': 'Volatility Index (S&P 500)',
    'USGG30YR': 'US 30-Year Treasury Yield',
    'GT10': 'Global 10-Year Bond Yield',
    'USGG2YR': 'US 2-Year Treasury Yield',
    'USGG3M': 'US 3-Month Treasury Yield',
    'US0001M': 'US 1-Month Treasury Yield',
    'GTDEM30Y': 'Germany 30-Year Bond Yield',
    'GTDEM10Y': 'Germany 10-Year Bond Yield',
    'GTDEM2Y': 'Germany 2-Year Bond Yield',
    'EONIA': 'Euro Overnight Index Average',
    'GTITL30YR': 'Italy 30-Year Bond Yield',
    'GTITL10YR': 'Italy 10-Year Bond Yield',
    'GTITL2YR': 'Italy 2-Year Bond Yield',
    'GTJPY30YR': 'Japan 30-Year Bond Yield',
    'GTJPY10YR': 'Japan 10-Year Bond Yield',
    'GTJPY2YR': 'Japan 2-Year Bond Yield',
    'GTGBP30Y': 'UK 30-Year Bond Yield',
    'GTGBP20Y': 'UK 20-Year Bond Yield',
    'GTGBP2Y': 'UK 2-Year Bond Yield',
    'LUMSTRUU': 'Lumestra US Treasury Index',
    'LMBITR': 'Lumestra Bitcoin Index',
    'LUACTRUU': 'Lumestra US Corporate Bond Index',
    'LF98TRUU': 'Lumestra US High Yield Bond Index',
    'LG30TRUU': 'Lumestra Global 30-Year Bond Index',
    'LP01TREU': 'Lumestra Eurozone Bond Index',
    'EMUSTRUU': 'Emerging Markets US Treasury Index',
    'LF94TRUU': 'Lumestra US Investment Grade Bond Index',
    'MXUS': 'MSCI USA Index',
    'MXEU': 'MSCI Europe Index',
    'MXJP': 'MSCI Japan Index',
    'MXBR': 'MSCI Brazil Index',
    'MXRU': 'MSCI Russia Index',
    'MXIN': 'MSCI India Index',
    'MXCN': 'MSCI China Index'
}

# Streamlit App
st.title("Market Crash Prediction System")
st.write("Provide financial market data to predict potential market crashes and receive investment strategy recommendations.")

# Loading the Excel file
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    data['Date'] = pd.to_datetime(data['Date']).dt.date  
    data.set_index('Date', inplace=True)  
    return data

file_path = "data_1.xlsx" 
data = load_data(file_path)

# Dropdown to select an example
example_names = data.index.tolist()  # the list of dates
selected_example = st.sidebar.selectbox("Select a date", example_names)

# Sidebar inputs
inputs = {}
for feature, description in feature_descriptions.items():
    # showing feature name with description 
    label = f"{description} ({feature})"
    inputs[feature] = st.sidebar.number_input(label, value=data.loc[selected_example, feature])


input_data = pd.DataFrame([inputs])

if st.button("Predict"):
    try:
        # Preprocessing the input
        input_imputed = pd.DataFrame(imputer.transform(input_data), columns=feature_descriptions.keys())
        input_scaled = pd.DataFrame(scaler.transform(input_imputed), columns=feature_descriptions.keys())

        # the prediction
        probabilities = model.predict_proba(input_scaled)[:, 1]
        prediction = (probabilities >= 0.4).astype(int)

        # Displaying results
        st.write(f"**Crash Probability:** {probabilities[0]:.2f}")
        if prediction[0] == 1:
            st.error("Potential market crash detected! Take necessary precautions.")
        else:
            st.success("No market crash detected. Conditions appear stable.")

        # Generating Investment Strategy Recommendations using an LLM call
        st.subheader("Investment Strategy Recommendations")
        if prediction[0] == 1:
            prompt = f"""
            For a market with these values {input_imputed}, Based on a predicted market crash with a probability of {probabilities[0]:.2f}, provide a detailed investment strategy to minimize losses. 
            Include suggestions for asset allocation, safe-haven assets, hedging strategies, and any other relevant recommendations.
            """
        else:
            prompt = f"""
            For a market with these values {input_imputed}, Based on a predicted stable market condition with a crash probability of {probabilities[0]:.2f}, provide a detailed investment strategy to maximize returns. 
            Include suggestions for high-growth assets, emerging markets, and any other relevant recommendations.
            """

        client_1 = OpenAI(api_key="sk-525486eb3f874d12b11cad8af45407e2", base_url="https://api.deepseek.com")
        deepseek_response = client_1.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a financial expert providing investment strategy recommendations."},
                {"role": "user", "content": prompt},
            ],
            stream=True 
        )

        response_container = st.empty()
        full_response = ""
        for chunk in deepseek_response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_container.markdown(full_response)

    except Exception as e:
        st.error(f"An error occurred: {e}")
