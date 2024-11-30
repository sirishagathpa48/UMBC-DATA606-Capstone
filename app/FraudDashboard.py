#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Streamlit App for Loading and Interacting with Random Forest Model


# In[2]:

import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler

import sklearn
print(sklearn.__version__)
print(st.__version__)
print(pd.__version__)

# Load your trained model and scaler
with open('app/RandomForest_model.pkl', 'rb') as file:
    RandomForest_model = pickle.load(file)

with open('app/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
# Load transaction data for visualization
transaction_data = pd.read_csv("app/EncodedFields_fraudtest.csv")  # Load your dataset here

# Apply custom CSS to enhance visual appeal
st.markdown("""
    <style>
    body {
        background-color: #f3f4f6;
    }
    .stSidebar {
        background-color: #e1e5ea;
    }
    .stApp {
        font-family: Arial, sans-serif;
    }
    .header {
        font-size: 2.0em;
        color: #1f77b4;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# Define a function to preprocess input features
def preprocess_data(data):
    scaled_data = scaler.transform(pd.DataFrame([data]))  # Use the loaded scaler
    return scaled_data

# Streamlit UI Title and Description

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(90deg, #ff8a00, #e52e71, #9c27b0, #673ab7);
        -webkit-background-clip: text;
        color: transparent;
        text-align: center;
        margin-top: 0px; /* Removed top margin */
    }
    </style>
    <h1 class="main-title">Fraud Detection Prediction App</h1>
""", unsafe_allow_html=True)

st.write("Predict fraud transactions and view aggregated statistics across categories, states, and more.")

# Sidebar Filters for visualization
st.sidebar.header("Filter Visualization Data")
selected_category = st.sidebar.selectbox("Select Category", ["All"] + sorted(transaction_data['category'].unique()))
selected_state = st.sidebar.selectbox("Select State", ["All"] + sorted(transaction_data['state'].unique()))

# Button to show visualizations
show_visualizations = st.sidebar.button("Visualize Data")

# Input fields for prediction
st.markdown("<div class='header'>Predict Transaction Fraud</div>", unsafe_allow_html=True)
amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
trans_hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, step=1)
category = st.selectbox("Category", ["Select Option"] + sorted(transaction_data['category'].unique()))
state = st.selectbox("State", ["Select Option"] + sorted(transaction_data['state'].unique()))
trans_dayOfWeek = st.selectbox("Day of the Week", ["Select Option", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
trans_month = st.selectbox("Transaction Month", ["Select Option", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
job = st.selectbox("Job", ["Select Option"] + sorted(transaction_data['job'].unique()))

# Manual encoding for categorical variables
category_map = {
    "health_fitness": 0, "kids_pets": 1, "home": 2, "entertainment": 3, "shopping_pos": 4, "personal_care": 5,
    "misc_pos": 6, "misc_net": 7, "food_dining": 8, "shopping_net": 9, "travel": 10, "gas_transport": 11,
    "grocery_pos": 12, "grocery_net": 13
}

state_map = {
    "CA": 0, "AL": 1, "TX": 2, "KS": 3, "IL": 4, "AR": 5, "NM": 6, "OH": 7, 
    "WV": 8, "SD": 9, "VA": 10, "NY": 11, "UT": 12, "SC": 13, "MO": 14,
    "MN": 15, "FL": 16, "ND": 17, "IA": 18, "MS": 19, "WY": 20, "IN": 21, "CT": 22, "NC": 23,
    "KY": 24, "OR": 25, "PA": 26, "NH": 27, "GA": 28, "NJ": 29, "WI": 30,
    "OK": 31, "NE": 32, "MD": 33, "CO": 34, "MI": 35, "WA": 36, "AK": 37
}

day_of_week_map = {
    "Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, 
    "Saturday": 6
}

month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, 
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

job_map = {
    "Therapist, occupational": 0, "Materials engineer": 1, "Exhibition designer": 2, "Film/video editor": 3,
    "Environmental consultant": 4, "Scientist, audiological": 5, "Licensed conveyancer": 6, "Designer, ceramics/pottery": 7,
    "Systems developer": 8, "Sub": 9, "Financial adviser": 10, "Surveyor, land/geomatics": 11, "Comptroller": 12,
    "Counsellor": 13, "IT trainer": 14, "Quantity surveyor": 15, "Engineer, biomedical": 16, "Commissioning editor": 17,
    "Research scientist (physical sciences)": 18, "Naval architect": 19, "Science writer": 20, "Colour technologist": 21
}

# Prediction button
if (category != "Select Option" and state != "Select Option" and trans_dayOfWeek != "Select Option" and trans_month != "Select Option" and job != "Select Option"):
    category_encoded = category_map[category]
    state_encoded = state_map[state]
    day_of_week_encoded = day_of_week_map[trans_dayOfWeek]
    month_encoded = month_map[trans_month]
    job_encoded = job_map[job]
    input_data = [category_encoded, amt, state_encoded, job_encoded, trans_hour, month_encoded, day_of_week_encoded]
    preprocessed_data = preprocess_data(input_data)
    
    if st.button("Predict"):
        fraud_threshold = 0.5
        fraud_prob = RandomForest_model.predict_proba(preprocessed_data)[0][1]
        
        if fraud_prob >= fraud_threshold:
            st.error("The transaction is predicted to be **FRAUDULENT**!", icon="🚨")
        else:
            st.success("The transaction is predicted to be **NON-FRAUDULENT**.", icon="✅")
else:
    st.warning("Please select an option for each field.", icon="⚠️")

# Visualizations
if show_visualizations:
    st.markdown("<div class='header'>Aggregated Fraud Statistics</div>", unsafe_allow_html=True)

    # Filter data for selected category and state
    filtered_data = transaction_data.copy()
    if selected_category != "All":
        filtered_data = filtered_data[filtered_data['category'] == selected_category]
    if selected_state != "All":
        filtered_data = filtered_data[filtered_data['state'] == selected_state]

    # Aggregated Statistics for Visualizations
    fraud_category_counts = filtered_data.groupby('category')['is_fraud'].mean().reset_index()
    fraud_job_counts = filtered_data.groupby('job')['is_fraud'].mean().reset_index()
    fraud_hour_trend = filtered_data.groupby('trans_hour')['is_fraud'].mean().reset_index()
    fraud_dayOfWeek_trend = filtered_data.groupby('trans_dayOfWeek')['is_fraud'].mean().reset_index()
    location_data = filtered_data.groupby(['state', 'lat', 'long']).size().reset_index(name='Fraud_Count')

    # Define the desired order for days of the week
    day_order = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    fraud_dayOfWeek_trend['trans_dayOfWeek'] = pd.Categorical(fraud_dayOfWeek_trend['trans_dayOfWeek'], categories=day_order, ordered=True)
    fraud_dayOfWeek_trend = fraud_dayOfWeek_trend.sort_values('trans_dayOfWeek')

    # Bar Chart: Fraud Percentage by Category with colors
    fig1 = px.bar(fraud_category_counts, x="category", y="is_fraud", title="Fraud Percentage by Category", labels={"is_fraud": "Fraud Rate"}, color="category")
    st.plotly_chart(fig1)

    # Bar Chart: Fraud Percentage by Job with colors
    fig2 = px.bar(fraud_job_counts, x="job", y="is_fraud", title="Fraud Percentage by Job", labels={"is_fraud": "Fraud Rate"}, color="job")
    st.plotly_chart(fig2)

    # Line Chart: Fraud Percentage Over Time with color styling
    fig3 = px.line(fraud_hour_trend, x="trans_hour", y="is_fraud", title="Fraud Percentage Over Time", labels={"is_fraud": "Fraud Rate"}, line_shape="spline")
    fig3.update_traces(line=dict(color="#FF5733"))
    st.plotly_chart(fig3)

    # Line Chart: Fraud Percentage Over Day Of Week
    fig4 = px.line(fraud_dayOfWeek_trend, x="trans_dayOfWeek", y="is_fraud", title="Fraud Percentage Over DayOfWeek", labels={"is_fraud": "Fraud Rate"}, category_orders={"trans_dayOfWeek": day_order}, line_shape="spline")
    fig4.update_traces(line=dict(color="#1f77b4"))
    st.plotly_chart(fig4)

    # Map Visualization with a color scale for Fraud Count
    fig5 = px.scatter_mapbox(location_data, lat="lat", lon="long", size="Fraud_Count", color="Fraud_Count", color_continuous_scale=px.colors.sequential.Plasma,
                             hover_name="state", zoom=3, mapbox_style="carto-positron",
                             title="Fraudulent Transactions by State")
    st.plotly_chart(fig5)
elif show_visualizations:
    st.info("Please select a category or state from the sidebar to view visualizations.")
