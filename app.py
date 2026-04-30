import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        min-height: 100vh;
    }
    
    /* Header styling with gradient */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    /* Subheader gradient */
    h2 {
        color: #000000 !important;
        font-weight: bold;
    }
    
    h3 {
        color: #000000 !important;
    }
    
    p, span, div {
        color: #000000 !important;
    }
    
    /* Sidebar gradient */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card-like sections */
    .stMarkdown {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
        color: #000000;
    }
    
    /* Button styling with gradient */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    
    /* Slider styling */
    .stSlider {
        padding: 10px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.9);
        color: #000000;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(90deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #84fab0;
        color: #000000 !important;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='main-header'>
    <h1> Employee Salary Prediction System</h1>
    <p style='font-size: 16px; opacity: 0.9;'>AI-Powered Salary Prediction & Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_employee_data.csv")

data = load_data()

st.sidebar.markdown("""
<div style='background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h2 style='color: white; margin: 0;'>⚙️ Employee Details</h2>
    <p style='color: rgba(255,255,255,0.8); margin: 5px 0 0 0;'>Adjust the sliders to predict salary</p>
</div>
""", unsafe_allow_html=True)

age = st.sidebar.slider("Age", 18, 60, 25)
experience = st.sidebar.slider("Years of Experience", 0, 40, 2)
projects = st.sidebar.slider("Projects Completed", 0, 50, 5)
rating = st.sidebar.slider("Performance Rating", 1, 5, 3)
hours = st.sidebar.slider("Weekly Work Hours", 20, 80, 40)

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

input_data = np.array([[age, experience, projects, rating, hours]])

if st.sidebar.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f" Predicted Salary: ₹ {round(prediction, 2)}")

st.markdown("""
<div style='background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; margin: 20px 0;'>
    <h2 style='color: white; margin: 0;'>📊 Data Visualization & Analytics</h2>
    <p style='color: white; margin: 5px 0 0 0; font-size: 14px;'>Graphs update based on your selected features</p>
</div>
""", unsafe_allow_html=True)

# Filter data based on selected features (with wider ranges for better data visibility)
filtered_data = data[
    (data['age'] >= age - 10) & (data['age'] <= age + 10) &
    (data['experience'] >= experience - 10) & (data['experience'] <= experience + 10) &
    (data['projects'] >= projects - 10) & (data['projects'] <= projects + 10) &
    (data['rating'] >= max(1, rating - 2)) & (data['rating'] <= min(5, rating + 2)) &
    (data['hours'] >= hours - 10) & (data['hours'] <= hours + 10)
]

if len(filtered_data) == 0:
    filtered_data = data.copy()
    st.warning(" No exact matches found. Showing all employee data instead.")
else:
    st.success(f" Showing {len(filtered_data)} employees matching your criteria (out of {len(data)} total)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; border-left: 5px solid #667eea;'>
        <h3 style='margin-top: 0;'>Salary Distribution (Filtered)</h3>
    </div>
    """, unsafe_allow_html=True)
    fig, ax = plt.subplots()
    if len(filtered_data) > 0:
        sns.histplot(filtered_data['salary'], kde=True, ax=ax, color='#667eea', bins=15)
        ax.set_title(f"Salary Range: ₹ {filtered_data['salary'].min():.0f} - ₹ {filtered_data['salary'].max():.0f}")
    else:
        ax.text(0.5, 0.5, 'No data matches your criteria', ha='center', va='center')
    ax.set_facecolor('#f0f0f0')
    st.pyplot(fig)

with col2:
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; border-left: 5px solid #764ba2;'>
        <h3 style='margin-top: 0;'>Experience vs Salary (Filtered)</h3>
    </div>
    """, unsafe_allow_html=True)
    fig, ax = plt.subplots()
    if len(filtered_data) > 0:
        sns.scatterplot(x='experience', y='salary', data=filtered_data, ax=ax, color='#764ba2', s=100)
        ax.set_title(f"Experience Range: {filtered_data['experience'].min()}-{filtered_data['experience'].max()} years")
    else:
        ax.text(0.5, 0.5, 'No data matches your criteria', ha='center', va='center')
    ax.set_facecolor('#f0f0f0')
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; border-left: 5px solid #f5576c;'>
        <h3 style='margin-top: 0;'>Performance Rating vs Salary</h3>
    </div>
    """, unsafe_allow_html=True)
    fig, ax = plt.subplots()
    if len(filtered_data) > 0:
        sns.boxplot(x='rating', y='salary', data=filtered_data, ax=ax, hue='rating', legend=False, palette='Set2')
        ax.set_title('Salary by Performance Rating')
    else:
        ax.text(0.5, 0.5, 'No data matches your criteria', ha='center', va='center')
    ax.set_facecolor('#f0f0f0')
    st.pyplot(fig)

with col4:
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; border-left: 5px solid #84fab0;'>
        <h3 style='margin-top: 0;'>Projects vs Salary</h3>
    </div>
    """, unsafe_allow_html=True)
    fig, ax = plt.subplots()
    if len(filtered_data) > 0:
        sns.scatterplot(x='projects', y='salary', data=filtered_data, ax=ax, color='#84fab0', s=100)
        ax.set_title(f"Projects Range: {filtered_data['projects'].min()}-{filtered_data['projects'].max()}")
    else:
        ax.text(0.5, 0.5, 'No data matches your criteria', ha='center', va='center')
    ax.set_facecolor('#f0f0f0')
    st.pyplot(fig)

st.markdown("""
<div style='background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; border-left: 5px solid #f5576c; margin: 20px 0;'>
    <h3 style='margin-top: 0;'> Correlation Heatmap (Filtered Data)</h3>
</div>
""", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 6))
if len(filtered_data) > 1:
    sns.heatmap(filtered_data.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
else:
    ax.text(0.5, 0.5, 'Not enough data for correlation analysis', ha='center', va='center')
ax.set_facecolor('#f0f0f0')
st.pyplot(fig)

st.markdown("""
<div style='margin: 20px 0;'>
    <p style='font-size: 14px; color: #666;'>✨ Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)
if st.checkbox("Show Feature Importance"):
    try:
        importance = model.coef_
        features = ['Age', 'Experience', 'Projects', 'Rating', 'Hours']
        
        fig, ax = plt.subplots()
        ax.barh(features, importance)
        st.pyplot(fig)
    except:
        st.warning("Model does not support feature importance.")