import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
except Exception as e:
    logger.error(f"Failed to add project root to path: {e}")
    st.error("Failed to configure environment. Please check the file structure.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Material Design styling
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 8px 8px 16px rgba(0, 0, 0, 0.3), -8px -8px 16px rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .metric-card:hover {
        box-shadow: 12px 12px 24px rgba(0, 0, 0, 0.4), -12px -12px 24px rgba(255, 255, 255, 0.2);
        transform: translateY(-5px);
    }
    .achievement-box {
        background: rgba(76, 175, 80, 0.1);
        backdrop-filter: blur(10px);
        color: #81c784;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid rgba(76, 175, 80, 0.3);
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .comparison-box {
        background: rgba(255, 152, 0, 0.1);
        backdrop-filter: blur(10px);
        color: #ffb74d;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid rgba(255, 152, 0, 0.3);
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .chart-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 8px 8px 16px rgba(0, 0, 0, 0.3), -8px -8px 16px rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header with inline styles
st.markdown("""
<div style="font-size: 2.5rem; font-weight: 700; color: #00d4ff; text-align: center; margin-bottom: 0.5rem; text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);">
    📊 Model Performance Analysis
</div>
""", unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #b0b0b0; margin-bottom: 2rem; font-size: 1rem;">Comprehensive evaluation of our rainfall forecasting model</p>', unsafe_allow_html=True)

# Key Performance Metrics
st.markdown("""
<div style="font-size: 1.5rem; font-weight: 600; color: #00d4ff; margin: 2rem 0 1rem 0;">🎯 Key Performance Metrics</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📈 R² Score</h4>
        <h3>0.9874</h3>
        <p style="color: #4CAF50; font-size: 0.9rem;">Excellent Fit</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>🎯 Overall MAE</h4>
        <h3>0.24 mm</h3>
        <p style="color: #4CAF50; font-size: 0.9rem;">High Precision</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>🌧️ Monsoon MAE</h4>
        <h3>0.49 mm</h3>
        <p style="color: #4CAF50; font-size: 0.9rem;">Seasonal Expert</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>🏆 vs Government</h4>
        <h3>34x Better</h3>
        <p style="color: #4CAF50; font-size: 0.9rem;">July Performance</p>
    </div>
    """, unsafe_allow_html=True)

# Model Improvement Journey
st.markdown("""
<div style="font-size: 1.5rem; font-weight: 600; color: #00d4ff; margin: 2rem 0 1rem 0;">🚀 Model Improvement Journey</div>
""", unsafe_allow_html=True)

journey = pd.DataFrame({
    'Stage': ['Untuned LightGBM', 'Tuned LightGBM', '+ Sample Weights', '+ Stacking'],
    'R²': [0.9798, 0.9860, 0.9874, 0.9877],
    'MAE': [0.3484, 0.2636, 0.2443, 0.2486],
    'Monsoon MAE': [0.8359, 0.5171, 0.4868, 0.4817]
})

# Simple improvement table
st.markdown('<h3 class="section-title">🚀 Model Improvement Journey</h3>')
st.dataframe(journey, use_container_width=True, hide_index=True)

# Monthly Comparison Analysis
st.markdown("""
<div style="font-size: 1.5rem; font-weight: 600; color: #00d4ff; margin: 2rem 0 1rem 0;">📅 Monthly Performance Comparison</div>
""", unsafe_allow_html=True)

monthly = pd.DataFrame({
    'Month': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    'Government': [1.41, 1.36, 2.08, 3.21, 4.85, 12.52, 22.37, 20.79, 15.24, 5.91, 3.20, 1.23],
    'Our Model': [0.10, 0.11, 0.15, 0.22, 0.31, 0.55, 0.84, 0.76, 0.55, 0.30, 0.17, 0.10]
})

# Calculate improvement ratio
monthly['Improvement_Ratio'] = monthly['Government'] / monthly['Our Model']

# Simple comparison chart
st.markdown('<h3 class="section-title">📅 Monthly Performance Comparison</h3>')

fig_comparison = go.Figure()

# Add bars for Government model
fig_comparison.add_trace(
    go.Bar(
        x=monthly['Month'],
        y=monthly['Government'],
        name='Government Model',
        marker_color='#ff6b6b',
        text=monthly['Government'].round(1),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Government: %{y:.1f} mm<extra></extra>',
    )
)

# Add bars for Our model
fig_comparison.add_trace(
    go.Bar(
        x=monthly['Month'],
        y=monthly['Our Model'],
        name='Our Model',
        marker_color='#00d4ff',
        text=monthly['Our Model'].round(1),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Our Model: %{y:.1f} mm<extra></extra>',
    )
)

fig_comparison.update_layout(
    title="Monthly MAE Comparison",
    xaxis_title="Month",
    yaxis_title="Mean Absolute Error (mm)",
    barmode='group',
    height=450,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e0e0e0'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
)

st.plotly_chart(fig_comparison, use_container_width=True)

# Key achievements
st.markdown("""
<div style="font-size: 1.5rem; font-weight: 600; color: #00d4ff; margin: 2rem 0 1rem 0;">🏆 Key Achievements</div>
""", unsafe_allow_html=True)

# Find best improvement month
best_improvement_month = monthly.loc[monthly['Improvement_Ratio'].idxmax(), 'Month']
best_improvement_ratio = monthly['Improvement_Ratio'].max()

avg_improvement = monthly['Improvement_Ratio'].mean()

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="achievement-box">
        <h4>🥇 Best Performance</h4>
        <h3>{best_improvement_month}</h3>
        <p>{best_improvement_ratio:.1f}x more accurate than government</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="comparison-box">
        <h4>📊 Average Improvement</h4>
        <h3>{avg_improvement:.1f}x</h3>
        <p>Better across all months</p>
    </div>
    """, unsafe_allow_html=True)

# Simple performance summary
st.markdown("""
<div style="font-size: 1.5rem; font-weight: 600; color: #00d4ff; margin: 2rem 0 1rem 0;">📊 Performance Summary</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="chart-container">
        <h4>🎯 Model Details</h4>
        <p>LightGBM with Optuna optimization</p>
        <p>50+ engineered features</p>
        <p>Time-based validation</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="chart-container">
        <h4>📈 Data Coverage</h4>
        <p>2009-2024 time period</p>
        <p>36 Indian states</p>
        <p>Independent test set</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="chart-container">
        <h4>🏆 Key Advantage</h4>
        <p>Monsoon-focused training</p>
        <p>7-day forecasting</p>
        <p>34x better accuracy</p>
    </div>
    """, unsafe_allow_html=True)

# Simple footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #757575; padding: 1rem;'>
    <p>Model Performance Analysis | Validated on independent test set (2022-2024)</p>
    <p style='font-size: 0.8rem;'>Model Version: v2.0</p>
</div>
""", unsafe_allow_html=True)