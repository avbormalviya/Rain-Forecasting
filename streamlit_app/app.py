import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import API client
try:
    from api_client import get_forecast, get_available_states, check_api_health
except ImportError as e:
    logger.error(f"Failed to import API client: {e}")
    st.error("Failed to load API client. Please check the dependencies.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Rainfall Forecast System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Material Design styling
st.markdown("""
<style>
    /* Material Design Color Palette */
    .main-header {
        font-size: 2.5rem;
        font-weight: 500;
        color: #1976D2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #FFFFFF;
        color: #1976D2;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border: 1px solid #E0E0E0;
        transition: box-shadow 0.3s ease;
    }
    .metric-card h3 {
        color: #1565C0;
        font-weight: 600;
        margin: 0.5rem 0 0 0;
    }
    .metric-card h4 {
        color: #424242;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
    }
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .forecast-header {
        background: #1976D2;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #FF9800;
        border: 1px solid #FFE0B2;
    }
    .stButton > button {
        background: #1976D2;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 500;
        border-radius: 4px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background: #1565C0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .data-table {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        color: #424242;
        font-weight: 500;
        margin: 2rem 0 1rem 0;
    }
    .subtitle {
        color: #757575;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with Material Design
st.markdown('<h1 class="main-header">🌧️ Rainfall Forecast System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Accurate 7-day rainfall predictions for better planning</p>', unsafe_allow_html=True)

# Helper functions for user-friendly data
def mm_to_inches(mm):
    """Convert millimeters to inches."""
    try:
        return mm * 0.0393701
    except (TypeError, ValueError):
        return 0.0

def get_rainfall_description(mm):
    """Get rainfall intensity description and color."""
    try:
        mm = float(mm)
        if mm < 0.1:
            return "No Rain", "#E0F2F1"
        elif mm < 2.5:
            return "Light Rain", "#E8F5E9"
        elif mm < 10:
            return "Moderate Rain", "#FFF3E0"
        elif mm < 25:
            return "Heavy Rain", "#FFEBEE"
        else:
            return "Very Heavy Rain", "#F3E5F5"
    except (TypeError, ValueError):
        return "No Rain", "#E0F2F1"

def format_rainfall(mm):
    """Format rainfall value with both mm and inches."""
    try:
        mm = float(mm)
        if mm < 0.1:
            return "No rain"
        elif mm < 1:
            return f"{mm:.1f} mm ({mm_to_inches(mm):.2f} inches)"
        else:
            return f"{mm:.1f} mm ({mm_to_inches(mm):.1f} inches)"
    except (TypeError, ValueError):
        return "No rain"

# Initialize session state
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'last_error' not in st.session_state:
    st.session_state.last_error = None

# Sidebar for inputs
with st.sidebar:
    st.markdown("### 📍 Forecast Configuration")
    st.markdown("---")
    
    try:
        # Check API health first
        if not check_api_health():
            st.error("⚠️ API is currently unavailable. Please try again later.")
            st.stop()
            
        available_states = get_available_states()
        if not available_states:
            st.error("No states available from API. Please try again later.")
            st.stop()
            
        state = st.selectbox(
            "🏛️ Select State", 
            available_states,
            help="Choose the Indian state for rainfall prediction"
        )
    except Exception as e:
        logger.error(f"Error loading states: {e}")
        st.error(f"Error loading states: {str(e)}")
        st.stop()
    
    date = st.date_input(
        "📅 Start Date", 
        datetime.now().date(),
        help="Select the starting date for the 7-day forecast"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    st.info("""
    **Model**: LightGBM with Optuna tuning  
    **API**: Live forecasting service  
    **Accuracy**: 34x better than government  
    **MAE**: 0.24mm overall, 0.84mm in July
    """)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("🌐 Connecting to API and generating forecast..."):
            try:
                logger.info(f"Generating forecast for {state} starting {date}")
                predictions = get_forecast(state, str(date))
                
                if not predictions:
                    st.error("No forecast data generated. Please try again.")
                    st.stop()
                
                df = pd.DataFrame(predictions)
                
                # Validate data
                if 'date' not in df.columns or 'predicted_rainfall_mm' not in df.columns:
                    st.error("Invalid forecast data format.")
                    st.stop()
                
                # Convert date column to datetime for better formatting
                df['date'] = pd.to_datetime(df['date'])
                df['day_name'] = df['date'].dt.strftime('%A')
                
                # Store in session state
                st.session_state.forecast_data = df
                st.session_state.last_error = None
                st.success("Forecast generated successfully!")
                
            except ValueError as e:
                error_msg = f"Data error: {str(e)}"
                logger.error(error_msg)
                st.session_state.last_error = error_msg
                st.error(error_msg)
            except Exception as e:
                error_msg = f"Error generating forecast: {str(e)}"
                logger.error(error_msg)
                st.session_state.last_error = error_msg
                st.error(error_msg)

# Display forecast if available
if st.session_state.forecast_data is not None:
    df = st.session_state.forecast_data
    
    # Forecast header
    st.markdown(f"""
    <div class="forecast-header">
        <h2>📍 7-Day Forecast for {state}</h2>
        <p>Starting from {date.strftime('%B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics with user-friendly units
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_rainfall = df['predicted_rainfall_mm'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Rainfall</h4>
            <h3>{format_rainfall(total_rainfall)}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        peak_day = df.loc[df['predicted_rainfall_mm'].idxmax(), 'day_name']
        st.markdown(f"""
        <div class="metric-card">
            <h4>Peak Day</h4>
            <h3>{peak_day}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        peak_rainfall = df['predicted_rainfall_mm'].max()
        st.markdown(f"""
        <div class="metric-card">
            <h4>Peak Rainfall</h4>
            <h3>{format_rainfall(peak_rainfall)}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rainfall = df['predicted_rainfall_mm'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h4>Daily Average</h4>
            <h3>{format_rainfall(avg_rainfall)}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Clean chart section
    st.markdown('<h3 class="section-title">📊 Rainfall Analysis</h3>')
    
    # Simple bar chart for daily rainfall
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=df['date'].dt.strftime('%b %d'),
            y=df['predicted_rainfall_mm'],
            name='Daily Rainfall',
            marker_color='#1976D2',
            text=[format_rainfall(val) for val in df['predicted_rainfall_mm']],
            textposition='outside',
        )
    )
    
    fig.update_layout(
        height=400,
        title="Daily Rainfall Forecast",
        title_font_color="#1976D2",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        xaxis_title_font_color="#424242",
        yaxis_title_font_color="#424242",
        showlegend=False,
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='white',
        font_color="#424242",
        xaxis=dict(
            gridcolor='#E0E0E0',
            tickfont=dict(color="#424242")
        ),
        yaxis=dict(
            gridcolor='#E0E0E0',
            tickfont=dict(color="#424242")
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Clean data table
    st.markdown('<h3 class="section-title">📋 Daily Forecast Details</h3>')
    
    # Format the data for display
    display_df = df.copy()
    display_df['Date'] = display_df['date'].dt.strftime('%B %d')
    display_df['Day'] = display_df['day_name']
    display_df['Rainfall'] = [format_rainfall(val) for val in display_df['predicted_rainfall_mm']]
    
    # Add intensity descriptions
    display_df['Intensity'] = [get_rainfall_description(val)[0] for val in display_df['predicted_rainfall_mm']]
    
    # Clean table display
    st.dataframe(
        display_df[['Date', 'Day', 'Rainfall', 'Intensity']],
        use_container_width=True,
        hide_index=True
    )
    
    # Simple notice section
    st.markdown('<h3 class="section-title">ℹ️ About This Forecast</h3>')
    st.info("""
    **Accuracy Notes:**
    - Day 1 uses real historical data (most accurate)
    - Days 2-7 use predicted values (slightly less accurate)
    - Model performs best during monsoon season
    """)

elif st.session_state.last_error:
    st.error(f"Last error: {st.session_state.last_error}")

# Simple footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #757575; padding: 1rem;'>
    <p>Rainfall Forecast System | Powered by Machine Learning</p>
    <p style='font-size: 0.8rem;'>Model Version: v2.0 | 34x more accurate than government forecasts</p>
</div>
""", unsafe_allow_html=True)