# ====================================================================
# streamlit_dashboard.py
# ====================================================================

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Delta Lake imports
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import streamlit as st

from pyspark.sql import SparkSession
import streamlit as st

@st.cache_resource
def init_spark():
    builder = SparkSession.builder \
        .appName("DashboardApp") \
        .master("local[*]") \
        .config("spark.jars.packages",
                "io.delta:delta-core_2.12:2.4.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    spark = builder.getOrCreate()
    return spark


# ====================================================================
# DATA LOADING FUNCTIONS
# ====================================================================
@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_silver_data(symbol=None, hours_back=24):
    """Load silver data from Delta Lake"""
    try:
        spark = init_spark()
        df = spark.read.format("delta").load("/app/storage/silver_stock")
        
        if symbol and symbol != "All":
            df = df.filter(df.symbol == symbol)
        
        # Get recent data
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        df = df.filter(df.timestamp >= cutoff_time)
        
        return df.toPandas()
    except Exception as e:
        st.error(f"Error loading silver data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_gold_data():
    """Load gold KPI data from Delta Lake"""
    try:
        spark = init_spark()
        df = spark.read.format("delta").load("/app/storage/gold_stock")
        return df.toPandas()
    except Exception as e:
        st.error(f"Error loading gold data: {e}")
        return pd.DataFrame()

# ====================================================================
# VIZUALIZATION FUNCTIONS
# ====================================================================
def create_price_chart(df, symbol):
    """Create interactive price chart"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Volume', 'RSI', 'MACD'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price trace
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['price'],
                  name='Price', line=dict(color='#1f77b4', width=2),
                  hovertemplate='Price: $%{y:.2f}<extra></extra>'),
        row=1, col=1
    )
    
    # Volume as bar chart (secondary y-axis)
    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['volume'],
              name='Volume', marker_color='rgba(128, 128, 128, 0.5)',
              hovertemplate='Volume: %{y:,}<extra></extra>'),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['rsi'],
                  name='RSI', line=dict(color='#ff7f0e', width=2),
                  hovertemplate='RSI: %{y:.2f}<extra></extra>'),
        row=2, col=1
    )
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['macd'],
                  name='MACD', line=dict(color='#2ca02c', width=2),
                  hovertemplate='MACD: %{y:.4f}<extra></extra>'),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} - Price Analysis",
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price / Volume", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def create_technical_indicators(df):
    """Create technical indicators visualization"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Volatility', 'Price Change Distribution',
                       'RSI Distribution', 'Price Direction')
    )
    
    # Volatility over time
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['volatility'],
                  mode='lines', name='Volatility',
                  line=dict(color='#d62728')),
        row=1, col=1
    )
    
    # Price change distribution
    fig.add_trace(
        go.Histogram(x=df['price_change'], nbinsx=50,
                    name='Price Changes', marker_color='#9467bd'),
        row=1, col=2
    )
    
    # RSI distribution
    fig.add_trace(
        go.Histogram(x=df['rsi'], nbinsx=30,
                    name='RSI', marker_color='#8c564b'),
        row=2, col=1
    )
    
    # Price direction pie chart
    direction_counts = df['price_direction'].value_counts()
    fig.add_trace(
        go.Pie(labels=['Down', 'Up'], values=direction_counts.values,
              name='Price Direction', marker_colors=['#ff6b6b', '#51cf66']),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    
    return fig

def create_kpi_dashboard(gold_df):
    """Create KPI dashboard from gold data"""
    if gold_df.empty:
        st.info("No gold data available yet. Waiting for pipeline to process...")
        return
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = gold_df['avg_price'].mean()
        st.metric("Avg Price", f"${avg_price:,.2f}")
    
    with col2:
        total_volume = gold_df['total_volume'].sum()
        st.metric("Total Volume", f"{total_volume:,.0f}")
    
    with col3:
        avg_volatility = gold_df['avg_volatility'].mean()
        st.metric("Avg Volatility", f"{avg_volatility:.4f}")
    
    with col4:
        avg_upward = gold_df['pct_upward'].mean()
        st.metric("Avg Upward %", f"{avg_upward:.1f}%")
    
    # Create symbol comparison charts
    fig1 = go.Figure(data=[
        go.Bar(name='Avg Price', x=gold_df['symbol'], y=gold_df['avg_price'],
              marker_color='#1f77b4'),
        go.Bar(name='Avg Volatility', x=gold_df['symbol'], y=gold_df['avg_volatility'],
              marker_color='#ff7f0e', yaxis='y2')
    ])
    
    fig1.update_layout(
        title="Symbol Performance",
        yaxis=dict(title="Avg Price"),
        yaxis2=dict(title="Avg Volatility", overlaying='y', side='right'),
        barmode='group'
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Upward percentage gauge
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_upward,
        title={'text': "Overall Upward Percentage"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': avg_upward
            }
        }
    ))
    
    st.plotly_chart(fig2, use_container_width=True)

def create_correlation_matrix(df):
    """Create correlation matrix heatmap"""
    if df.empty or len(df) < 10:
        return go.Figure()
    
    numeric_cols = ['price', 'volume', 'rsi', 'macd', 'volatility', 
                   'price_change', 'volume_change']
    
    # Ensure columns exist
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(numeric_cols) < 2:
        return go.Figure()
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500
    )
    
    return fig

# ====================================================================
# MAIN STREAMLIT APP
# ====================================================================
def main():
    st.set_page_config(
        page_title="Stock Analytics Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #2ca02c;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2784/2784459.png", 
                 width=100)
        st.markdown("## Filters")
        
        # Time range selector
        hours_back = st.slider(
            "Hours of data to show",
            min_value=1,
            max_value=168,
            value=24,
            help="Select how many hours of historical data to display"
        )
        
        # Symbol selector
        try:
            spark = init_spark()
            symbols_df = spark.read.format("delta").load("/app/storage/silver_stock")
            symbols = ["All"] + [row.symbol for row in symbols_df.select("symbol").distinct().collect()]
            selected_symbol = st.selectbox("Select Symbol", symbols)
        except:
            selected_symbol = "All"
        
        st.markdown("---")
        st.markdown("### Pipeline Status")
        
        # Check pipeline health
        col1, col2 = st.columns(2)
        with col1:
            try:
                bronze_count = spark.read.format("delta").load("/app/storage/bronze_stock").count()
                st.metric("Bronze Records", f"{bronze_count:,}")
            except:
                st.metric("Bronze Records", "N/A")
        
        with col2:
            try:
                silver_count = spark.read.format("delta").load("/app/storage/silver_stock").count()
                st.metric("Silver Records", f"{silver_count:,}")
            except:
                st.metric("Silver Records", "N/A")
        
        st.markdown("---")
        st.markdown("### Data Update")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        **Dashboard Features:**
        - Real-time stock analytics
        - Technical indicators
        - KPI monitoring
        - Correlation analysis
        """)
    
    # Load data
    with st.spinner("Loading data..."):
        silver_data = load_silver_data(selected_symbol if selected_symbol != "All" else None, 
                                      hours_back)
        gold_data = load_gold_data()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Price Analysis", 
        "📈 Technical Indicators", 
        "🏆 KPI Dashboard",
        "🔗 Correlations"
    ])
    
    # Tab 1: Price Analysis
    with tab1:
        st.markdown('<h2 class="section-header">Price & Volume Analysis</h2>', 
                   unsafe_allow_html=True)
        
        if not silver_data.empty:
            fig = create_price_chart(silver_data, 
                                    selected_symbol if selected_symbol != "All" else "All Symbols")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data summary
            with st.expander("View Data Summary"):
                st.dataframe(silver_data.describe(), use_container_width=True)
        else:
            st.warning("No data available for the selected filters. Try adjusting the time range.")
    
    # Tab 2: Technical Indicators
    with tab2:
        st.markdown('<h2 class="section-header">Technical Indicators</h2>', 
                   unsafe_allow_html=True)
        
        if not silver_data.empty:
            fig = create_technical_indicators(silver_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'volatility' in silver_data.columns:
                    current_vol = silver_data['volatility'].iloc[-1] if len(silver_data) > 0 else 0
                    st.metric("Current Volatility", f"{current_vol:.4f}")
            
            with col2:
                if 'rsi' in silver_data.columns:
                    current_rsi = silver_data['rsi'].iloc[-1] if len(silver_data) > 0 else 50
                    st.metric("Current RSI", f"{current_rsi:.2f}")
                    if current_rsi > 70:
                        st.warning("RSI indicates overbought condition")
                    elif current_rsi < 30:
                        st.info("RSI indicates oversold condition")
            
            with col3:
                if 'macd' in silver_data.columns:
                    current_macd = silver_data['macd'].iloc[-1] if len(silver_data) > 0 else 0
                    st.metric("Current MACD", f"{current_macd:.4f}")
        else:
            st.warning("No technical indicator data available.")
    
    # Tab 3: KPI Dashboard
    with tab3:
        st.markdown('<h2 class="section-header">Key Performance Indicators</h2>', 
                   unsafe_allow_html=True)
        create_kpi_dashboard(gold_data)
        
        # Raw gold data
        with st.expander("View Raw KPI Data"):
            if not gold_data.empty:
                st.dataframe(gold_data, use_container_width=True)
    
    # Tab 4: Correlations
    with tab4:
        st.markdown('<h2 class="section-header">Feature Correlations</h2>', 
                   unsafe_allow_html=True)
        
        if not silver_data.empty and len(silver_data) > 10:
            fig = create_correlation_matrix(silver_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            st.markdown("#### Correlation Insights")
            if 'price' in silver_data.columns and 'volume' in silver_data.columns:
                price_vol_corr = silver_data['price'].corr(silver_data['volume'])
                st.write(f"**Price-Volume Correlation**: {price_vol_corr:.3f}")
                
                if abs(price_vol_corr) > 0.5:
                    if price_vol_corr > 0:
                        st.info("Strong positive correlation between price and volume")
                    else:
                        st.info("Strong negative correlation between price and volume")
        else:
            st.warning("Insufficient data for correlation analysis. Need more data points.")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown(
            f"<p style='text-align: center; color: gray;'>"
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"Data source: Kafka → Spark → Delta Lake"
            f"</p>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()