import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta
import time
import random

# Configure page
st.set_page_config(
    page_title="Market Making Simulator - Edgehog Trading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #f0f2f6;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Risk gauge colors */
    .risk-low { color: #28a745; }
    .risk-medium { color: #ffc107; }
    .risk-high { color: #dc3545; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1rem;
        text-align: center;
        background: #f8f9fa;
        border-radius: 5px;
        font-style: italic;
        color: #6c757d;
    }
    
    /* Success/Warning boxes */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 0.25rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Professional Header for Edgehog Trading
st.markdown("""
<div class="main-header">
    <h1>⚡ EDGEHOG TRADING</h1>
    <p>Options Market Making Simulator | Built by Muchiri Kahwai</p>
    <p style="font-size: 1rem; margin-top: 1rem;">
        <strong>Demonstrating:</strong> Real-time Risk Management • Portfolio Optimization • Electronic Market Making
    </p>
</div>
""", unsafe_allow_html=True)

# Clean Description and Disclaimer Section using Streamlit components
st.markdown("### 📋 About This Application")

st.info("""
**Options Market Making Simulator** - Demonstrating core quantitative finance and risk management skills 
relevant to the **Junior Trader** position at Edgehog Trading.
""")

st.markdown("**Key Features:**")
st.markdown("""
• **Real-time Options Pricing:** Black-Scholes implementation with Greeks calculations  
• **Market Making Interface:** Bid/ask spread management and trade execution  
• **Risk Monitoring:** Portfolio-level risk metrics with visual alerts and limits  
• **Scenario Analysis:** P&L curves and stress testing across market conditions  
• **Position Management:** Real-time portfolio tracking and exposure monitoring  
""")

st.warning("""
**⚠️ Disclaimer:** This application is **not affiliated with or property of Edgehog Trading**. 
It was independently developed by Muchiri Kahwai to showcase quantitative finance and programming skills 
for consideration in the **Junior Trader** role. All market data is simulated for demonstration purposes.
""")

st.caption("*Built with Python, Streamlit, NumPy, Pandas, and Plotly • Source code available upon request*")

st.markdown("---")

# Black-Scholes Functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    if option_type == 'call':
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = theta / 365  # Convert to daily
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% volatility change
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

# Initialize session state
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'current_spot' not in st.session_state:
    st.session_state.current_spot = 100.0
if 'pnl_history' not in st.session_state:
    st.session_state.pnl_history = []

# Enhanced Sidebar
st.sidebar.markdown('<p class="sidebar-header">🎯 Market Parameters</p>', unsafe_allow_html=True)

# Market Data with enhanced styling
st.sidebar.markdown("**📊 Current Market Data**")
spot_price = st.sidebar.number_input("Spot Price ($)", value=100.0, min_value=50.0, max_value=200.0, step=0.1)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=10.0, step=0.1) / 100
implied_vol = st.sidebar.number_input("Implied Volatility (%)", value=20.0, min_value=5.0, max_value=100.0, step=1.0) / 100

st.sidebar.markdown("---")

# Market Making Parameters with enhanced styling
st.sidebar.markdown("**⚙️ Market Making Setup**")
bid_spread = st.sidebar.slider("Bid Spread (bps)", 5, 100, 20, help="Spread below theoretical price") / 10000
ask_spread = st.sidebar.slider("Ask Spread (bps)", 5, 100, 20, help="Spread above theoretical price") / 10000
max_position = st.sidebar.number_input("Max Position Size", value=100, min_value=10, max_value=1000, step=10)
skew_adjustment = st.sidebar.slider("Volatility Skew Adjustment", -5.0, 5.0, 0.0, 0.1, help="Adjust for volatility smile") / 100

st.sidebar.markdown("---")

# Market Status
current_time = datetime.now().strftime("%H:%M:%S")
st.sidebar.markdown("**📈 Market Status**")
st.sidebar.success(f"🟢 **LIVE** - {current_time}")
st.sidebar.metric("Market State", "ACTIVE", delta="Normal Volatility")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Trading", "📊 Risk Dashboard", "💰 P&L Analysis", "📋 Position Manager"])

with tab1:
    st.markdown("### 📈 Live Market Making Interface")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced Options Chain
        st.markdown("#### Options Chain")
        
        strikes = np.arange(80, 121, 5)
        expirations = [7, 14, 30, 60, 90]
        
        col_exp, col_refresh = st.columns([3, 1])
        with col_exp:
            selected_expiry = st.selectbox("📅 Select Expiration (Days)", expirations, index=2)
        with col_refresh:
            if st.button("🔄 Refresh", key="refresh_chain"):
                st.rerun()
        
        time_to_expiry = selected_expiry / 365
        
        # Build enhanced options chain
        chain_data = []
        
        for strike in strikes:
            # Calculate theoretical prices
            call_price = black_scholes_call(spot_price, strike, time_to_expiry, risk_free_rate, implied_vol)
            put_price = black_scholes_put(spot_price, strike, time_to_expiry, risk_free_rate, implied_vol)
            
            # Add market making spreads
            call_bid = call_price * (1 - bid_spread)
            call_ask = call_price * (1 + ask_spread)
            put_bid = put_price * (1 - bid_spread) 
            put_ask = put_price * (1 + ask_spread)
            
            # Calculate Greeks
            call_greeks = calculate_greeks(spot_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'call')
            put_greeks = calculate_greeks(spot_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'put')
            
            # Color coding for moneyness
            if abs(strike - spot_price) <= 2.5:
                moneyness = "🔵 ATM"
            elif strike < spot_price:
                moneyness = "🟢 ITM" if call_price > put_price else "🔴 OTM"
            else:
                moneyness = "🔴 OTM" if call_price > put_price else "🟢 ITM"
            
            chain_data.append({
                'Strike': f"{strike:.0f}",
                'Type': moneyness,
                'Call Bid': f"{call_bid:.2f}",
                'Call Ask': f"{call_ask:.2f}",
                'Call Δ': f"{call_greeks['delta']:.3f}",
                'Put Bid': f"{put_bid:.2f}",
                'Put Ask': f"{put_ask:.2f}",
                'Put Δ': f"{put_greeks['delta']:.3f}"
            })
        
        chain_df = pd.DataFrame(chain_data)
        
        # Style the dataframe
        st.dataframe(
            chain_df, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike": st.column_config.TextColumn("Strike", width="small"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Call Bid": st.column_config.TextColumn("Call Bid", width="medium"),
                "Call Ask": st.column_config.TextColumn("Call Ask", width="medium"),
            }
        )
    
    with col2:
        # Enhanced Quick Trade Panel
        st.markdown("#### ⚡ Quick Trade")
        
        # Trading form in a container
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            
            trade_strike = st.selectbox("📍 Strike", strikes, index=4)
            trade_type = st.selectbox("📊 Option Type", ['Call', 'Put'])
            trade_side = st.selectbox("📈 Side", ['Buy', 'Sell'])
            trade_quantity = st.number_input("📦 Quantity", value=10, min_value=1, max_value=100)
            
            # Calculate estimated price
            option_price = black_scholes_call(spot_price, trade_strike, time_to_expiry, risk_free_rate, implied_vol) if trade_type == 'Call' else black_scholes_put(spot_price, trade_strike, time_to_expiry, risk_free_rate, implied_vol)
            
            if trade_side == 'Buy':
                estimated_price = option_price * (1 + ask_spread)
                price_color = "#dc3545"  # Red for buying (paying ask)
            else:
                estimated_price = option_price * (1 - bid_spread)
                price_color = "#28a745"  # Green for selling (receiving bid)
            
            st.markdown(f"**Estimated Price:** <span style='color: {price_color}; font-weight: bold;'>${estimated_price:.2f}</span>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Execute Trade", type="primary", use_container_width=True):
                # Calculate trade price
                if trade_side == 'Buy':
                    trade_price = option_price * (1 + ask_spread)
                else:
                    trade_price = option_price * (1 - bid_spread)
                
                # Record trade
                trade = {
                    'timestamp': datetime.now(),
                    'strike': trade_strike,
                    'type': trade_type,
                    'side': trade_side,
                    'quantity': trade_quantity if trade_side == 'Buy' else -trade_quantity,
                    'price': trade_price,
                    'expiry_days': selected_expiry
                }
                
                st.session_state.trades.append(trade)
                
                # Success message with styling
                st.markdown(f"""
                <div class="success-box">
                    ✅ <strong>Trade Executed!</strong><br>
                    {trade_side} {trade_quantity} {trade_type} ${trade_strike} @ ${trade_price:.2f}
                </div>
                """, unsafe_allow_html=True)
        
        # Market insights panel
        st.markdown("#### 📊 Market Insights")
        
        with st.container():
            current_iv_percentile = np.random.uniform(20, 80)  # Simulated
            vol_regime = "High" if current_iv_percentile > 70 else "Medium" if current_iv_percentile > 30 else "Low"
            
            st.metric("IV Percentile", f"{current_iv_percentile:.0f}%", delta=f"{vol_regime} Vol")
            st.metric("Bid-Ask Spread", f"{(bid_spread + ask_spread) * 10000:.0f} bps", delta="Tight")
            
            # Quick risk check
            total_delta = sum([
                calculate_greeks(spot_price, trade['strike'], trade['expiry_days']/365, risk_free_rate, implied_vol, trade['type'].lower())['delta'] * trade['quantity']
                for trade in st.session_state.trades
            ])
            
            if abs(total_delta) > 30:
                st.warning(f"⚠️ High Delta Exposure: {total_delta:.1f}")
            else:
                st.success(f"✅ Delta Neutral: {total_delta:.1f}")

with tab2:
    st.markdown("### 🛡️ Risk Monitoring Dashboard")
    
    if st.session_state.trades:
        # Calculate portfolio Greeks with enhanced presentation
        portfolio_delta = 0
        portfolio_gamma = 0
        portfolio_theta = 0
        portfolio_vega = 0
        total_notional = 0
        
        for trade in st.session_state.trades:
            time_to_exp = trade['expiry_days'] / 365
            greeks = calculate_greeks(spot_price, trade['strike'], time_to_exp, risk_free_rate, implied_vol, trade['type'].lower())
            
            portfolio_delta += greeks['delta'] * trade['quantity']
            portfolio_gamma += greeks['gamma'] * trade['quantity']
            portfolio_theta += greeks['theta'] * trade['quantity']
            portfolio_vega += greeks['vega'] * trade['quantity']
            total_notional += abs(trade['quantity'] * trade['price'])
        
        # Enhanced Risk Metrics with color coding
        st.markdown("#### 📊 Portfolio Greeks")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_color = "normal" if abs(portfolio_delta) < 30 else "inverse"
            st.metric(
                "Portfolio Delta", 
                f"{portfolio_delta:.2f}", 
                delta=f"{'⚠️ High' if abs(portfolio_delta) > 50 else '✅ Normal'} Exposure",
                delta_color=delta_color
            )
        
        with col2:
            gamma_color = "normal" if abs(portfolio_gamma) < 5 else "inverse" 
            st.metric(
                "Portfolio Gamma", 
                f"{portfolio_gamma:.3f}", 
                delta=f"{'⚠️ High' if abs(portfolio_gamma) > 10 else '✅ Normal'} Convexity",
                delta_color=gamma_color
            )
        
        with col3:
            theta_color = "inverse" if portfolio_theta < -20 else "normal"
            st.metric(
                "Portfolio Theta", 
                f"${portfolio_theta:.2f}", 
                delta=f"{'📉 High Decay' if portfolio_theta < -50 else '📈 Low Decay'}",
                delta_color=theta_color
            )
        
        with col4:
            vega_color = "normal" if abs(portfolio_vega) < 50 else "inverse"
            st.metric(
                "Portfolio Vega", 
                f"{portfolio_vega:.2f}", 
                delta=f"{'⚠️ High' if abs(portfolio_vega) > 100 else '✅ Normal'} Vol Risk",
                delta_color=vega_color
            )
        
        st.markdown("---")
        
        # Enhanced Risk Limits Monitoring
        st.markdown("#### 🎯 Risk Limits Monitor")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Enhanced Delta limit gauge
            delta_limit = 50
            delta_utilization = min(abs(portfolio_delta) / delta_limit, 1.0)
            
            # Determine gauge color based on utilization
            if delta_utilization < 0.6:
                gauge_color = "lightgreen"
                bar_color = "#28a745"
            elif delta_utilization < 0.8:
                gauge_color = "yellow" 
                bar_color = "#ffc107"
            else:
                gauge_color = "red"
                bar_color = "#dc3545"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = abs(portfolio_delta),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "🔺 Delta Risk", 'font': {'size': 16}},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [None, delta_limit], 'tickwidth': 1},
                    'bar': {'color': bar_color, 'thickness': 0.8},
                    'steps': [
                        {'range': [0, delta_limit*0.6], 'color': "#e8f5e8"},
                        {'range': [delta_limit*0.6, delta_limit*0.8], 'color': "#fff3cd"},
                        {'range': [delta_limit*0.8, delta_limit], 'color': "#f8d7da"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': delta_limit}}))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Enhanced Vega limit gauge
            vega_limit = 100
            vega_utilization = min(abs(portfolio_vega) / vega_limit, 1.0)
            
            if vega_utilization < 0.6:
                gauge_color = "lightgreen"
                bar_color = "#28a745"
            elif vega_utilization < 0.8:
                gauge_color = "yellow"
                bar_color = "#ffc107" 
            else:
                gauge_color = "red"
                bar_color = "#dc3545"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = abs(portfolio_vega),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "📊 Vega Risk", 'font': {'size': 16}},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [None, vega_limit], 'tickwidth': 1},
                    'bar': {'color': bar_color, 'thickness': 0.8},
                    'steps': [
                        {'range': [0, vega_limit*0.6], 'color': "#e8f5e8"},
                        {'range': [vega_limit*0.6, vega_limit*0.8], 'color': "#fff3cd"},
                        {'range': [vega_limit*0.8, vega_limit], 'color': "#f8d7da"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': vega_limit}}))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Portfolio summary metrics
            st.markdown("**📈 Portfolio Summary**")
            
            total_positions = len([t for t in st.session_state.trades if t['quantity'] != 0])
            net_premium = sum([t['quantity'] * t['price'] for t in st.session_state.trades])
            
            st.metric("Active Positions", total_positions)
            st.metric("Net Premium", f"${net_premium:.2f}")
            st.metric("Total Notional", f"${total_notional:.2f}")
            
            # Risk status indicator
            risk_level = "🟢 LOW" if max(delta_utilization, vega_utilization) < 0.6 else "🟡 MEDIUM" if max(delta_utilization, vega_utilization) < 0.8 else "🔴 HIGH"
            st.markdown(f"**Overall Risk:** {risk_level}")
        
        st.markdown("---")
        
        # Enhanced Scenario Analysis
        st.markdown("#### 📈 Scenario Analysis")
        
        spot_scenarios = np.linspace(spot_price * 0.8, spot_price * 1.2, 50)
        pnl_scenarios = []
        
        for scenario_spot in spot_scenarios:
            scenario_pnl = 0
            for trade in st.session_state.trades:
                time_to_exp = trade['expiry_days'] / 365
                if trade['type'] == 'Call':
                    current_price = black_scholes_call(scenario_spot, trade['strike'], time_to_exp, risk_free_rate, implied_vol)
                else:
                    current_price = black_scholes_put(scenario_spot, trade['strike'], time_to_exp, risk_free_rate, implied_vol)
                
                pnl = (current_price - trade['price']) * trade['quantity']
                scenario_pnl += pnl
            
            pnl_scenarios.append(scenario_pnl)
        
        # Create enhanced P&L scenario chart
        fig = go.Figure()
        
        # Color the P&L line based on profit/loss
        colors = ['red' if pnl < 0 else 'green' for pnl in pnl_scenarios]
        
        fig.add_trace(go.Scatter(
            x=spot_scenarios, 
            y=pnl_scenarios, 
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)',
            name='Portfolio P&L',
            hovertemplate='Spot: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
        ))
        
        # Add current spot line
        fig.add_vline(
            x=spot_price, 
            line_dash="dash", 
            line_color="red", 
            line_width=2,
            annotation_text=f"Current Spot: ${spot_price:.2f}",
            annotation_position="top"
        )
        
        # Add break-even line
        fig.add_hline(
            y=0, 
            line_dash="dot", 
            line_color="gray",
            annotation_text="Break-even",
            annotation_position="right"
        )
        
        fig.update_layout(
            title="📊 Portfolio P&L vs Spot Price Movement",
            xaxis_title="Underlying Spot Price ($)",
            yaxis_title="Portfolio P&L ($)",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk scenarios summary
        max_loss = min(pnl_scenarios)
        max_gain = max(pnl_scenarios)
        current_pnl = pnl_scenarios[len(pnl_scenarios)//2]  # Approximate current P&L
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📉 Max Loss", f"${max_loss:.2f}", delta="Worst Case")
        with col2:
            st.metric("📈 Max Gain", f"${max_gain:.2f}", delta="Best Case")
        with col3:
            st.metric("🎯 Current P&L", f"${current_pnl:.2f}", delta="At Current Spot")
        
    else:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>No Active Positions</strong><br>
            Go to the Live Trading tab to start building positions and see risk metrics here.
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.header("P&L Analysis")
    
    if st.session_state.trades:
        # Calculate current P&L
        total_pnl = 0
        trade_pnls = []
        
        for i, trade in enumerate(st.session_state.trades):
            time_to_exp = trade['expiry_days'] / 365
            if trade['type'] == 'Call':
                current_price = black_scholes_call(spot_price, trade['strike'], time_to_exp, risk_free_rate, implied_vol)
            else:
                current_price = black_scholes_put(spot_price, trade['strike'], time_to_exp, risk_free_rate, implied_vol)
            
            trade_pnl = (current_price - trade['price']) * trade['quantity']
            total_pnl += trade_pnl
            
            trade_pnls.append({
                'Trade_ID': i+1,
                'Strike': trade['strike'],
                'Type': trade['type'],
                'Quantity': trade['quantity'],
                'Entry_Price': f"{trade['price']:.2f}",
                'Current_Price': f"{current_price:.2f}",
                'P&L': f"{trade_pnl:.2f}"
            })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total P&L", f"${total_pnl:.2f}", delta=f"{total_pnl:.2f}")
        with col2:
            total_premium = sum([abs(trade['quantity'] * trade['price']) for trade in st.session_state.trades])
            pnl_percentage = (total_pnl / total_premium * 100) if total_premium > 0 else 0
            st.metric("P&L %", f"{pnl_percentage:.2f}%", delta=f"{pnl_percentage:.2f}%")
        with col3:
            st.metric("Total Premium", f"${total_premium:.2f}")
        
        # P&L by Trade
        st.subheader("P&L by Trade")
        pnl_df = pd.DataFrame(trade_pnls)
        st.dataframe(pnl_df, use_container_width=True)
        
        # P&L Chart
        st.subheader("P&L Over Time")
        
        # Simulate time-based P&L (for demo purposes)
        if len(st.session_state.pnl_history) < 50:
            st.session_state.pnl_history.append({
                'timestamp': datetime.now(),
                'pnl': total_pnl,
                'spot': spot_price
            })
        
        if st.session_state.pnl_history:
            pnl_history_df = pd.DataFrame(st.session_state.pnl_history)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                              subplot_titles=('Portfolio P&L', 'Spot Price'))
            
            fig.add_trace(go.Scatter(x=pnl_history_df['timestamp'], y=pnl_history_df['pnl'], 
                                   mode='lines', name='P&L'), row=1, col=1)
            fig.add_trace(go.Scatter(x=pnl_history_df['timestamp'], y=pnl_history_df['spot'], 
                                   mode='lines', name='Spot Price'), row=2, col=1)
            
            fig.update_layout(height=600, title_text="P&L and Market Movement")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No trades executed yet. P&L analysis will appear after trading.")

with tab4:
    st.header("Position Manager")
    
    if st.session_state.trades:
        # Aggregate positions
        positions = {}
        
        for trade in st.session_state.trades:
            key = f"{trade['type']}_{trade['strike']}_{trade['expiry_days']}"
            if key not in positions:
                positions[key] = {
                    'strike': trade['strike'],
                    'type': trade['type'],
                    'expiry_days': trade['expiry_days'],
                    'quantity': 0,
                    'avg_price': 0,
                    'total_premium': 0
                }
            
            positions[key]['quantity'] += trade['quantity']
            positions[key]['total_premium'] += trade['quantity'] * trade['price']
        
        # Calculate average prices
        for pos in positions.values():
            if pos['quantity'] != 0:
                pos['avg_price'] = pos['total_premium'] / pos['quantity']
        
        # Filter out zero positions
        active_positions = {k: v for k, v in positions.items() if v['quantity'] != 0}
        
        if active_positions:
            position_data = []
            
            for pos_id, pos in active_positions.items():
                time_to_exp = pos['expiry_days'] / 365
                if pos['type'] == 'Call':
                    current_price = black_scholes_call(spot_price, pos['strike'], time_to_exp, risk_free_rate, implied_vol)
                else:
                    current_price = black_scholes_put(spot_price, pos['strike'], time_to_exp, risk_free_rate, implied_vol)
                
                greeks = calculate_greeks(spot_price, pos['strike'], time_to_exp, risk_free_rate, implied_vol, pos['type'].lower())
                position_pnl = (current_price - pos['avg_price']) * pos['quantity']
                
                position_data.append({
                    'Position_ID': pos_id,
                    'Strike': pos['strike'],
                    'Type': pos['type'],
                    'Quantity': pos['quantity'],
                    'Avg_Price': f"{pos['avg_price']:.2f}",
                    'Current_Price': f"{current_price:.2f}",
                    'P&L': f"{position_pnl:.2f}",
                    'Delta': f"{greeks['delta'] * pos['quantity']:.2f}",
                    'Gamma': f"{greeks['gamma'] * pos['quantity']:.2f}",
                    'Theta': f"{greeks['theta'] * pos['quantity']:.2f}",
                    'Vega': f"{greeks['vega'] * pos['quantity']:.2f}"
                })
            
            positions_df = pd.DataFrame(position_data)
            st.dataframe(positions_df, use_container_width=True)
            
            # Position closing
            st.subheader("Close Positions")
            
            if st.button("Close All Positions", type="secondary"):
                st.session_state.trades = []
                st.session_state.pnl_history = []
                st.success("All positions closed!")
                st.rerun()
        
        else:
            st.info("No active positions.")
    
    else:
        st.info("No positions to manage. Start trading to see positions here.")

# Real-time updates with enhanced styling
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Market Data", use_container_width=True):
    # Simulate market movement
    st.session_state.current_spot = spot_price + np.random.normal(0, 0.5)
    st.success("Market data refreshed!")
    st.rerun()

# Enhanced Footer
st.markdown("""
<div class="footer">
    <h4>⚡ EDGEHOG TRADING MARKET MAKING SIMULATOR</h4>
    <p><strong>Built by Muchiri Kahwai</strong> | Demonstrating Quantitative Finance & Risk Management Skills</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
        <em>Featuring: Black-Scholes Pricing • Greeks Calculation • Portfolio Risk Management • Real-time P&L Tracking</em>
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem; color: #6c757d;">
        📧 mk@Muchiri.tech | 📱 +1(859)319-6196 | 
        <a href="https://linkedin.com/in/muchiri-kahwai" style="color: #667eea;">LinkedIn</a> | 
        <a href="https://github.com/muchirikahwai" style="color: #667eea;">GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)