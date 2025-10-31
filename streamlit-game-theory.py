import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from datetime import datetime
import math

# Page configuration
st.set_page_config(
    page_title="Relationship Game Theory Analyzer",
    page_icon="💑",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .outcome-matched {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .outcome-engaged {
        background-color: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
    }
    .outcome-complicated {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .outcome-not-matched {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .elo-rating {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .elo-grandmaster {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .elo-master {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .elo-expert {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .elo-intermediate {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    .elo-novice {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'game_history' not in st.session_state:
    st.session_state.game_history = []
if 'round_number' not in st.session_state:
    st.session_state.round_number = 0
if 'elo_a' not in st.session_state:
    st.session_state.elo_a = 1500  # Starting ELO
if 'elo_b' not in st.session_state:
    st.session_state.elo_b = 1500  # Starting ELO
if 'elo_history_a' not in st.session_state:
    st.session_state.elo_history_a = [1500]
if 'elo_history_b' not in st.session_state:
    st.session_state.elo_history_b = [1500]

# YAML Configuration
yaml_config = """
game_classification:
  cooperation_type: "Non-Cooperative (but can transition to Cooperative)"
  action_order: "Simultaneous"
  information_type: "Imperfect Information, Incomplete Information"
  sum_type: "Non-Zero-Sum"
  symmetry: "Asymmetric (becomes Symmetric at equilibrium)"
  dynamics: "Dynamic/Repeated Game"
  
payoff_matrix:
  cooperate_cooperate: {a: 8, b: 8, outcome: "MATCHED"}
  cooperate_defect: {a: 2, b: 9, outcome: "NOT MATCHED"}
  defect_cooperate: {a: 9, b: 2, outcome: "NOT MATCHED"}
  defect_defect: {a: 4, b: 4, outcome: "COMPLICATED"}
  mixed_strategy: {a: 5, b: 6, outcome: "CONFUSED"}
  asymmetric_investment: {a: 7, b: 7, outcome: "ENGAGED"}

elo_tiers:
  novice: {min: 0, max: 1199, description: "Learning Relationship Dynamics"}
  intermediate: {min: 1200, max: 1399, description: "Understanding Partner Needs"}
  expert: {min: 1400, max: 1599, description: "Skilled Relationship Navigator"}
  master: {min: 1600, max: 1799, description: "Masterful Partner"}
  grandmaster: {min: 1800, max: 3000, description: "Relationship Expert"}
"""

config = yaml.safe_load(yaml_config)

# ELO Rating Functions
def calculate_expected_score(elo_a, elo_b):
    """Calculate expected score using ELO formula"""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_current, expected, actual, k_factor=32):
    """Update ELO rating based on game outcome"""
    return elo_current + k_factor * (actual - expected)

def get_elo_tier(elo):
    """Get ELO tier and color"""
    tiers = config['elo_tiers']
    if elo >= tiers['grandmaster']['min']:
        return "Grandmaster", "elo-grandmaster", "👑"
    elif elo >= tiers['master']['min']:
        return "Master", "elo-master", "🏆"
    elif elo >= tiers['expert']['min']:
        return "Expert", "elo-expert", "⭐"
    elif elo >= tiers['intermediate']['min']:
        return "Intermediate", "elo-intermediate", "📈"
    else:
        return "Novice", "elo-novice", "🌱"

def calculate_actual_score(payoff_a, payoff_b):
    """Convert payoffs to ELO scores (0, 0.5, or 1)"""
    if payoff_a > payoff_b:
        return 1.0, 0.0  # Player A wins
    elif payoff_b > payoff_a:
        return 0.0, 1.0  # Player B wins
    else:
        return 0.5, 0.5  # Draw

def calculate_k_factor(elo, rounds_played):
    """Calculate dynamic K-factor based on ELO and experience"""
    # Higher K-factor for new players, lower for experienced
    if rounds_played < 10:
        return 40
    elif elo < 1400:
        return 32
    elif elo < 1600:
        return 24
    else:
        return 16

# Game Theory Functions
def calculate_utility(metrics):
    """Calculate utility from player metrics"""
    return np.mean(list(metrics.values()))

def determine_strategy(utility):
    """Determine if player cooperates or defects"""
    return "Cooperate" if utility >= 5.5 else "Defect"

def calculate_payoff(strategy_a, strategy_b, utility_a, utility_b):
    """Calculate payoffs based on strategies"""
    payoff_matrix = config['payoff_matrix']
    
    if strategy_a == "Cooperate" and strategy_b == "Cooperate":
        return payoff_matrix['cooperate_cooperate']
    elif strategy_a == "Cooperate" and strategy_b == "Defect":
        return payoff_matrix['cooperate_defect']
    elif strategy_a == "Defect" and strategy_b == "Cooperate":
        return payoff_matrix['defect_cooperate']
    elif strategy_a == "Defect" and strategy_b == "Defect":
        return payoff_matrix['defect_defect']
    else:
        avg_utility = (utility_a + utility_b) / 2
        if avg_utility >= 6.5:
            return payoff_matrix['asymmetric_investment']
        return payoff_matrix['mixed_strategy']

def identify_game_type(history):
    """Identify the type of game being played"""
    if len(history) < 3:
        return "Initial Assessment Phase"
    
    recent = history[-5:]
    coop_count = sum(1 for h in recent if h['strategy_a'] == "Cooperate" and h['strategy_b'] == "Cooperate")
    
    last_round = history[-1]
    if last_round['strategy_a'] == "Defect" and last_round['strategy_b'] == "Defect":
        return "Prisoner's Dilemma"
    elif last_round['strategy_a'] != last_round['strategy_b'] and coop_count < 2:
        return "Battle of Sexes"
    elif coop_count >= 3:
        return "Stag Hunt (Trust Building)"
    elif len(history) > 5:
        return "Repeated Game"
    return "Chicken Game (Brinkmanship)"

def calculate_outcome_score(player_a, player_b):
    """Calculate final outcome score (f)"""
    utility_a = calculate_utility(player_a)
    utility_b = calculate_utility(player_b)
    
    prompt_quality = (utility_a + utility_b) / 2 / 10
    response_authenticity = (player_a.get('authenticity_ratio', 5) + player_b.get('reciprocity_level', 5)) / 2 / 10
    resource_balance = 1 - abs(utility_a - utility_b) / 10
    time_investment = (player_a.get('time_investment', 5) + player_b.get('emotional_availability', 5)) / 2 / 10
    
    f = prompt_quality * response_authenticity * resource_balance * time_investment
    
    if f > 0.8:
        return f, "MATCHED", "🎉"
    elif f > 0.6:
        return f, "ENGAGED", "💍"
    elif f > 0.4:
        return f, "COMPLICATED", "⚠️"
    elif f > 0.2:
        return f, "CONFUSED", "🤔"
    else:
        return f, "NOT MATCHED", "💔"

def classify_game_comprehensive():
    """Comprehensive game classification"""
    classification = {
        "Cooperation Type": {
            "Category": "Non-Cooperative → Cooperative Transition",
            "Description": "Starts as non-cooperative (each player maximizes own utility) but can transition to cooperative when Nash equilibrium is reached",
            "Evidence": "Players initially use manipulation tactics, but optimal outcome requires cooperation"
        },
        "Action Order": {
            "Category": "Simultaneous",
            "Description": "Both players make decisions simultaneously without knowing the other's choice",
            "Evidence": "Each player adjusts their strategy based on metrics without real-time knowledge of opponent's exact strategy"
        },
        "Information Availability": {
            "Category": "Imperfect & Incomplete Information",
            "Perfect/Imperfect": "Imperfect - Players don't observe all actions (e.g., true intentions behind 'pseudo-altruism')",
            "Complete/Incomplete": "Incomplete - Players don't know opponent's exact payoff functions or true preferences",
            "Evidence": "Player A may display 'pseudo-altruism' masking true intentions; Player B may use 'strategic refusal' hiding true feelings"
        },
        "Sum Type": {
            "Category": "Non-Zero-Sum",
            "Description": "Total payoffs vary by outcome; both can win (8,8) or both can lose (4,4)",
            "Evidence": "Cooperate-Cooperate yields (8,8)=16 total, while Defect-Defect yields (4,4)=8 total",
            "Pareto Efficiency": "Mutual cooperation (8,8) is Pareto optimal"
        },
        "Symmetry": {
            "Category": "Asymmetric → Symmetric",
            "Description": "Starts asymmetric (different strategies/resources) but becomes symmetric at equilibrium",
            "Evidence": "Player A uses power/network, Player B uses EI/culture. At equilibrium, roles become 'interchangeable'"
        },
        "Dynamics": {
            "Category": "Dynamic/Repeated Game",
            "Description": "Multi-stage game where past actions influence future decisions",
            "Evidence": "Game history affects trust levels, enabling tit-for-tat strategies and reputation effects"
        }
    }
    return classification

# Header
st.title("💑 Relationship Game Theory Analyzer with ELO Rating System")
st.markdown("### Nash Equilibrium & Strategic Interaction Analysis")

# ELO Display Section
st.markdown("---")
st.markdown("## 🏆 ELO Rating System")

elo_col1, elo_col2, elo_col3 = st.columns(3)

with elo_col1:
    tier_a, class_a, emoji_a = get_elo_tier(st.session_state.elo_a)
    st.markdown(f"""
    <div class="elo-rating {class_a}">
        {emoji_a} Player A<br>
        {int(st.session_state.elo_a)}<br>
        <small>{tier_a}</small>
    </div>
    """, unsafe_allow_html=True)
    
with elo_col2:
    elo_diff = st.session_state.elo_a - st.session_state.elo_b
    expected_a = calculate_expected_score(st.session_state.elo_a, st.session_state.elo_b)
    st.metric("ELO Difference", f"{int(abs(elo_diff))}", 
              delta=f"Player {'A' if elo_diff > 0 else 'B'} favored")
    st.metric("Win Probability A", f"{expected_a*100:.1f}%")
    st.metric("Win Probability B", f"{(1-expected_a)*100:.1f}%")

with elo_col3:
    tier_b, class_b, emoji_b = get_elo_tier(st.session_state.elo_b)
    st.markdown(f"""
    <div class="elo-rating {class_b}">
        {emoji_b} Player B<br>
        {int(st.session_state.elo_b)}<br>
        <small>{tier_b}</small>
    </div>
    """, unsafe_allow_html=True)

# ELO Tier Explanation
with st.expander("📊 ELO Rating System Explanation"):
    st.markdown("""
    ### How ELO Works in Relationship Game Theory
    
    **ELO Rating System** measures player skill in strategic relationship dynamics:
    
    - **Starting Rating:** 1500 (Neutral)
    - **Rating Changes:** Based on actual vs. expected performance
    - **K-Factor:** Determines rating volatility (higher for new players)
    
    **Tiers:**
    - 🌱 **Novice (0-1199):** Learning relationship dynamics
    - 📈 **Intermediate (1200-1399):** Understanding partner needs
    - ⭐ **Expert (1400-1599):** Skilled relationship navigator
    - 🏆 **Master (1600-1799):** Masterful partner
    - 👑 **Grandmaster (1800+):** Relationship expert
    
    **Rating Updates:**
    - Win (higher payoff): Rating increases
    - Loss (lower payoff): Rating decreases
    - Draw (equal payoff): Small adjustments based on expected outcome
    - Upset victories (low-rated beats high-rated): Larger rating changes
    """)

st.markdown("---")

# Game Theory Classification Section
with st.expander("📊 COMPREHENSIVE GAME THEORY CLASSIFICATION", expanded=False):
    classification = classify_game_comprehensive()
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("#### 🤝 Cooperation Type")
        st.info(f"**{classification['Cooperation Type']['Category']}**")
        st.write(classification['Cooperation Type']['Description'])
        st.caption(f"Evidence: {classification['Cooperation Type']['Evidence']}")
        
        st.markdown("#### ⏱️ Action Order")
        st.info(f"**{classification['Action Order']['Category']}**")
        st.write(classification['Action Order']['Description'])
        st.caption(f"Evidence: {classification['Action Order']['Evidence']}")
    
    with cols[1]:
        st.markdown("#### 🔍 Information Type")
        st.info(f"**{classification['Information Availability']['Category']}**")
        st.write(f"**Perfect/Imperfect:** {classification['Information Availability']['Perfect/Imperfect']}")
        st.write(f"**Complete/Incomplete:** {classification['Information Availability']['Complete/Incomplete']}")
        st.caption(f"Evidence: {classification['Information Availability']['Evidence']}")
        
        st.markdown("#### 💰 Sum Type")
        st.info(f"**{classification['Sum Type']['Category']}**")
        st.write(classification['Sum Type']['Description'])
        st.caption(f"Evidence: {classification['Sum Type']['Evidence']}")
    
    with cols[2]:
        st.markdown("#### ⚖️ Symmetry")
        st.info(f"**{classification['Symmetry']['Category']}**")
        st.write(classification['Symmetry']['Description'])
        st.caption(f"Evidence: {classification['Symmetry']['Evidence']}")
        
        st.markdown("#### 🔄 Dynamics")
        st.info(f"**{classification['Dynamics']['Category']}**")
        st.write(classification['Dynamics']['Description'])
        st.caption(f"Evidence: {classification['Dynamics']['Evidence']}")

st.markdown("---")

# Player Input Sections
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### 🛡️ Player A: Power Strategist")
    st.markdown(f"*ELO: {int(st.session_state.elo_a)} ({tier_a}) {emoji_a}*")
    
    player_a = {
        'prompt_frequency': st.slider("Prompt Frequency", 0, 10, 5, key='a_pf'),
        'message_depth': st.slider("Message Depth", 0, 10, 5, key='a_md'),
        'vulnerability_level': st.slider("Vulnerability Level", 0, 10, 5, key='a_vl'),
        'time_investment': st.slider("Time Investment", 0, 10, 5, key='a_ti'),
        'financial_gestures': st.slider("Financial Gestures", 0, 10, 5, key='a_fg'),
        'authenticity_ratio': st.slider("Authenticity Ratio (vs Pseudo-altruism)", 0, 10, 5, key='a_ar'),
        'network_introduction': st.slider("Network Introduction", 0, 10, 5, key='a_ni')
    }
    
    utility_a = calculate_utility(player_a)
    strategy_a = determine_strategy(utility_a)
    
    st.metric("Utility Score", f"{utility_a:.2f}")
    st.metric("Current Strategy", strategy_a, delta="Cooperative" if strategy_a == "Cooperate" else "Defensive")

with col2:
    st.markdown(f"### 👑 Player B: Quality Curator")
    st.markdown(f"*ELO: {int(st.session_state.elo_b)} ({tier_b}) {emoji_b}*")
    
    player_b = {
        'prompt_frequency': st.slider("Prompt Frequency", 0, 10, 5, key='b_pf'),
        'message_depth': st.slider("Message Depth", 0, 10, 5, key='b_md'),
        'vulnerability_level': st.slider("Vulnerability Level", 0, 10, 5, key='b_vl'),
        'emotional_availability': st.slider("Emotional Availability", 0, 10, 5, key='b_ea'),
        'lifestyle_inclusion': st.slider("Lifestyle Inclusion", 0, 10, 5, key='b_li'),
        'cultural_sharing': st.slider("Cultural Sharing (vs Gatekeeping)", 0, 10, 5, key='b_cs'),
        'reciprocity_level': st.slider("Reciprocity Level", 0, 10, 5, key='b_rl')
    }
    
    utility_b = calculate_utility(player_b)
    strategy_b = determine_strategy(utility_b)
    
    st.metric("Utility Score", f"{utility_b:.2f}")
    st.metric("Current Strategy", strategy_b, delta="Cooperative" if strategy_b == "Cooperate" else "Defensive")

st.markdown("---")

# Calculate outcome
f_score, outcome, emoji = calculate_outcome_score(player_a, player_b)

# Display current outcome
outcome_class = f"outcome-{outcome.lower().replace(' ', '-')}"
st.markdown(f"""
<div class="{outcome_class}">
    <h2 style='text-align: center;'>{emoji} Current Outcome: {outcome} {emoji}</h2>
    <h3 style='text-align: center;'>Outcome Score (f): {f_score:.3f}</h3>
    <p style='text-align: center;'>Player A: {strategy_a} | Player B: {strategy_b}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Action buttons
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    if st.button("🎲 Run Simulation Round", type="primary", use_container_width=True):
        payoff = calculate_payoff(strategy_a, strategy_b, utility_a, utility_b)
        game_type = identify_game_type(st.session_state.game_history)
        
        # Calculate ELO updates
        expected_a = calculate_expected_score(st.session_state.elo_a, st.session_state.elo_b)
        expected_b = 1 - expected_a
        
        actual_a, actual_b = calculate_actual_score(payoff['a'], payoff['b'])
        
        k_factor_a = calculate_k_factor(st.session_state.elo_a, st.session_state.round_number)
        k_factor_b = calculate_k_factor(st.session_state.elo_b, st.session_state.round_number)
        
        new_elo_a = update_elo(st.session_state.elo_a, expected_a, actual_a, k_factor_a)
        new_elo_b = update_elo(st.session_state.elo_b, expected_b, actual_b, k_factor_b)
        
        elo_change_a = new_elo_a - st.session_state.elo_a
        elo_change_b = new_elo_b - st.session_state.elo_b
        
        round_data = {
            'round': st.session_state.round_number + 1,
            'strategy_a': strategy_a,
            'strategy_b': strategy_b,
            'utility_a': utility_a,
            'utility_b': utility_b,
            'payoff_a': payoff['a'],
            'payoff_b': payoff['b'],
            'outcome': payoff['outcome'],
            'game_type': game_type,
            'f_score': f_score,
            'elo_a': st.session_state.elo_a,
            'elo_b': st.session_state.elo_b,
            'new_elo_a': new_elo_a,
            'new_elo_b': new_elo_b,
            'elo_change_a': elo_change_a,
            'elo_change_b': elo_change_b,
            'expected_a': expected_a,
            'actual_a': actual_a,
            'timestamp': datetime.now()
        }
        
        st.session_state.game_history.append(round_data)
        st.session_state.round_number += 1
        
        # Update ELO ratings
        st.session_state.elo_a = new_elo_a
        st.session_state.elo_b = new_elo_b
        st.session_state.elo_history_a.append(new_elo_a)
        st.session_state.elo_history_b.append(new_elo_b)
        
        st.rerun()

with col2:
    if st.button("🔄 Reset Simulation", use_container_width=True):
        st.session_state.game_history = []
        st.session_state.round_number = 0
        st.session_state.elo_a = 1500
        st.session_state.elo_b = 1500
        st.session_state.elo_history_a = [1500]
        st.session_state.elo_history_b = [1500]
        st.rerun()

with col3:
    if st.button("🔃 Reset ELO Only", use_container_width=True):
        st.session_state.elo_a = 1500
        st.session_state.elo_b = 1500
        st.session_state.elo_history_a = [1500]
        st.session_state.elo_history_b = [1500]
        st.rerun()

with col4:
    st.metric("Round", st.session_state.round_number)

# Show last round ELO changes
if st.session_state.game_history:
    last_round = st.session_state.game_history[-1]
    st.markdown("### 📊 Last Round ELO Changes")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Player A ELO Change", f"{last_round['elo_change_a']:+.1f}", 
                  delta=f"{last_round['elo_change_a']:+.1f}")
    with col2:
        st.metric("Player B ELO Change", f"{last_round['elo_change_b']:+.1f}",
                  delta=f"{last_round['elo_change_b']:+.1f}")
    with col3:
        result = "Win" if last_round['actual_a'] == 1.0 else "Loss" if last_round['actual_a'] == 0.0 else "Draw"
        st.metric("Result", result)

# Visualizations
if st.session_state.game_history:
    st.markdown("---")
    st.markdown("## 📊 Game Analysis & Visualizations")
    
    df = pd.DataFrame(st.session_state.game_history)
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 ELO Evolution", "📈 Payoff Evolution", "🎯 Strategy Matrix", "🎮 Game Type Evolution", "📋 Detailed History"])
    
    with tab1:
        # ELO Rating Evolution
        fig_elo = go.Figure()
        fig_elo.add_trace(go.Scatter(
            x=list(range(len(st.session_state.elo_history_a))), 
            y=st.session_state.elo_history_a, 
            mode='lines+markers', 
            name='Player A ELO',
            line=dict(color='#3B82F6', width=3),
            marker=dict(size=8)
        ))
        fig_elo.add_trace(go.Scatter(
            x=list(range(len(st.session_state.elo_history_b))), 
            y=st.session_state.elo_history_b, 
            mode='lines+markers', 
            name='Player B ELO',
            line=dict(color='#EC4899', width=3),
            marker=dict(size=8)
        ))
        
        # Add tier lines
        fig_elo.add_hline(y=1800, line_dash="dash", line_color="purple", annotation_text="Grandmaster")
        fig_elo.add_hline(y=1600, line_dash="dash", line_color="red", annotation_text="Master")
        fig_elo.add_hline(y=1400, line_dash="dash", line_color="blue", annotation_text="Expert")
        fig_elo.add_hline(y=1200, line_dash="dash", line_color="green", annotation_text="Intermediate")
        
        fig_elo.update_layout(
            title="ELO Rating Evolution",
            xaxis_title="Round",
            yaxis_title="ELO Rating",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_elo, use_container_width=True)
        
        # ELO Change per round
        fig_elo_change = go.Figure()
        fig_elo_change.add_trace(go.Bar(
            x=df['round'],
            y=df['elo_change_a'],
            name='Player A ELO Change',
            marker_color='#3B82F6'
        ))
        fig_elo_change.add_trace(go.Bar(
            x=df['round'],
            y=df['elo_change_b'],
            name='Player B ELO Change',
            marker_color='#EC4899'
        ))
        fig_elo_change.update_layout(
            title="ELO Rating Changes per Round",
            xaxis_title="Round",
            yaxis_title="ELO Change",
            height=400,
            barmode='group'
        )
        st.plotly_chart(fig_elo_change, use_container_width=True)
        
        # Win Probability Evolution
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            x=df['round'],
            y=df['expected_a'] * 100,
            mode='lines+markers',
            name='Player A Win Probability',
            line=dict(color='#3B82F6', width=2),
            fill='tonexty'
        ))
        fig_prob.add_trace(go.Scatter(
            x=df['round'],
            y=(1 - df['expected_a']) * 100,
            mode='lines+markers',
            name='Player B Win Probability',
            line=dict(color='#EC4899', width=2)
        ))
        fig_prob.update_layout(
            title="Win Probability Evolution (Based on ELO)",
            xaxis_title="Round",
            yaxis_title="Win Probability (%)",
            height=400
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    
    with tab2:
        # Payoff evolution chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['round'], y=df['payoff_a'], mode='lines+markers', name='Player A', line=dict(color='#3B82F6', width=3)))
        fig.add_trace(go.Scatter(x=df['round'], y=df['payoff_b'], mode='lines+markers', name='Player B', line=dict(color='#EC4899', width=3)))
        fig.update_layout(title="Payoff Evolution Over Rounds", xaxis_title="Round", yaxis_title="Payoff", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # F-score evolution
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['round'], y=df['f_score'], mode='lines+markers', fill='tozeroy', line=dict(color='#8B5CF6', width=3)))
        fig2.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Matched Threshold")
        fig2.add_hline(y=0.6, line_dash="dash", line_color="blue", annotation_text="Engaged Threshold")
        fig2.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="Complicated Threshold")
        fig2.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Confused Threshold")
        fig2.update_layout(title="Outcome Score (f) Evolution", xaxis_title="Round", yaxis_title="f Score", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Strategy distribution
        strategy_counts = df.groupby(['strategy_a', 'strategy_b']).size().reset_index(name='count')
        
        fig3 = go.Figure(data=[go.Bar(
            x=strategy_counts.apply(lambda x: f"A:{x['strategy_a']}<br>B:{x['strategy_b']}", axis=1),
            y=strategy_counts['count'],
            marker_color=['#10B981' if 'Cooperate' in str(x) and 'Cooperate' in str(y) else '#EF4444' 
                         for x, y in zip(strategy_counts['strategy_a'], strategy_counts['strategy_b'])]
        )])
        fig3.update_layout(title="Strategy Combination Frequency", xaxis_title="Strategy Pair", yaxis_title="Frequency", height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Outcome distribution
        outcome_counts = df['outcome'].value_counts()
        fig4 = go.Figure(data=[go.Pie(labels=outcome_counts.index, values=outcome_counts.values, hole=0.4)])
        fig4.update_layout(title="Outcome Distribution", height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab4:
        # Game type evolution
        fig5 = go.Figure()
        game_types = df['game_type'].unique()
        for gt in game_types:
            mask = df['game_type'] == gt
            fig5.add_trace(go.Scatter(x=df[mask]['round'], y=[gt]*sum(mask), mode='markers', name=gt, marker=dict(size=12)))
        fig5.update_layout(title="Game Type Evolution", xaxis_title="Round", yaxis_title="Game Type", height=400)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Utility comparison
        fig6 = make_subplots(rows=1, cols=2, subplot_titles=("Player A Utility", "Player B Utility"))
        fig6.add_trace(go.Scatter(x=df['round'], y=df['utility_a'], mode='lines+markers', name='Utility A', line=dict(color='#3B82F6')), row=1, col=1)
        fig6.add_trace(go.Scatter(x=df['round'], y=df['utility_b'], mode='lines+markers', name='Utility B', line=dict(color='#EC4899')), row=1, col=2)
        fig6.update_xaxes(title_text="Round", row=1, col=1)
        fig6.update_xaxes(title_text="Round", row=1, col=2)
        fig6.update_yaxes(title_text="Utility", row=1, col=1)
        fig6.update_yaxes(title_text="Utility", row=1, col=2)
        fig6.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab5:
        # Detailed history table
        display_df = df[['round', 'strategy_a', 'strategy_b', 'payoff_a', 'payoff_b', 
                         'outcome', 'game_type', 'f_score', 'elo_a', 'elo_b', 
                         'elo_change_a', 'elo_change_b']].copy()
        
        # Round values for display
        display_df['elo_a'] = display_df['elo_a'].round(0).astype(int)
        display_df['elo_b'] = display_df['elo_b'].round(0).astype(int)
        display_df['elo_change_a'] = display_df['elo_change_a'].round(1)
        display_df['elo_change_b'] = display_df['elo_change_b'].round(1)
        display_df['f_score'] = display_df['f_score'].round(3)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'round': 'Round',
                'strategy_a': 'Strategy A',
                'strategy_b': 'Strategy B',
                'payoff_a': 'Payoff A',
                'payoff_b': 'Payoff B',
                'outcome': 'Outcome',
                'game_type': 'Game Type',
                'f_score': 'f Score',
                'elo_a': 'ELO A',
                'elo_b': 'ELO B',
                'elo_change_a': 'Δ ELO A',
                'elo_change_b': 'Δ ELO B'
            }
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Game History CSV",
            data=csv,
            file_name=f"relationship_game_theory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ELO Statistics
if st.session_state.game_history:
    st.markdown("---")
    st.markdown("## 🏆 ELO Statistics & Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_elo_a = max(st.session_state.elo_history_a)
        st.metric("Player A Peak ELO", int(max_elo_a), 
                  delta=f"+{int(max_elo_a - 1500)} from start")
    
    with col2:
        max_elo_b = max(st.session_state.elo_history_b)
        st.metric("Player B Peak ELO", int(max_elo_b),
                  delta=f"+{int(max_elo_b - 1500)} from start")
    
    with col3:
        total_change_a = st.session_state.elo_a - 1500
        st.metric("Player A Total Change", f"{total_change_a:+.0f}",
                  delta="Improved" if total_change_a > 0 else "Declined")
    
    with col4:
        total_change_b = st.session_state.elo_b - 1500
        st.metric("Player B Total Change", f"{total_change_b:+.0f}",
                  delta="Improved" if total_change_b > 0 else "Declined")
    
    # Performance metrics
    st.markdown("### 📊 Performance Breakdown")
    
    wins_a = sum(1 for h in st.session_state.game_history if h['actual_a'] == 1.0)
    wins_b = sum(1 for h in st.session_state.game_history if h['actual_a'] == 0.0)
    draws = sum(1 for h in st.session_state.game_history if h['actual_a'] == 0.5)
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Player A Wins", wins_a, delta=f"{wins_a/len(st.session_state.game_history)*100:.1f}%")
    
    with perf_col2:
        st.metric("Player B Wins", wins_b, delta=f"{wins_b/len(st.session_state.game_history)*100:.1f}%")
    
    with perf_col3:
        st.metric("Draws", draws, delta=f"{draws/len(st.session_state.game_history)*100:.1f}%")
    
    with perf_col4:
        avg_elo_change = df['elo_change_a'].abs().mean()
        st.metric("Avg ELO Volatility", f"{avg_elo_change:.1f}")
    
    # Upset victories
    upsets_a = sum(1 for h in st.session_state.game_history 
                   if h['actual_a'] == 1.0 and h['expected_a'] < 0.5)
    upsets_b = sum(1 for h in st.session_state.game_history 
                   if h['actual_a'] == 0.0 and h['expected_a'] > 0.5)
    
    if upsets_a > 0 or upsets_b > 0:
        st.markdown("### 🎉 Upset Victories")
        upset_col1, upset_col2 = st.columns(2)
        with upset_col1:
            st.metric("Player A Upsets", upsets_a, 
                     delta="Won as underdog" if upsets_a > 0 else None)
        with upset_col2:
            st.metric("Player B Upsets", upsets_b,
                     delta="Won as underdog" if upsets_b > 0 else None)

# Strategic Insights
st.markdown("---")
st.markdown("## 💡 Strategic Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Current State Analysis")
    
    utility_diff = abs(utility_a - utility_b)
    balance_status = "Balanced" if utility_diff < 2 else "Imbalanced"
    
    st.write(f"**Utility Difference:** {utility_diff:.2f} ({balance_status})")
    st.write(f"**Nash Equilibrium Status:** {'✅ Achieved' if strategy_a == 'Cooperate' and strategy_b == 'Cooperate' else '❌ Not Achieved'}")
    st.write(f"**Power Dynamic:** {'Equal' if utility_diff < 1.5 else 'Asymmetric'}")
    st.write(f"**ELO Balance:** {'Competitive' if abs(st.session_state.elo_a - st.session_state.elo_b) < 100 else 'Unbalanced'}")
    
    if st.session_state.game_history:
        recent_outcomes = [h['outcome'] for h in st.session_state.game_history[-5:]]
        cooperation_rate = sum(1 for h in st.session_state.game_history[-5:] if h['strategy_a'] == 'Cooperate' and h['strategy_b'] == 'Cooperate') / min(5, len(st.session_state.game_history))
        st.write(f"**Recent Cooperation Rate:** {cooperation_rate*100:.0f}%")
        st.write(f"**Trend:** {'📈 Improving' if cooperation_rate > 0.6 else '📉 Declining' if cooperation_rate < 0.4 else '➡️ Stable'}")
        
        # ELO momentum
        if len(st.session_state.elo_history_a) >= 5:
            elo_momentum_a = st.session_state.elo_history_a[-1] - st.session_state.elo_history_a[-5]
            elo_momentum_b = st.session_state.elo_history_b[-1] - st.session_state.elo_history_b[-5]
            st.write(f"**ELO Momentum A:** {elo_momentum_a:+.0f} (last 5 rounds)")
            st.write(f"**ELO Momentum B:** {elo_momentum_b:+.0f} (last 5 rounds)")

with col2:
    st.markdown("### 📋 Recommendations")
    
    if outcome == "MATCHED":
        st.success("🎉 **Excellent!** You've reached Nash equilibrium. Maintain current cooperation level.")
    elif outcome == "ENGAGED":
        st.info("💍 **Good Progress!** Continue building trust and increasing mutual investment.")
    elif outcome == "COMPLICATED":
        st.warning("⚠️ **Power Games Detected.** Both players should increase authenticity and reduce defensive tactics.")
    elif outcome == "CONFUSED":
        st.warning("🤔 **Mixed Signals.** Improve communication clarity and consistency.")
    else:
        st.error("💔 **Misalignment.** Significant strategy change needed. Consider if goals are compatible.")
    
    # ELO-based recommendations
    elo_diff = abs(st.session_state.elo_a - st.session_state.elo_b)
    if elo_diff > 200:
        if st.session_state.elo_a > st.session_state.elo_b:
            st.write("**ELO Gap:** Player A has significant skill advantage. Player B should learn from A's strategies.")
        else:
            st.write("**ELO Gap:** Player B has significant skill advantage. Player A should learn from B's strategies.")
    
    if strategy_a == "Defect" or strategy_b == "Defect":
        st.write("**Path Forward:**")
        st.write("- Increase vulnerability and authenticity")
        st.write("- Reduce manipulation tactics")
        st.write("- Build trust through consistent actions")
        st.write("- Move from defection to conditional cooperation")
        st.write("- Focus on win-win outcomes to improve both ELO ratings")
    else:
        st.write("**Maintain Success:**")
        st.write("- Continue transparent communication")
        st.write("- Keep investing mutually")
        st.write("- Avoid complacency")
        st.write("- Regular relationship check-ins")
        st.write("- Sustain high ELO through consistent cooperation")

# Game Theory Conclusion
st.markdown("---")
st.markdown("## 🎓 Comprehensive Game Theory Conclusion")

conclusion_col1, conclusion_col2 = st.columns([2, 1])

with conclusion_col1:
    st.markdown("""
    ### This Relationship Game is:
    
    **1. Non-Cooperative Game (Transitioning to Cooperative)**
    - Players independently maximize their utility initially
    - Can evolve into cooperative game at Nash equilibrium
    - Strategic interaction without binding agreements
    
    **2. Simultaneous Move Game**
    - Both players adjust strategies without real-time opponent knowledge
    - No sequential advantage
    
    **3. Imperfect & Incomplete Information Game**
    - Imperfect: Cannot observe all actions (hidden intentions, pseudo-altruism)
    - Incomplete: Don't know exact payoff functions of opponent
    - Creates strategic uncertainty
    
    **4. Non-Zero-Sum Game**
    - Both can win (8,8) or both can lose (4,4)
    - Total welfare varies by outcome
    - Cooperation increases total payoff
    
    **5. Asymmetric Game (Becoming Symmetric)**
    - Different initial resources and strategies
    - Becomes symmetric when equilibrium reached
    - Role interchangeability at Nash equilibrium
    
    **6. Dynamic/Repeated Game**
    - Multiple rounds with history
    - Reputation effects matter
    - Enables tit-for-tat and learning strategies
    
    **7. ELO Rating System Integration**
    - Measures player skill in strategic relationship dynamics
    - Adapts K-factor based on experience and rating
    - Tracks performance evolution over time
    - Identifies upsets and momentum shifts
    
    **8. Similar to Classic Games:**
    - **Prisoner's Dilemma** (when trust is low)
    - **Battle of Sexes** (coordination with preference differences)
    - **Stag Hunt** (high cooperation payoff, requires trust)
    - **Chicken Game** (brinkmanship, who yields first)
    """)

with conclusion_col2:
    st.markdown("### Key Properties")
    
    properties = {
        "Pareto Optimal": "(8,8) - Mutual Cooperation",
        "Nash Equilibrium": "(8,8) when both cooperate",
        "Dominant Strategy": "None (depends on opponent)",
        "Best Response": "Tit-for-tat with forgiveness",
        "Evolutionarily Stable": "Cooperation in repeated game",
        "ELO Starting Rating": "1500",
        "ELO Range": "0-3000"
    }
    
    for prop, value in properties.items():
        st.metric(prop, value)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | Game Theory Analysis Framework with ELO Rating System</p>
    <p>Based on Nash Equilibrium, Prisoner's Dilemma, Repeated Game Theory, and ELO Rating Algorithm</p>
    <p>ELO system adapted from chess ratings - measures strategic relationship skill</p>
</div>
""", unsafe_allow_html=True)
            
