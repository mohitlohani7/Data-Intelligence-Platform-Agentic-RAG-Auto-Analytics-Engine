import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from fpdf import FPDF
from io import BytesIO
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

st.set_page_config(page_title="Data Intelligence Platform", layout="wide", initial_sidebar_state="expanded")

# ================= PREMIUM CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background-color: #060b17;
        background-image:
            radial-gradient(ellipse at 10% 10%, rgba(78, 205, 196, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(255, 107, 107, 0.03) 0%, transparent 60%);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    h1 {
        background: linear-gradient(135deg, #A0FFED 0%, #4ECDC4 40%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(78, 205, 196, 0.15);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: all 0.3s ease;
        margin-bottom: 0.5rem;
    }
    .kpi-card:hover {
        border-color: rgba(78, 205, 196, 0.4);
        box-shadow: 0 12px 40px rgba(78, 205, 196, 0.1);
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ECDC4, #A0FFED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    .kpi-icon {
        font-size: 1.5rem;
        margin-bottom: 0.4rem;
    }

    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.08), rgba(139, 92, 246, 0.08));
        border-left: 3px solid #4ECDC4;
        border-radius: 0 12px 12px 0;
        padding: 0.7rem 1.2rem;
        margin: 1.5rem 0 1rem 0;
        font-weight: 700;
        font-size: 1.1rem;
        color: #e2e8f0;
    }

    /* Chart containers */
    .chart-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Data Quality badges */
    .quality-good { color: #4ECDC4; font-weight: 700; }
    .quality-warn { color: #FBBF24; font-weight: 700; }
    .quality-bad  { color: #FF6B6B; font-weight: 700; }

    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(78, 205, 196, 0.05));
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stChatInput > div {
        background-color: rgba(30, 41, 59, 0.7) !important;
        color: white !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    .stTextInput > div > div > input:focus,
    .stChatInput > div:focus-within {
        border-color: #4ECDC4 !important;
        box-shadow: 0 0 15px rgba(78, 205, 196, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: #0b0f19;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.6rem 1.5rem;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(78, 205, 196, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(78, 205, 196, 0.5);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 16, 30, 0.98) 0%, rgba(6, 11, 23, 0.98) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 0.4rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #64748b;
        font-weight: 600;
        font-size: 0.95rem;
        border: none !important;
        padding: 0.5rem 1.2rem !important;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.15), rgba(139, 92, 246, 0.1)) !important;
        color: #4ECDC4 !important;
        border: 1px solid rgba(78, 205, 196, 0.2) !important;
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* Chat bubbles */
    .stChatMessage { background-color: transparent !important; border: none !important; }
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(46, 49, 65, 0.9), rgba(30, 32, 43, 0.9)) !important;
        border-radius: 18px 18px 0 18px !important;
        max-width: 85%; margin-left: auto !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(20, 30, 48, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 18px 18px 18px 0 !important;
        max-width: 90%; margin-right: auto !important;
        border: 1px solid rgba(78, 205, 196, 0.1) !important;
    }

    /* Metric delta */
    [data-testid="stMetricDelta"] { color: #4ECDC4; }

    /* Progress bars */
    .stProgress > div > div { background: linear-gradient(90deg, #4ECDC4, #8B5CF6); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ================= CONFIG =================
API_URL = "http://127.0.0.1:8080"

# Session state init
for key, default in [
    ("chat_history", []),
    ("csv_chat_history", []),
    ("backend_ready", False),
    ("finance_analysis_complete", False),
    ("finance_llm_text", ""),
    ("chart_img_paths", []),
    ("df_cached", None),
    ("backend_status", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Backend status check
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("🟢 Backend Online")
            st.session_state.backend_status = "online"
        else:
            st.warning("🟡 Backend Issue")
    except:
        st.error("🔴 Backend Offline")
        st.session_state.backend_status = "offline"

    st.markdown("---")
    provider = st.selectbox("🤖 LLM Provider", ["Groq (Free)", "OpenAI", "Claude"])

    if provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY", "")
        model_name = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
        prov_key = "OpenAI"
    elif provider == "Claude":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        model_name = st.selectbox("Model", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"])
        prov_key = "Claude"
    else:
        api_key = os.getenv("GROQ_API_KEY", "")
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
        prov_key = "Groq"

    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.2)

    if not api_key:
        st.error(f"⚠️ No {provider} key found in .env!")
    else:
        st.success(f"✅ {provider} key loaded")

    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown("""
    1. **RAG Tab** → Upload PDFs → Chat
    2. **Dashboard Tab** → Upload CSV → Auto-analyze
    3. **AI Report** → Generate LLM insights
    """)

# ============================================
# CHART HELPER: dark theme style
# ============================================
DARK_BG = '#0a0f1e'
CARD_BG = '#0f172a'
ACCENT1 = '#4ECDC4'
ACCENT2 = '#FF6B6B'
ACCENT3 = '#8B5CF6'
ACCENT4 = '#FBBF24'
PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, '#06B6D4', '#F472B6', '#34D399', '#FB923C']

def apply_dark_style(ax, fig):
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(DARK_BG)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.tick_params(colors='#64748b', labelsize=9)
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
        spine.set_linewidth(0.8)
    ax.grid(color='#1e293b', linestyle='--', linewidth=0.5, alpha=0.7)

def save_fig(fig, name):
    path = os.path.join(tempfile.gettempdir(), f"{name}.png")
    fig.savefig(path, facecolor=CARD_BG, bbox_inches='tight', dpi=150)
    return path


# ============================================
# FULL AUTO-DASHBOARD FUNCTION
# ============================================
def render_dashboard(df):
    numeric_cols  = df.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns.tolist()
    cat_cols      = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    date_cols     = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'month' in c.lower() or 'year' in c.lower() or 'day' in c.lower()]

    # Try parse date cols
    parsed_dates = []
    for dc in date_cols:
        try:
            df[dc] = pd.to_datetime(df[dc], errors='coerce')
            if df[dc].notna().sum() > 0:
                parsed_dates.append(dc)
        except:
            pass

    # ---- KPI ROW ----
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    null_pct = round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 1)
    dup_count = df.duplicated().sum()

    kpi_data = [
        ("📋", f"{df.shape[0]:,}", "Total Rows"),
        ("🗂️", f"{df.shape[1]}", "Columns"),
        ("🔢", f"{len(numeric_cols)}", "Numeric"),
        ("🏷️", f"{len(cat_cols)}", "Categorical"),
        ("⚠️", f"{null_pct}%", "Null Rate"),
        ("♊", f"{dup_count:,}", "Duplicates"),
        ("💾", f"{round(df.memory_usage(deep=True).sum() / 1024, 1)} KB", "Memory"),
        ("📅", f"{len(parsed_dates)}", "Date Cols"),
    ]

    cols = st.columns(len(kpi_data))
    for col, (icon, val, label) in zip(cols, kpi_data):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    all_img_paths = []
    chart_titles  = []

    # ---- DISTRIBUTION CHARTS (numeric) ----
    if numeric_cols:
        st.markdown('<div class="section-header">📈 Numeric Distributions & Trends</div>', unsafe_allow_html=True)

        show_cols = numeric_cols[:8]   # max 8 numeric charts
        grid_cols = st.columns(2)

        for idx, col_name in enumerate(show_cols):
            with grid_cols[idx % 2]:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                apply_dark_style(ax, fig)
                data_clean = df[col_name].dropna()
                sns.histplot(data_clean, kde=True, ax=ax, color=PALETTE[idx % len(PALETTE)],
                             edgecolor='none', alpha=0.75, linewidth=2)
                ax.lines[-1].set_color('white') if ax.lines else None
                ax.set_title(f"Distribution: {col_name}", fontweight='bold', fontsize=11, pad=10)
                ax.set_xlabel(col_name)
                ax.set_ylabel("Frequency")
                plt.tight_layout()
                st.pyplot(fig)
                path = save_fig(fig, f"hist_{idx}")
                all_img_paths.append(path)
                chart_titles.append(f"Distribution: {col_name}")
                plt.close(fig)

    # ---- TIME SERIES ----
    if parsed_dates and numeric_cols:
        st.markdown('<div class="section-header">📅 Time Series Analysis</div>', unsafe_allow_html=True)
        date_col = parsed_dates[0]
        ts_cols  = st.columns(min(2, len(numeric_cols)))

        for idx, nc in enumerate(numeric_cols[:4]):
            with ts_cols[idx % 2]:
                try:
                    temp = df[[date_col, nc]].dropna().sort_values(date_col)
                    # resample if needed
                    temp = temp.set_index(date_col)
                    if len(temp) > 500:
                        temp = temp.resample('W').mean()
                    temp = temp.reset_index()

                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    apply_dark_style(ax, fig)
                    ax.fill_between(temp[date_col], temp[nc], alpha=0.15, color=PALETTE[idx % len(PALETTE)])
                    ax.plot(temp[date_col], temp[nc], color=PALETTE[idx % len(PALETTE)], linewidth=2, marker='o', markersize=2)
                    ax.set_title(f"Trend: {nc} over Time", fontweight='bold', fontsize=11, pad=10)
                    ax.set_xlabel("Date")
                    ax.set_ylabel(nc)
                    plt.xticks(rotation=35)
                    plt.tight_layout()
                    st.pyplot(fig)
                    path = save_fig(fig, f"ts_{idx}")
                    all_img_paths.append(path)
                    chart_titles.append(f"Time Series: {nc}")
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Could not plot {nc} over time: {e}")

    # ---- CATEGORICAL CHARTS ----
    if cat_cols:
        st.markdown('<div class="section-header">🏷️ Categorical Breakdown</div>', unsafe_allow_html=True)

        cat_grid = st.columns(2)
        shown = 0

        for idx, col_name in enumerate(cat_cols):
            nunique = df[col_name].nunique()
            if nunique < 2 or nunique > 50:
                continue
            with cat_grid[shown % 2]:
                counts = df[col_name].value_counts().head(12)
                fig, ax = plt.subplots(figsize=(6, max(3, len(counts) * 0.45)))
                apply_dark_style(ax, fig)

                colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
                bars = ax.barh(counts.index.astype(str), counts.values, color=colors, edgecolor='none', height=0.65)

                # Value labels
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_width() + max(counts.values) * 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:,}', va='center', ha='left', color='#94a3b8', fontsize=8)

                ax.set_title(f"Distribution: {col_name}", fontweight='bold', fontsize=11, pad=10)
                ax.set_xlabel("Count")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                path = save_fig(fig, f"cat_{idx}")
                all_img_paths.append(path)
                chart_titles.append(f"Category: {col_name}")
                plt.close(fig)
            shown += 1
            if shown >= 6:
                break

        # Pie charts for top categorical cols with < 8 unique values
        st.markdown("#### 🥧 Proportion Analysis")
        pie_cols_filtered = [c for c in cat_cols if 2 <= df[c].nunique() <= 8][:4]
        if pie_cols_filtered:
            pie_grid = st.columns(len(pie_cols_filtered))
            for idx, col_name in enumerate(pie_cols_filtered):
                with pie_grid[idx]:
                    counts = df[col_name].value_counts().head(8)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    fig.patch.set_facecolor(CARD_BG)
                    wedges, _ = ax.pie(counts.values, labels=None, colors=PALETTE[:len(counts)],
                                        startangle=90, wedgeprops=dict(edgecolor=CARD_BG, linewidth=2))
                    ax.legend(wedges, [str(l) for l in counts.index], loc='lower center',
                              bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7,
                              frameon=False, labelcolor='#94a3b8')
                    ax.set_title(col_name, color='#e2e8f0', fontweight='bold', fontsize=10, pad=8)
                    circle = plt.Circle((0, 0), 0.65, color=CARD_BG)
                    ax.add_patch(circle)
                    plt.tight_layout()
                    st.pyplot(fig)
                    path = save_fig(fig, f"pie_{idx}")
                    all_img_paths.append(path)
                    chart_titles.append(f"Proportion: {col_name}")
                    plt.close(fig)

    # ---- CORRELATION HEATMAP ----
    if len(numeric_cols) >= 2:
        st.markdown('<div class="section-header">🔗 Correlation Matrix</div>', unsafe_allow_html=True)

        corr_cols = numeric_cols[:12]
        corr = df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(max(8, len(corr_cols) * 0.9), max(6, len(corr_cols) * 0.75)))
        apply_dark_style(ax, fig)

        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(160, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, ax=ax, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt='.2f', annot_kws={'size': 8, 'color': 'white'},
                    linewidths=0.5, linecolor='#0f172a',
                    cbar_kws={'shrink': 0.8})

        ax.set_title("Feature Correlation Heatmap", fontweight='bold', fontsize=13, pad=15)
        ax.tick_params(axis='x', rotation=45, labelsize=8, colors='#94a3b8')
        ax.tick_params(axis='y', rotation=0, labelsize=8, colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig)
        path = save_fig(fig, "heatmap")
        all_img_paths.append(path)
        chart_titles.append("Correlation Heatmap")
        plt.close(fig)

        # ---- SCATTER MATRIX (top 4 correlated pairs) ----
        st.markdown("#### 🔵 Scatter Analysis (Top Correlated Pairs)")
        corr_pairs = []
        corr_vals  = corr.abs().unstack()
        corr_vals  = corr_vals[corr_vals < 1].sort_values(ascending=False)
        seen = set()
        for (a, b), v in corr_vals.items():
            pair = tuple(sorted([a, b]))
            if pair not in seen:
                corr_pairs.append((a, b, v))
                seen.add(pair)
            if len(corr_pairs) == 4:
                break

        scatter_grid = st.columns(2)
        for idx, (col_a, col_b, corr_v) in enumerate(corr_pairs):
            with scatter_grid[idx % 2]:
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                apply_dark_style(ax, fig)
                sample = df[[col_a, col_b]].dropna().sample(min(1000, len(df)))
                ax.scatter(sample[col_a], sample[col_b],
                           color=PALETTE[idx % len(PALETTE)], alpha=0.5, s=15, edgecolors='none')

                # Trend line
                try:
                    z = np.polyfit(sample[col_a], sample[col_b], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(sample[col_a].min(), sample[col_a].max(), 100)
                    ax.plot(x_line, p(x_line), color='white', linewidth=1.5, alpha=0.8, linestyle='--')
                except:
                    pass

                ax.set_title(f"{col_a} vs {col_b} (r={corr_v:.2f})", fontweight='bold', fontsize=10, pad=8)
                ax.set_xlabel(col_a); ax.set_ylabel(col_b)
                plt.tight_layout()
                st.pyplot(fig)
                path = save_fig(fig, f"scatter_{idx}")
                all_img_paths.append(path)
                chart_titles.append(f"Scatter: {col_a} vs {col_b}")
                plt.close(fig)

    # ---- BOX PLOTS (outlier detection) ----
    if numeric_cols:
        st.markdown('<div class="section-header">📦 Outlier Detection (Box Plots)</div>', unsafe_allow_html=True)

        box_cols = numeric_cols[:8]
        fig, axes = plt.subplots(1, len(box_cols), figsize=(max(10, len(box_cols) * 2), 4))
        fig.patch.set_facecolor(CARD_BG)
        if len(box_cols) == 1:
            axes = [axes]

        for i, (ax, col_name) in enumerate(zip(axes, box_cols)):
            data = df[col_name].dropna()
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            flierprops=dict(marker='o', markerfacecolor=ACCENT2, markersize=3, alpha=0.5),
                            medianprops=dict(color='white', linewidth=2),
                            boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], alpha=0.6),
                            whiskerprops=dict(color='#64748b'),
                            capprops=dict(color='#64748b'))
            apply_dark_style(ax, fig)
            ax.set_title(col_name[:14], fontsize=8, fontweight='bold', color='#e2e8f0', pad=5)
            ax.set_xticks([])

        fig.suptitle("Outlier Detection — Box Plots", color='#e2e8f0', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        path = save_fig(fig, "boxplots")
        all_img_paths.append(path)
        chart_titles.append("Box Plots — Outlier Detection")
        plt.close(fig)

    # ---- DATA QUALITY REPORT ----
    st.markdown('<div class="section-header">🛡️ Data Quality Report</div>', unsafe_allow_html=True)

    q_cols = st.columns(3)
    with q_cols[0]:
        st.markdown("**Missing Values by Column**")
        null_df = df.isnull().sum().reset_index()
        null_df.columns = ['Column', 'Missing']
        null_df['%'] = (null_df['Missing'] / len(df) * 100).round(1)
        null_df = null_df[null_df['Missing'] > 0].sort_values('Missing', ascending=False)
        if null_df.empty:
            st.success("✅ No missing values!")
        else:
            st.dataframe(null_df, use_container_width=True, hide_index=True)

    with q_cols[1]:
        st.markdown("**Unique Value Counts**")
        uniq_df = pd.DataFrame({
            'Column': df.columns,
            'Unique': [df[c].nunique() for c in df.columns],
            'Type': [str(df[c].dtype) for c in df.columns]
        })
        st.dataframe(uniq_df, use_container_width=True, hide_index=True)

    with q_cols[2]:
        st.markdown("**Statistical Summary**")
        if numeric_cols:
            stats = df[numeric_cols[:5]].describe().round(2)
            st.dataframe(stats, use_container_width=True)

    # ---- AUTO INSIGHTS ----
    st.markdown('<div class="section-header">💡 Auto-Generated Insights</div>', unsafe_allow_html=True)

    insights = []
    if numeric_cols:
        for nc in numeric_cols[:3]:
            series = df[nc].dropna()
            skew = series.skew()
            if abs(skew) > 1:
                insights.append(f"📊 **{nc}** is {'right' if skew > 0 else 'left'}-skewed (skewness={skew:.2f}) — consider log transformation for modeling.")
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
            if outliers > 0:
                insights.append(f"⚠️ **{nc}** has **{outliers}** outliers ({outliers/len(df)*100:.1f}% of data).")

    if len(numeric_cols) >= 2:
        corr_vals_flat = df[numeric_cols].corr().abs().unstack()
        corr_vals_flat = corr_vals_flat[corr_vals_flat < 1].sort_values(ascending=False)
        if not corr_vals_flat.empty:
            top_pair = corr_vals_flat.index[0]
            top_val  = corr_vals_flat.iloc[0]
            insights.append(f"🔗 Strongest correlation: **{top_pair[0]}** ↔ **{top_pair[1]}** (r={top_val:.2f})")

    if dup_count > 0:
        insights.append(f"♊ Found **{dup_count}** duplicate rows — consider deduplication before analysis.")

    if null_pct > 10:
        insights.append(f"❗ **{null_pct}%** of data is missing — imputation or exclusion strategy needed.")
    elif null_pct == 0:
        insights.append("✅ Dataset is **complete** — no missing values detected.")

    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    if not insights:
        st.info("Upload data to see auto-generated insights.")

    return all_img_paths, chart_titles


# ============================================
# PDF REPORT
# ============================================
def create_pdf_report(img_paths, llm_text, df_shape):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(6, 11, 23)
    pdf.rect(0, 0, 210, 297, 'F')

    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(78, 205, 196)
    pdf.cell(0, 12, "Data Intelligence Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(150, 150, 170)
    from datetime import datetime
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Rows: {df_shape[0]:,} | Cols: {df_shape[1]}", ln=True, align='C')
    pdf.ln(8)

    y_offset = 40
    for idx, img_path in enumerate(img_paths):
        if os.path.exists(img_path):
            if y_offset > 200:
                pdf.add_page()
                pdf.set_fill_color(6, 11, 23)
                pdf.rect(0, 0, 210, 297, 'F')
                y_offset = 15
            try:
                pdf.image(img_path, x=8, y=y_offset, w=193)
                y_offset += 68
            except:
                pass

    pdf.add_page()
    pdf.set_fill_color(6, 11, 23)
    pdf.rect(0, 0, 210, 297, 'F')
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(78, 205, 196)
    pdf.cell(0, 10, "AI-Generated Analysis", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(200, 210, 220)
    clean_text = llm_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, clean_text)

    export_path = os.path.join(tempfile.gettempdir(), "data_intelligence_report.pdf")
    pdf.output(export_path)
    return export_path


# ============================================
# MAIN APP
# ============================================
st.title("Data Intelligence Platform")
st.markdown('<p style="color:#64748b; margin-top:-0.5rem; margin-bottom:1.5rem;">Enterprise RAG · Auto-Analytics · AI Insights</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📚 Document RAG Chat", "📊 Auto Analytics Dashboard"])

# ====================== TAB 1: RAG ======================
with tab1:
    colA, colB = st.columns([1, 2])
    with colA:
        st.markdown("#### 📁 Knowledge Upload")
        pdf_docs = st.file_uploader("Drop PDFs to vectorize", type=["pdf"], accept_multiple_files=True)
        if st.button("⚡ Vectorize Documents", key="vectorize_btn"):
            if pdf_docs:
                with st.spinner("Embedding PDFs into vector store..."):
                    files_data = [("files", (pdf.name, pdf.getvalue(), "application/pdf")) for pdf in pdf_docs]
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files_data, timeout=120)
                        if res.status_code == 200:
                            st.success(f"✅ {res.json()['chunks']} vector chunks created!")
                            st.session_state.backend_ready = True
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Backend error: {e}")
            else:
                st.warning("Upload at least one PDF first.")

    with colB:
        st.markdown("#### 💬 Autonomous RAG Chat")
        if not st.session_state.backend_ready:
            st.info("👆 Upload and vectorize PDFs first to enable chat.")

        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        user_input = st.chat_input("Ask anything about your documents...", key="rag_input")
        if user_input:
            if not st.session_state.backend_ready:
                st.error("Vectorize documents first.")
            elif not api_key:
                st.error("No API key found in .env file.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    with st.chat_message("assistant"):
                        with st.spinner("Reasoning..."):
                            payload = {
                                "query": user_input, "provider": prov_key, "api_key": api_key,
                                "model_name": model_name, "temperature": temperature,
                                "use_fallback": False,
                                "chat_history": st.session_state.chat_history[:-1]
                            }
                            try:
                                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                                if res.status_code == 200:
                                    output = res.json()["response"]
                                    st.markdown(output)
                                    st.session_state.chat_history.append({"role": "assistant", "content": output})
                                else:
                                    err = res.json().get("detail", res.text)
                                    st.error(f"API Error: {err}")
                            except Exception as e:
                                st.error(f"Connection error: {e}")


# ====================== TAB 2: DASHBOARD ======================
with tab2:
    st.markdown("#### 📂 Upload CSV Dataset")
    csv_file = st.file_uploader("Drop any CSV file for instant auto-analysis", type=["csv"], key="csv_upload")

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.session_state.df_cached = df

            # Push to backend (silent)
            try:
                file_obj = ("file", (csv_file.name, csv_file.getvalue(), "text/csv"))
                requests.post(f"{API_URL}/upload_csv", files=[file_obj], timeout=10)
            except:
                pass

            st.success(f"✅ **{csv_file.name}** loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

            # Preview
            with st.expander("👁️ Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            st.markdown("---")

            # FULL AUTO DASHBOARD
            all_img_paths, chart_titles = render_dashboard(df)
            st.session_state.chart_img_paths = all_img_paths

            st.markdown("---")

            # ---- AI ANALYSIS SECTION ----
            st.markdown('<div class="section-header">🤖 AI-Powered Deep Analysis</div>', unsafe_allow_html=True)

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols     = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if not api_key:
                st.warning("⚠️ Add an API key in .env to enable AI analysis. Charts above work without API key.")
            else:
                if st.button("🚀 Generate AI Analysis Report", key="analyze_btn"):
                    with st.spinner("AI is analyzing your data..."):
                        summary_dict = {
                            "filename": csv_file.name,
                            "rows": len(df),
                            "columns": df.shape[1],
                            "numeric_columns": numeric_cols[:10],
                            "categorical_columns": cat_cols[:10],
                            "describe": df.describe().round(2).to_dict(),
                            "charts_generated": chart_titles,
                            "top_categories": {c: df[c].value_counts().head(5).to_dict() for c in cat_cols[:3]},
                            "nulls": df.isnull().sum().to_dict(),
                            "correlations": df[numeric_cols[:6]].corr().round(2).to_dict() if len(numeric_cols) >= 2 else {}
                        }
                        payload = {
                            "data_summary": str(summary_dict)[:4000],
                            "provider": prov_key,
                            "api_key": api_key,
                            "model_name": model_name
                        }
                        try:
                            res = requests.post(f"{API_URL}/analyze_finance", json=payload, timeout=90)
                            if res.status_code == 200:
                                st.session_state.finance_llm_text = res.json()["analysis"]
                                st.session_state.finance_analysis_complete = True
                            else:
                                err = res.json().get("detail", res.text)
                                st.error(f"Analysis error: {err}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")

            if st.session_state.finance_analysis_complete:
                st.markdown(f'<div class="insight-card">{st.session_state.finance_llm_text}</div>', unsafe_allow_html=True)

                with st.spinner("Building PDF report..."):
                    pdf_path = create_pdf_report(
                        st.session_state.chart_img_paths,
                        st.session_state.finance_llm_text,
                        df.shape
                    )
                    with open(pdf_path, 'rb') as f:
                        pdf_bytes = f.read()

                st.download_button(
                    label="📥 Download Full Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"Data_Report_{csv_file.name.replace('.csv','')}.pdf",
                    mime="application/pdf",
                )

            # ---- CSV CHAT ----
            st.markdown("---")
            st.markdown('<div class="section-header">💬 Chat with Your Data</div>', unsafe_allow_html=True)

            if not api_key:
                st.info("Add an API key to chat with your dataset.")
            else:
                chat_box = st.container(height=280)
                with chat_box:
                    for message in st.session_state.csv_chat_history:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                csv_query = st.chat_input("Ask questions about your data (e.g. 'What is the average sales?')...", key="csv_chat_input")
                if csv_query:
                    st.session_state.csv_chat_history.append({"role": "user", "content": csv_query})
                    with chat_box:
                        with st.chat_message("user"):
                            st.markdown(csv_query)
                        with st.chat_message("assistant"):
                            with st.spinner("Analyzing data..."):
                                payload = {
                                    "query": csv_query, "provider": prov_key, "api_key": api_key,
                                    "model_name": model_name, "use_fallback": False, "temperature": 0.0,
                                    "chat_history": st.session_state.csv_chat_history[:-1]
                                }
                                try:
                                    res = requests.post(f"{API_URL}/chat_csv", json=payload, timeout=60)
                                    if res.status_code == 200:
                                        out = res.json()["response"]
                                        st.markdown(out)
                                        st.session_state.csv_chat_history.append({"role": "assistant", "content": out})
                                    else:
                                        err = res.json().get("detail", res.text)
                                        st.error(f"Error: {err}")
                                except Exception as e:
                                    st.error(f"Server error: {e}")

        except Exception as e:
            st.error(f"Failed to read CSV: {str(e)}")
    else:
        # Landing state
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; background: rgba(15,23,42,0.5); border-radius: 20px; border: 1px dashed rgba(78,205,196,0.2);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #4ECDC4;">Drop any CSV to unlock your data</h3>
            <p style="color: #64748b; max-width: 500px; margin: 0 auto;">
                Auto-generates distributions, time series, correlation heatmaps, scatter plots, 
                outlier detection, category breakdowns, pie charts, data quality report, 
                AI insights, and PDF export — all instantly.
            </p>
            <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <span style="color: #4ECDC4; font-size: 0.9rem;">✓ 10+ Auto Charts</span>
                <span style="color: #4ECDC4; font-size: 0.9rem;">✓ Correlation Analysis</span>
                <span style="color: #4ECDC4; font-size: 0.9rem;">✓ Outlier Detection</span>
                <span style="color: #4ECDC4; font-size: 0.9rem;">✓ AI Deep Dive</span>
                <span style="color: #4ECDC4; font-size: 0.9rem;">✓ PDF Export</span>
                <span style="color: #4ECDC4; font-size: 0.9rem;">✓ Data Quality Score</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
