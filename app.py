import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import zipfile
import os

# Extract model if not already extracted
if not os.path.exists("best_crop_classifier.pkl"):
    with zipfile.ZipFile("best_crop_classifier.zip", "r") as zip_ref:
        zip_ref.extractall()

st.set_page_config(
    page_title="Smart Crop Recommendation — Bangladesh",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 2rem !important; max-width: 1200px !important; }
.hero {
    background: linear-gradient(135deg, #1a3c2e 0%, #2d6a4f 50%, #40916c 100%);
    border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.12); color: #b7e4c7;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    padding: 4px 14px; border-radius: 99px; border: 1px solid rgba(255,255,255,0.2); margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 700;
    color: #ffffff; margin: 0 0 0.5rem; line-height: 1.2;
}
.hero-sub { color: #b7e4c7; font-size: 0.95rem; font-weight: 300; margin: 0 0 1.5rem; }
.hero-stats { display: flex; gap: 2rem; flex-wrap: wrap; }
.hero-stat-val { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; color: #fff; }
.hero-stat-label { font-size: 0.7rem; color: #95d5b2; text-transform: uppercase; letter-spacing: 0.08em; }
.section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #2d6a4f; margin-bottom: 0.4rem; display: block;
}
.info-box {
    background: #edf7f1; border-left: 3px solid #2d6a4f; border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #1a3c2e;
}
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1rem; }
.stat-chip {
    background: #f0f7f2; border: 1px solid #c7e8d0; border-radius: 8px;
    padding: 0.4rem 0.8rem; font-size: 0.78rem; color: #1a3c2e;
}
.stat-chip b { color: #2d6a4f; }
.rank1-card {
    background: linear-gradient(135deg, #1a3c2e, #2d6a4f); border-radius: 18px;
    padding: 1.8rem; color: white; margin-bottom: 1.2rem;
    box-shadow: 0 8px 28px rgba(45,106,79,0.3); position: relative;
}
.best-badge {
    position: absolute; top: 1.2rem; right: 1.4rem;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    color: #b7e4c7; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.08em; padding: 3px 10px; border-radius: 99px;
}
.crop-name-big {
    font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: #fff; margin: 0 0 0.2rem;
}
.conf-label { font-size: 0.8rem; color: #95d5b2; margin-bottom: 0.3rem; }
.bar-bg { background: rgba(255,255,255,0.15); border-radius: 99px; height: 7px; margin-bottom: 1.4rem; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #95d5b2, #52b788); }
.metrics-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem; }
.metric-box {
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); border-radius: 10px; padding: 0.8rem;
}
.metric-box.wide { grid-column: span 3; }
.metric-lbl { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em; color: #95d5b2; margin-bottom: 3px; }
.metric-val { font-size: 0.9rem; font-weight: 600; color: #fff; }
.metric-val.big { font-size: 1.2rem; }
.alt-card {
    background: #fff; border-radius: 14px; padding: 1.2rem 1.4rem;
    border: 1.5px solid #e8ede9; margin-bottom: 0.8rem; position: relative;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.alt-rank {
    position: absolute; top: 1rem; right: 1rem; background: #f0f7f2; color: #2d6a4f;
    font-size: 0.68rem; font-weight: 700; padding: 3px 10px; border-radius: 99px; border: 1px solid #c7e8d0;
}
.alt-name { font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 600; color: #1a3c2e; margin: 0 0 0.2rem; }
.alt-conf { font-size: 0.8rem; color: #6b7280; margin-bottom: 0.7rem; }
.alt-conf b { color: #2d6a4f; }
.alt-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; }
.alt-item { background: #f5f7f4; border-radius: 8px; padding: 0.45rem 0.65rem; }
.alt-item-lbl { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 2px; }
.alt-item-val { font-size: 0.82rem; font-weight: 500; color: #374151; }
.input-summary { background: #f0f7f2; border-radius: 10px; padding: 0.8rem 1.1rem; font-size: 0.82rem; color: #374151; margin-top: 0.8rem; }
.input-summary b { color: #1a3c2e; }
.placeholder-box { background: #fff; border: 2px dashed #d1d9d4; border-radius: 16px; padding: 3rem 2rem; text-align: center; color: #9ca3af; }
.placeholder-icon { font-size: 3rem; margin-bottom: 0.8rem; }
.placeholder-title { font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #374151; margin-bottom: 0.4rem; }
.placeholder-text { font-size: 0.85rem; max-width: 260px; margin: 0 auto; }
.footer { text-align: center; padding: 1.5rem; color: #9ca3af; font-size: 0.75rem; border-top: 1px solid #e8ede9; margin-top: 2rem; }
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2d6a4f, #40916c) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.75rem 2rem !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 600 !important; width: 100% !important;
    box-shadow: 0 4px 14px rgba(45,106,79,0.3) !important; margin-top: 0.5rem !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1a3c2e, #2d6a4f) !important;
    box-shadow: 0 6px 20px rgba(45,106,79,0.4) !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# ── Month standardisation (matches notebook Cell 13 exactly) ─────────────────
MONTH_MAP = {
    'jan': 'January',  'feb': 'February', 'mar': 'March',
    'apr': 'April',    'may': 'May',       'jun': 'June',
    'jul': 'July',     'aug': 'August',    'sep': 'September',
    'oct': 'October',  'nov': 'November',  'dec': 'December',
}
 
def std_month(v):
    v = str(v).strip()
    for sh, full in MONTH_MAP.items():
        if v.lower().startswith(sh):
            return full
    return v
 
 
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
 
    def lpkl(fname):
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            st.error(f"❌ Missing file: **{fname}** — please upload it to your repository.")
            st.stop()
        with open(fpath, 'rb') as f:
            return pickle.load(f)
 
    def lcsv(fname):
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            st.error(f"❌ Missing file: **{fname}** — please upload it to your repository.")
            st.stop()
        return pd.read_csv(fpath)
 
    return (
        lpkl('best_crop_classifier.pkl'),
        lpkl('best_yield_regressor.pkl'),
        lpkl('encoders.pkl'),
        lpkl('ds_agg.pkl'),
        lpkl('dist_agg.pkl'),
        lpkl('seas_agg.pkl'),
        lpkl('train_medians.pkl'),
        lcsv('crop_info_lookup.csv'),
        lcsv('SPAS-Dataset-BD.csv'),   # raw dataset for per-crop climate rows
    )
 
 
cls_model, reg_model, enc, ds_agg, dist_agg, seas_agg, TRAIN_MEDIANS, crop_info_raw, df_raw = load_artifacts()
 
le_district = enc['district']
le_season   = enc['season']
le_crop     = enc['crop']
 
# ── Rebuild the cleaned df exactly as in the notebook ────────────────────────
@st.cache_data(show_spinner=False)
def build_df(_df_raw):
    """Replicates notebook Cells 12–14 preprocessing so inference rows match training."""
    df = _df_raw.copy()
    df = df[df['Crop Name'] != '#REF!']
    df = df.dropna(subset=['Season', 'District', 'Crop Name'])
    df['Area']     = pd.to_numeric(df['Area'],     errors='coerce').fillna(0)
    df['AP Ratio'] = pd.to_numeric(df['AP Ratio'], errors='coerce')
    df['AP Ratio'] = df['AP Ratio'].fillna(df['AP Ratio'].median())
    df = df[df['Area'] > 0].reset_index(drop=True)
    for col in ['District', 'Season', 'Crop Name', 'Transplant', 'Harvest', 'Growth']:
        df[col] = df[col].astype(str).str.strip()
 
    # Humidity column swap — exactly as notebook Cell 12
    df.rename(columns={
        'Max Relative Humidity': '_tmp',
        'Min Relative Humidity': 'Max Relative Humidity',
    }, inplace=True)
    df.rename(columns={'_tmp': 'Min Relative Humidity'}, inplace=True)
 
    # Standardise transplant month labels (notebook Cell 13)
    df['Transplant'] = df['Transplant'].apply(std_month)
 
    # Derived features (notebook Cell 14)
    df['Temp_Range']     = df['Max Temp'] - df['Min Temp']
    df['Humidity_Range'] = df['Max Relative Humidity'] - df['Min Relative Humidity']
    df['Log_Area']       = np.log1p(df['Area'])
    df['AP_x_Area']      = df['AP Ratio'] * df['Log_Area']
    df['Temp_x_Hum']     = df['Avg Temp'] * df['Avg Humidity']
    df['Yield_per_Area'] = df['Production'] / (df['Area'] + 1)
 
    return df
 
 
df = build_df(df_raw)
 
# ── Rebuild crop_info from the cleaned df (notebook Cell 20) ─────────────────
@st.cache_data(show_spinner=False)
def build_crop_info(_df):
    return _df.groupby('Crop Name').agg(
        Transplant=('Transplant', lambda x: x.mode()[0]),
        Harvest=('Harvest',       lambda x: x.mode()[0]),
        Growth=('Growth',         lambda x: x.mode()[0]),
    ).reset_index()
 
 
crop_info = build_crop_info(df)
 
# ── Feature sets (identical to notebook Cells 18) ────────────────────────────
CLASSIF_FEATURES = [
    'District_Enc', 'Season_Enc',
    'Avg Temp', 'Avg Humidity', 'Max Temp', 'Min Temp',
    'Max Relative Humidity', 'Min Relative Humidity',
    'Temp_Range', 'Humidity_Range', 'Temp_x_Hum',
    'AP Ratio', 'Log_Area', 'AP_x_Area',
    'ds_avg_temp', 'ds_avg_hum', 'ds_max_temp', 'ds_min_temp',
    'ds_total_area', 'ds_n_crops', 'ds_mean_ap', 'ds_std_ap',
    'dist_avg_temp', 'dist_avg_hum', 'dist_n_crops', 'dist_mean_ap',
    'season_avg_temp', 'season_avg_hum', 'season_n_crops',
]
REGRESS_FEATURES = CLASSIF_FEATURES + ['Crop_Enc', 'Yield_per_Area']
 
 
# ── Inference function — exact copy of notebook Cell 44 recommend_crop() ─────
def recommend(district: str, season: str, area: float, top_n: int = 3):
    district = district.strip()
    season   = season.strip()
 
    dist_enc = int(le_district.transform([district])[0])
    seas_enc = int(le_season.transform([season])[0])
 
    # Training-derived aggregate context (same as notebook)
    ds_row   = ds_agg[(ds_agg['District'] == district) & (ds_agg['Season'] == season)]
    dist_row = dist_agg[dist_agg['District'] == district]
    seas_row = seas_agg[seas_agg['Season'] == season]
 
    def safe(row, col, fallback):
        return float(row[col].values[0]) if not row.empty and col in row else fallback
 
    ds_avg_temp   = safe(ds_row,   'ds_avg_temp',    25.0)
    ds_avg_hum    = safe(ds_row,   'ds_avg_hum',     75.0)
    ds_max_temp   = safe(ds_row,   'ds_max_temp',    32.0)
    ds_min_temp   = safe(ds_row,   'ds_min_temp',    18.0)
    ds_total_area = safe(ds_row,   'ds_total_area', 5000.0)
    ds_n_crops    = safe(ds_row,   'ds_n_crops',     10.0)
    ds_mean_ap    = safe(ds_row,   'ds_mean_ap',      1.0)
    ds_std_ap     = safe(ds_row,   'ds_std_ap',       0.1)
    dist_avg_temp = safe(dist_row, 'dist_avg_temp',  25.0)
    dist_avg_hum  = safe(dist_row, 'dist_avg_hum',   75.0)
    dist_n_crops  = safe(dist_row, 'dist_n_crops',   20.0)
    dist_mean_ap  = safe(dist_row, 'dist_mean_ap',    1.0)
    seas_avg_temp = safe(seas_row, 'season_avg_temp', 25.0)
    seas_avg_hum  = safe(seas_row, 'season_avg_hum',  75.0)
    seas_n_crops  = safe(seas_row, 'season_n_crops',  20.0)
 
    log_area = float(np.log1p(area))
 
    # ── KEY FIX: use per-crop rows from the real dataset (notebook Cell 44) ──
    # The notebook iterates over actual rows for the district+season subset,
    # reading per-crop Avg Temp, Avg Humidity, Max/Min Temp, Max/Min RH, AP Ratio.
    # The old app.py used only the district-season aggregate for all crops → wrong.
    subset = df[(df['District'] == district) & (df['Season'] == season)]
    if subset.empty:
        return []
 
    records = []
    for _, row in subset.drop_duplicates('Crop Name').iterrows():
        avg_t  = float(row['Avg Temp'])
        avg_h  = float(row['Avg Humidity'])
        max_t  = float(row['Max Temp'])
        min_t  = float(row['Min Temp'])
        max_rh = float(row['Max Relative Humidity'])
        min_rh = float(row['Min Relative Humidity'])
        ap     = float(row['AP Ratio'])
        ypa    = float(row['Yield_per_Area']) if 'Yield_per_Area' in row else 0.0
 
        records.append({
            'Crop Name':              row['Crop Name'],
            'Yield_per_Area':         ypa,
            'District_Enc':           dist_enc,
            'Season_Enc':             seas_enc,
            'Avg Temp':               avg_t,
            'Avg Humidity':           avg_h,
            'Max Temp':               max_t,
            'Min Temp':               min_t,
            'Max Relative Humidity':  max_rh,
            'Min Relative Humidity':  min_rh,
            'Temp_Range':             max_t - min_t,
            'Humidity_Range':         max_rh - min_rh,
            'Temp_x_Hum':             avg_t * avg_h,
            'AP Ratio':               ap,
            'Log_Area':               log_area,
            'AP_x_Area':              ap * log_area,
            'ds_avg_temp':            ds_avg_temp,
            'ds_avg_hum':             ds_avg_hum,
            'ds_max_temp':            ds_max_temp,
            'ds_min_temp':            ds_min_temp,
            'ds_total_area':          ds_total_area,
            'ds_n_crops':             ds_n_crops,
            'ds_mean_ap':             ds_mean_ap,
            'ds_std_ap':              ds_std_ap,
            'dist_avg_temp':          dist_avg_temp,
            'dist_avg_hum':           dist_avg_hum,
            'dist_n_crops':           dist_n_crops,
            'dist_mean_ap':           dist_mean_ap,
            'season_avg_temp':        seas_avg_temp,
            'season_avg_hum':         seas_avg_hum,
            'season_n_crops':         seas_n_crops,
        })
 
    if not records:
        return []
 
    df_cand = pd.DataFrame(records)
    X_cand  = df_cand[CLASSIF_FEATURES].fillna(pd.Series(TRAIN_MEDIANS)).values
 
    # Score each candidate crop (same as notebook)
    proba        = cls_model.predict_proba(X_cand)
    classes_list = list(cls_model.classes_)
 
    crop_scores = []
    for i, crop in enumerate(df_cand['Crop Name']):
        try:
            ev    = int(le_crop.transform([crop])[0])
            score = float(proba[i][classes_list.index(ev)]) if ev in classes_list else 0.0
        except Exception:
            score = 0.0
        crop_scores.append((crop, score))
 
    crop_scores.sort(key=lambda x: x[1], reverse=True)
 
    # ── Regression for top-n crops (notebook only runs regression for best crop,
    #    but we run it for all top_n so the web UI can show production for each) ──
    results = []
    for rank, (crop, score) in enumerate(crop_scores[:top_n], 1):
        try:
            crop_enc_r = int(le_crop.transform([crop])[0])
            bc_row     = df_cand[df_cand['Crop Name'] == crop].iloc[0]
            reg_dict   = {f: bc_row[f] for f in CLASSIF_FEATURES}
            reg_dict['Crop_Enc']       = crop_enc_r
            reg_dict['Yield_per_Area'] = float(bc_row['Yield_per_Area'])
            X_reg = pd.DataFrame([reg_dict])[REGRESS_FEATURES].fillna(
                        pd.Series(TRAIN_MEDIANS)).values
            prod = max(0.0, float(reg_model.predict(X_reg)[0]))
        except Exception:
            prod = 0.0
 
        # Calendar lookup from freshly-rebuilt crop_info (notebook Cell 20)
        cal = crop_info[crop_info['Crop Name'] == crop]
        results.append({
            'rank':       rank,
            'crop':       crop,
            'score':      score,
            'production': prod,
            'transplant': cal['Transplant'].values[0] if not cal.empty else 'N/A',
            'growth':     cal['Growth'].values[0]     if not cal.empty else 'N/A',
            'harvest':    cal['Harvest'].values[0]    if not cal.empty else 'N/A',
        })
 
    return results
 
 
# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌾 Smart Agro Assistant Bangladesh </div>
    <div class="hero-title">Smart Crop Recommendation System</div>
    <p class="hero-sub">AI-powered crop advisory using climate and regional data from 64 districts across Bangladesh.</p>
    <div class="hero-stats">
        <div><div class="hero-stat-val">72</div><div class="hero-stat-label">Crop Varieties</div></div>
        <div><div class="hero-stat-val">64</div><div class="hero-stat-label">Districts</div></div>
        <div><div class="hero-stat-val">92%</div><div class="hero-stat-label">Model Accuracy</div></div>
        <div><div class="hero-stat-val">4,190</div><div class="hero-stat-label">Training Records</div></div>
    </div>
</div>
""", unsafe_allow_html=True)
 
left, right = st.columns([1, 1.6], gap="large")
 
with left:
    st.markdown('<span class="section-label">Input Parameters</span>', unsafe_allow_html=True)
    districts = sorted(le_district.classes_.tolist())
    seasons   = sorted(le_season.classes_.tolist())
 
    district = st.selectbox("📍 District", options=districts,
                             index=districts.index("Dhaka") if "Dhaka" in districts else 0)
    season   = st.selectbox("🗓️ Growing Season", options=seasons)
    area     = st.number_input("📐 Land Area (hectares)",
                               min_value=0.1, max_value=50000.0,
                               value=5.0, step=1")
 
    season_info = {
        "Kharif 1": "☀️ March – June. Early monsoon. High temperature, increasing rainfall.",
        "Kharif 2": "🌧️ July – October. Main monsoon. High humidity and rainfall.",
        "Rabi":     "❄️ November – February. Winter/dry season. Cool temperatures, low rainfall.",
    }
    st.markdown(f'<div class="info-box">{season_info.get(season,"")}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row">
        <div class="stat-chip">Classifier: <b>Random Forest</b></div>
        <div class="stat-chip">Accuracy: <b>92.00%</b></div>
        <div class="stat-chip">CV: <b>91.71%</b></div>
        <div class="stat-chip">R²: <b>0.9202</b></div>
    </div>""", unsafe_allow_html=True)
 
    run = st.button("🌱  Get Crop Recommendation", use_container_width=True)
 
with right:
    st.markdown('<span class="section-label">Recommendation Results</span>', unsafe_allow_html=True)
 
    if not run:
        st.markdown("""
        <div class="placeholder-box">
            <div class="placeholder-icon">🌾</div>
            <div class="placeholder-title">Ready to recommend</div>
            <p class="placeholder-text">Select your district, season, and land area, then click the button.</p>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Analysing climate and regional data…"):
            try:
                results = recommend(district, season, area)
            except Exception as e:
                st.error(f"Error: {e}")
                results = []
 
        if results:
            b   = results[0]
            bar = min(int(b['score'] * 100), 100)
            st.markdown(f"""
            <div class="rank1-card">
                <div class="best-badge">🥇 Best Match</div>
                <div class="crop-name-big">{b['crop']}</div>
                <div class="conf-label">Confidence — {b['score']*100:.1f}%</div>
                <div class="bar-bg"><div class="bar-fill" style="width:{bar}%"></div></div>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-lbl">🌱 Transplant</div>
                        <div class="metric-val">{b['transplant']}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-lbl">🌿 Growth Period</div>
                        <div class="metric-val">{b['growth']}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-lbl">🌾 Harvest</div>
                        <div class="metric-val">{b['harvest']}</div>
                    </div>
                    <div class="metric-box wide">
                        <div class="metric-lbl">📦 Estimated Production</div>
                        <div class="metric-val big">{b['production']:,.0f} metric tons</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
 
            if len(results) > 1:
                st.markdown("""<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#6b7280;margin:0.8rem 0 0.5rem;">
                    Other Suitable Crops</p>""", unsafe_allow_html=True)
                for r in results[1:]:
                    em = "🥈" if r['rank'] == 2 else "🥉"
                    st.markdown(f"""
                    <div class="alt-card">
                        <div class="alt-rank">{em} #{r['rank']} Match</div>
                        <div class="alt-name">{r['crop']}</div>
                        <div class="alt-conf">
                            Confidence: <b>{r['score']*100:.1f}%</b> &nbsp;·&nbsp;
                            Est. Production: <b>{r['production']:,.0f} MT</b>
                        </div>
                        <div class="alt-grid">
                            <div class="alt-item">
                                <div class="alt-item-lbl">🌱 Transplant</div>
                                <div class="alt-item-val">{r['transplant']}</div>
                            </div>
                            <div class="alt-item">
                                <div class="alt-item-lbl">🌿 Growth</div>
                                <div class="alt-item-val">{r['growth']}</div>
                            </div>
                            <div class="alt-item">
                                <div class="alt-item-lbl">🌾 Harvest</div>
                                <div class="alt-item-val">{r['harvest']}</div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
 
            st.markdown(f"""
            <div class="input-summary">
                📍 <b>{district}</b> &nbsp;·&nbsp; 🗓️ <b>{season}</b> &nbsp;·&nbsp; 📐 <b>{area:.1f} ha</b>
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Please try a different district or season.")
 
st.markdown("""
<div class="footer">
    Smart Crop Recommendation System &nbsp;·&nbsp; Bangladesh &nbsp;·&nbsp;
    Random Forest + XGBoost &nbsp;·&nbsp; SPAS-BD Dataset &nbsp;·&nbsp; 64 Districts · 72 Crops<br><br>
    ⚠️ This system provides data-driven suggestions based on historical agricultural records.
    Always consult local agricultural extension officers before making final planting decisions.
</div>""", unsafe_allow_html=True)
