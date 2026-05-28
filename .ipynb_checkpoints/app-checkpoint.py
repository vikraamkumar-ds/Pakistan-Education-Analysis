import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
import json
import os
import sys
import types
import warnings
warnings.filterwarnings('ignore')

# ── Suppress sklearn version mismatch warning ──────────────────────
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass

# ── Mock shap if not installed so pkl can still load ──────────────
if 'shap' not in sys.modules:
    try:
        import shap  # try real import first
    except ImportError:
        # Build a minimal mock so joblib can deserialise the bundle
        class _AutoMod(types.ModuleType):
            def __getattr__(self, name):
                child = _AutoMod(f"{self.__name__}.{name}")
                child.__path__ = []
                setattr(self, name, child)
                sys.modules[child.__name__] = child
                return child
        _shap = _AutoMod('shap')
        _shap.__path__ = []
        sys.modules['shap'] = _shap
        # register common sub-modules pickle looks for
        for _sub in ['explainers', 'explainers._tree', 'explainers.tree',
                     'explainers._linear', 'explainers._deep',
                     'explainers._gradient', 'plots', 'utils', 'maskers',
                     'benchmark', 'models']:
            _m = _AutoMod(f'shap.{_sub}')
            _m.__path__ = []
            sys.modules[f'shap.{_sub}'] = _m

st.set_page_config(
    page_title='Pakistan Education Dashboard',
    page_icon='🏫',
    layout='wide'
)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = 'data/aser_2023_clean.csv'
    if not os.path.exists(path):
        st.error(f'Dataset not found at {path}.')
        return None

    df = pd.read_csv(path)

    # Normalise column names
    if 'nonfunctional' not in df.columns and 'school_nonfunctional' in df.columns:
        df['nonfunctional'] = df['school_nonfunctional']

    # School gender label
    gender_map = {1.0: 'Boys Only', 2.0: 'Girls Only', 3.0: 'Mixed', 4.0: 'Other'}
    df['school_gender'] = df['S002'].map(gender_map).fillna('Other')

    # School level label
    df['school_level'] = df['STYPE'].map({1: 'Primary', 2: 'Middle/Higher'})

    # Strip whitespace
    df['RNAME'] = df['RNAME'].str.strip()
    df['DNAME'] = df['DNAME'].str.strip()

    # district_clean for GeoJSON join
    df['district_clean'] = df['DNAME'].str.lower().str.strip()

    return df


@st.cache_resource
def load_model():
    path = 'models/best_model_bundle.pkl'
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        bundle['province_classes'] = [p.strip() for p in bundle['province_classes']]
        return bundle
    except Exception as e:
        st.warning(f"⚠️ Model loaded with warning (safe to ignore): {e}")
        try:
            # second attempt after mock already registered above
            bundle = joblib.load(path)
            bundle['province_classes'] = [p.strip() for p in bundle['province_classes']]
            return bundle
        except Exception as e2:
            st.error(f"❌ Could not load model: {e2}. Run: pip install shap scikit-learn==1.7.2")
            return None


@st.cache_data
def load_geojson():
    path = 'data/pakistan_districts.geojson'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


df     = load_data()
bundle = load_model()
geo    = load_geojson()

if df is None:
    st.stop()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('🏫 Pakistan Education')
    st.caption('ASER 2023 | Vikram Kumar | Sindh University')
    st.divider()

    provinces   = ['All'] + sorted(df['RNAME'].dropna().unique().tolist())
    sel_province = st.selectbox('Province', provinces)

    levels      = ['All'] + sorted(df['school_level'].dropna().unique().tolist())
    sel_level   = st.selectbox('School Level', levels)

    genders     = ['All'] + sorted([g for g in df['school_gender'].dropna().unique() if g != 'Other'])
    sel_gender  = st.selectbox('School Gender Type', genders)

    st.divider()
    st.caption('Non-functional = school open but broken, or closed temporarily/permanently')

# Apply filters
fdf = df.copy()
if sel_province != 'All':
    fdf = fdf[fdf['RNAME'] == sel_province]
if sel_level != 'All':
    fdf = fdf[fdf['school_level'] == sel_level]
if sel_gender != 'All':
    fdf = fdf[fdf['school_gender'] == sel_gender]

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.title('🏫 Pakistan Rural Education Dashboard')
st.caption('ASER 2023 — 6,095 rural schools | 7 provinces | 151 districts | Predicting non-functional schools')
st.divider()

# ─────────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────────
total     = len(fdf)
nonfunc   = int(fdf['nonfunctional'].sum())
func      = total - nonfunc
rate      = fdf['nonfunctional'].mean() * 100 if total > 0 else 0
districts = fdf['DNAME'].nunique()

nat_rate  = df['nonfunctional'].mean() * 100
delta     = rate - nat_rate

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Total Schools',       f'{total:,}')
c2.metric('Functional',          f'{func:,}',    f'{100 - rate:.1f}%')
c3.metric('Non-Functional',      f'{nonfunc:,}',  f'{rate:.1f}%')
c4.metric('Non-Functional Rate', f'{rate:.1f}%',
          f'{delta:+.1f}pp vs national' if sel_province != 'All' else None)
c5.metric('Districts Covered',   f'{districts}')

st.divider()

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '📊 Overview',
    '👫 Gender Gap',
    '🗺️ District Analysis',
    '🤖 ML Predictor'
])

# ─────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader('Province-level Overview')

    col1, col2 = st.columns(2)

    with col1:
        prov_df = (
            df.groupby('RNAME')['nonfunctional']
            .agg(['mean', 'count', 'sum'])
            .reset_index()
        )
        prov_df.columns = ['Province', 'Rate', 'Total', 'NonFunc']
        prov_df['Rate'] = (prov_df['Rate'] * 100).round(1)
        prov_df = prov_df.sort_values('Rate', ascending=True)

        fig = px.bar(
            prov_df, x='Rate', y='Province', orientation='h',
            color='Rate', color_continuous_scale='RdYlGn_r',
            range_color=[0, 100], text='Rate',
            title='Non-functional rate by province (%)',
            labels={'Rate': 'Non-functional rate (%)', 'Province': ''}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # School status breakdown (S00 values)
        s00_map = {
            1: 'Open functional',
            2: 'Open non-functional',
            3: 'Closed temporarily',
            4: 'Closed permanently'
        }
        if 'S00' in fdf.columns:
            s00_counts = fdf['S00'].map(s00_map).value_counts().reset_index()
            s00_counts.columns = ['Status', 'Count']
            fig2 = px.pie(
                s00_counts, names='Status', values='Count',
                color='Status',
                color_discrete_map={
                    'Open functional'    : '#1D9E75',
                    'Open non-functional': '#EF9F27',
                    'Closed temporarily' : '#D85A30',
                    'Closed permanently' : '#E24B4A'
                },
                title='School status breakdown'
            )
        else:
            pie_df = pd.DataFrame({
                'Status': ['Functional', 'Non-functional'],
                'Count' : [func, nonfunc]
            })
            fig2 = px.pie(
                pie_df, names='Status', values='Count',
                color='Status',
                color_discrete_map={'Functional': '#1D9E75', 'Non-functional': '#E24B4A'},
                title='Functional vs non-functional'
            )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Infrastructure availability
    st.subheader('Infrastructure Availability')
    infra_cols = {
        'S03a1': 'Teacher present',
        'S03d1': 'Drinking water',
        'S03e1': 'Toilet facility',
        'S03f1': 'Electricity',
    }

    existing_infra = {k: v for k, v in infra_cols.items() if k in fdf.columns}
    if existing_infra:
        infra_data = []
        for col, label in existing_infra.items():
            avail = fdf[col].fillna(0).mean() * 100
            infra_data.append({'Facility': label, 'Available (%)': round(avail, 1)})

        infra_df = pd.DataFrame(infra_data).sort_values('Available (%)', ascending=True)

        fig3 = px.bar(
            infra_df, x='Available (%)', y='Facility', orientation='h',
            color='Available (%)', color_continuous_scale='RdYlGn',
            range_color=[0, 100], text='Available (%)',
            title='Infrastructure availability (%)',
            labels={'Available (%)': '% of schools with facility', 'Facility': ''}
        )
        fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig3.update_layout(coloraxis_showscale=False, height=280)
        st.plotly_chart(fig3, use_container_width=True)

    # Key findings
    st.subheader('Key Findings')
    col_a, col_b, col_c = st.columns(3)
    col_a.info('**50.3%** of rural Pakistani schools are non-functional nationally')
    col_b.warning('**Sindh** sits at 39.3% — below national average but still critical')
    col_c.error('**Gilgit-Baltistan** is worst at 68.9% non-functional rate')


# ─────────────────────────────────────────────────────────────────
# TAB 2 — GENDER GAP
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader('Gender Gap — Boys vs Girls Schools')

    gender_df = df[df['school_gender'].isin(['Boys Only', 'Girls Only'])]

    col1, col2 = st.columns(2)

    with col1:
        # National gender gap
        nat_gender = (
            gender_df.groupby('school_gender')['nonfunctional']
            .agg(['mean', 'count'])
            .reset_index()
        )
        nat_gender.columns = ['Gender', 'Rate', 'Count']
        nat_gender['Rate'] = (nat_gender['Rate'] * 100).round(1)

        fig = px.bar(
            nat_gender, x='Gender', y='Rate',
            color='Gender',
            color_discrete_map={'Boys Only': '#4C72B0', 'Girls Only': '#DD8452'},
            text='Rate',
            title='Non-functional rate — boys vs girls (national)',
            labels={'Rate': 'Non-functional rate (%)', 'Gender': ''}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=350, yaxis_range=[0, 80])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Gender gap by province
        prov_gender = (
            gender_df.groupby(['RNAME', 'school_gender'])['nonfunctional']
            .mean()
            .reset_index()
        )
        prov_gender['nonfunctional'] = (prov_gender['nonfunctional'] * 100).round(1)
        prov_gender.columns = ['Province', 'Gender', 'Rate']

        fig2 = px.bar(
            prov_gender, x='Province', y='Rate', color='Gender',
            barmode='group',
            color_discrete_map={'Boys Only': '#4C72B0', 'Girls Only': '#DD8452'},
            title='Non-functional rate by province and gender',
            labels={'Rate': 'Non-functional rate (%)', 'Province': ''}
        )
        fig2.update_layout(height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader('Sindh District Deep-Dive — Gender Gap')

    sindh_df = df[
        (df['RNAME'] == 'SINDH') &
        (df['school_gender'].isin(['Boys Only', 'Girls Only']))
    ]

    if len(sindh_df) > 0:
        sindh_gender = (
            sindh_df.groupby(['DNAME', 'school_gender'])['nonfunctional']
            .mean()
            .reset_index()
        )
        sindh_gender['nonfunctional'] = (sindh_gender['nonfunctional'] * 100).round(1)
        sindh_gender.columns = ['District', 'Gender', 'Rate']

        fig3 = px.bar(
            sindh_gender, x='District', y='Rate', color='Gender',
            barmode='group',
            color_discrete_map={'Boys Only': '#4C72B0', 'Girls Only': '#DD8452'},
            title='Sindh — non-functional rate by district and gender',
            labels={'Rate': 'Non-functional rate (%)', 'District': ''}
        )
        fig3.update_layout(height=420, xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info('Sindh gender data not available with current filters.')


# ─────────────────────────────────────────────────────────────────
# TAB 3 — DISTRICT ANALYSIS
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader('District-level Analysis')

    drill_prov = st.selectbox(
        'Select province',
        sorted(df['RNAME'].dropna().unique().tolist()),
        index=sorted(df['RNAME'].dropna().unique().tolist()).index('SINDH')
        if 'SINDH' in df['RNAME'].unique() else 0
    )

    drill_df = df[df['RNAME'] == drill_prov]

    dist_data = (
        drill_df.groupby('DNAME')
        .agg(total=('nonfunctional', 'count'),
             rate=('nonfunctional', 'mean'),
             nonfunc=('nonfunctional', 'sum'))
        .reset_index()
    )
    dist_data['rate_pct'] = (dist_data['rate'] * 100).round(1)
    dist_data = dist_data.sort_values('rate_pct', ascending=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            dist_data, x='rate_pct', y='DNAME',
            orientation='h',
            color='rate_pct',
            color_continuous_scale='RdYlGn_r',
            range_color=[0, 100], text='rate_pct',
            title=f'Non-functional school rate by district — {drill_prov}',
            labels={'rate_pct': 'Non-functional rate (%)', 'DNAME': 'District'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            coloraxis_showscale=False,
            height=max(350, len(dist_data) * 28)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f'**{drill_prov} Summary**')
        st.metric('Total schools',  f'{len(drill_df):,}')
        st.metric('Non-functional', f'{int(drill_df["nonfunctional"].sum()):,}')
        st.metric('Overall rate',   f'{drill_df["nonfunctional"].mean()*100:.1f}%')
        st.metric('Districts',      drill_df['DNAME'].nunique())
        st.divider()
        worst = dist_data.iloc[-1]
        best  = dist_data.iloc[0]
        st.markdown(f'🔴 **Worst:** {worst["DNAME"]} ({worst["rate_pct"]:.1f}%)')
        st.markdown(f'🟢 **Best:**  {best["DNAME"]} ({best["rate_pct"]:.1f}%)')

    # Choropleth map — uses pakistan_districts.geojson if available
    st.divider()
    st.subheader('District Choropleth Map')

    if geo is not None:
        try:
            # ── Detect GeoJSON level (division vs district) ──────────
            geo_names = [f['properties'].get('NAME_2', '') for f in geo['features']]
            geo_count = len(geo['features'])

            # Build district-level aggregate
            geo_df = (
                df.groupby('DNAME')['nonfunctional']
                .mean()
                .reset_index()
            )
            geo_df['rate_pct'] = (geo_df['nonfunctional'] * 100).round(1)
            geo_df['district_clean'] = geo_df['DNAME'].str.lower().str.strip()

            if geo_count <= 40:
                # ── Division-level GeoJSON (current file has 32 divisions) ──
                # Map each district to its division so we can aggregate
                DISTRICT_TO_DIVISION = {
                    # Punjab
                    'LAHORE': 'Lahore', 'SHEIKHUPURA': 'Lahore', 'NANKANA SAHIB': 'Lahore',
                    'GUJRANWALA': 'Gujranwala', 'HAFIZABAD': 'Gujranwala', 'MANDI BAHAUDDIN': 'Gujranwala',
                    'SIALKOT': 'Gujranwala', 'NAROWAL': 'Gujranwala', 'GUJRAT': 'Gujranwala',
                    'FAISALABAD': 'Faisalabad', 'TOBA TEK SINGH': 'Faisalabad', 'JHANG': 'Faisalabad', 'CHINIOT': 'Faisalabad',
                    'RAWALPINDI': 'Rawalpindi', 'ATTOCK': 'Rawalpindi', 'CHAKWAL': 'Rawalpindi', 'JHELUM': 'Rawalpindi',
                    'SARGODHA': 'Sargodha', 'KHUSHAB': 'Sargodha', 'MIANWALI': 'Sargodha', 'BHAKKAR': 'Sargodha',
                    'MULTAN': 'Multan', 'KHANEWAL': 'Multan', 'LODHRAN': 'Multan', 'VEHARI': 'Multan',
                    'BAHAWALPUR': 'Bahawalpur', 'BAHAWALNAGAR': 'Bahawalpur', 'RAHIM YAR KHAN': 'Bahawalpur',
                    'DERA GHAZI KHAN': 'DeraGhaziKhan', 'MUZAFFARGARH': 'DeraGhaziKhan',
                    'LAYYAH': 'DeraGhaziKhan', 'RAJANPUR': 'DeraGhaziKhan',
                    'KASUR': 'Lahore', 'OKARA': 'Lahore', 'SAHIWAL': 'Multan',
                    'PAKPATTAN': 'Multan',
                    # Sindh
                    'KARACHI MALIR': 'Karachi', 'KARACHI WEST': 'Karachi',
                    'HYDERABAD': 'Hyderabad', 'DADU': 'Hyderabad', 'MATIARI': 'Hyderabad',
                    'TANDO MUHAMMAD KHAN': 'Hyderabad', 'TANDO ALLAHYAR': 'Hyderabad', 'JAMSHORO': 'Hyderabad',
                    'SUKKUR': 'Sukkur', 'KHAIRPUR': 'Sukkur', 'GHOTKI': 'Sukkur',
                    'LARKANA': 'Larkana', 'SHIKARPUR': 'Larkana', 'JACOBABAD': 'Larkana',
                    'KAMBAR SHAHDAD KOT': 'Larkana', 'KASHMORE': 'Larkana', 'KASHMOR': 'Larkana',
                    'MIRPUR KHAS': 'MirpurKhas', 'SANGHAR': 'MirpurKhas', 'THARPARKAR': 'MirpurKhas',
                    'UMER KOT': 'MirpurKhas',
                    'BADIN': 'Hyderabad', 'THATTA': 'Hyderabad', 'SUJAWAL': 'Hyderabad',
                    'NAUSHAHRO FEROZE': 'Sukkur', 'SHAHEED BENAZIRABAD': 'Sukkur',
                    # KPK
                    'PESHAWAR': 'Peshawar', 'CHARSADDA': 'Peshawar', 'NOWSHERA': 'Peshawar', 'KOHAT': 'Kohat',
                    'HANGU': 'Kohat', 'KARAK': 'Kohat', 'BANNU': 'Bannu', 'LAKKI MARWAT': 'Bannu',
                    'TANK': 'Bannu', 'DERA ISMAIL KHAN': 'DeraIsmailKhan',
                    'MARDAN': 'Mardan', 'SWABI': 'Mardan',
                    'MALAKAND PROTECTED AREA': 'Malakand', 'LOWER DIR': 'Malakand', 'UPPER DIR': 'Malakand',
                    'SWAT': 'Malakand', 'SHANGLA': 'Malakand', 'BUNER': 'Malakand', 'CHITRAL': 'Malakand',
                    'LOWER CHITRAL': 'Malakand', 'UPPER CHITRAL': 'Malakand',
                    'ABBOTTABAD': 'Hazara', 'MANSEHRA': 'Hazara', 'BATAGRAM': 'Hazara',
                    'KOHISTAN': 'Hazara', 'LOWER KOHISTAN': 'Hazara', 'UPPER KOHISTAN': 'Hazara',
                    'TORGHAR': 'Hazara', 'HARIPUR': 'Hazara',
                    'MOHMAND AGENCY': 'Malakand', 'BAJAUR AGENCY': 'Malakand',
                    'ORAKZAI AGENCY': 'Kohat', 'KHYBER AGENCY': 'Peshawar',
                    'KURRAM AGENCY': 'Kohat', 'NORTH WAZIRISTAN AGENCY': 'Bannu',
                    'SOUTH WAZIRISTAN AGENCY': 'DeraIsmailKhan',
                    # Balochistan
                    'QUETTA': 'Quetta', 'PISHIN': 'Quetta', 'CHAMAN': 'Quetta',
                    'QILLA ABDULLAH': 'Quetta', 'MASTUNG': 'Quetta',
                    'KALAT': 'Kalat', 'KHUZDAR': 'Kalat', 'AWARAN': 'Kalat',
                    'LASBELA': 'Kalat', 'HUB': 'Kalat',
                    'MAKRAN': 'Makran', 'KECH': 'Makran', 'PANJGUR': 'Makran', 'GWADAR': 'Makran',
                    'NASIRABAD': 'Nasirabad', 'JAFARABAD': 'Nasirabad', 'SOHBATPUR': 'Nasirabad',
                    'KACHHI': 'Nasirabad', 'JHAL MAGSI': 'Nasirabad',
                    'SIBI': 'Sibi', 'HARNAI': 'Sibi', 'BARKHAN': 'Sibi',
                    'ZHOB': 'Zhob', 'SHERANI': 'Zhob', 'MUSAKHEL': 'Zhob', 'LORALAI': 'Zhob',
                    'KILLA SAIFULLAH': 'Zhob', 'ZIARAT': 'Sibi', 'DUKKI': 'Zhob',
                    'KHARAN': 'Makran', 'WASHUK': 'Makran', 'NUSHKI': 'Quetta',
                    'CHAGAI': 'Quetta', 'SURAB': 'Kalat', 'USTA MUHAMMAD': 'Nasirabad',
                    'KOHLU': 'Sibi', 'DERA BUGTI': 'Nasirabad',
                    # AJK
                    'BAGH': 'AzadKashmir', 'BHIMBER': 'AzadKashmir', 'HATTIAN': 'AzadKashmir',
                    'HAVELI': 'AzadKashmir', 'KOTLI': 'AzadKashmir', 'MIRPUR': 'AzadKashmir',
                    'MUZAFFARABAD': 'AzadKashmir', 'NEELUM': 'AzadKashmir',
                    'POONCH': 'AzadKashmir', 'SUDHNATI': 'AzadKashmir',
                    # GB
                    'GILGIT': 'NorthernAreas', 'ASTORE': 'NorthernAreas', 'DIAMER': 'NorthernAreas',
                    'GHANCHE': 'NorthernAreas', 'GHIZER': 'NorthernAreas', 'HUNZA-NAGAR': 'NorthernAreas',
                    'NAGAR': 'NorthernAreas', 'SKARDU': 'NorthernAreas', 'SHIGAR': 'NorthernAreas',
                    'KHARMANG': 'NorthernAreas',
                    # ICT
                    'ISLAMABAD': 'Islamabad',
                }

                geo_df['division'] = geo_df['DNAME'].str.strip().map(DISTRICT_TO_DIVISION).fillna('Unknown')
                div_df = (
                    geo_df[geo_df['division'] != 'Unknown']
                    .groupby('division')['rate_pct']
                    .mean()
                    .reset_index()
                )

                fig_map = px.choropleth(
                    div_df,
                    geojson=geo,
                    locations='division',
                    featureidkey='properties.NAME_2',
                    color='rate_pct',
                    color_continuous_scale='RdYlGn_r',
                    range_color=[0, 100],
                    title='Non-functional school rate by division (%) — upgrade GeoJSON for district-level view',
                    labels={'rate_pct': 'Non-functional rate (%)'}
                )
                st.info(
                    "🗺️ **Map is showing division-level data** (your GeoJSON has 32 divisions, not 151 districts). "
                    "To see district-level detail, replace `data/pakistan_districts.geojson` with a "
                    "district-level file from [GADM Level-3](https://gadm.org/download_country.html) "
                    "or [HDX](https://data.humdata.org/dataset/cod-ab-pak), then update "
                    "`featureidkey` in app.py to match the new file's property name."
                )
            else:
                # ── District-level GeoJSON (correct file) ─────────────
                # Try common property names
                sample_props = geo['features'][0]['properties']
                id_key = None
                for candidate in ['NAME_3', 'ADM3_EN', 'DISTRICT', 'NAME_2', 'district']:
                    if candidate in sample_props:
                        id_key = f'properties.{candidate}'
                        break
                if id_key is None:
                    id_key = f'properties.{list(sample_props.keys())[0]}'

                geo_df['geo_name'] = geo_df['DNAME'].str.title()
                fig_map = px.choropleth(
                    geo_df,
                    geojson=geo,
                    locations='geo_name',
                    featureidkey=id_key,
                    color='rate_pct',
                    color_continuous_scale='RdYlGn_r',
                    range_color=[0, 100],
                    title='Non-functional school rate by district (%)',
                    labels={'rate_pct': 'Non-functional rate (%)'}
                )

            fig_map.update_geos(fitbounds='locations', visible=False)
            fig_map.update_layout(height=520, margin={'r': 0, 't': 40, 'l': 0, 'b': 0})
            st.plotly_chart(fig_map, use_container_width=True)

        except Exception as e:
            st.error(f"Map error: {e}")
            st.info("Check that the GeoJSON property names match your district names.")
    else:
        st.info('GeoJSON not found. Place `pakistan_districts.geojson` in the `data/` folder.')


# ─────────────────────────────────────────────────────────────────
# TAB 4 — ML PREDICTOR
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.subheader('🤖 Non-Functional School Risk Predictor')
    st.caption(
        'Predicts school failure risk using infrastructure features only. '
        'S007P (student attendance) has been removed to avoid data leakage.'
    )

    if bundle is None:
        st.error('Model not found. Place best_model_bundle.pkl in the models/ folder.')
    else:
        model      = bundle['model']
        feat_names = bundle['feature_names']
        le_prov    = bundle['province_encoder']
        prov_list  = bundle['province_classes']

        # ── Feature descriptions shown to user ──
        feature_descriptions = {
            'is_girls_school' : 'Girls-only school',
            'is_boys_school'  : 'Boys-only school',
            'is_mixed_school' : 'Mixed school',
            'is_primary'      : 'Primary level',
            'is_middle'       : 'Middle/Higher level',
            'province_encoded': 'Province',
            'S03a1'           : 'Teacher present',
            'S03d1'           : 'Drinking water available',
            'S03e1'           : 'Toilet facility available',
            'S03f1'           : 'Electricity available',
        }

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('**School characteristics**')

            prov_input   = st.selectbox('Province', prov_list)
            gender_input = st.selectbox('School gender type',
                                        ['Boys Only', 'Girls Only', 'Mixed'])
            level_input  = st.selectbox('School level',
                                        ['Primary', 'Middle/Higher'])

        with col2:
            st.markdown('**Infrastructure**')

            s03a1_input = st.selectbox(
                'Teacher present (S03a1)',
                options=[1, 0],
                format_func=lambda x: 'Yes — teacher was present' if x == 1 else 'No — no teacher present'
            )
            s03d1_input = st.selectbox(
                'Drinking water (S03d1)',
                options=[1, 0],
                format_func=lambda x: 'Yes — water available' if x == 1 else 'No — no water'
            )
            s03e1_input = st.selectbox(
                'Toilet facility (S03e1)',
                options=[1, 0],
                format_func=lambda x: 'Yes — toilet available' if x == 1 else 'No — no toilet'
            )
            s03f1_input = st.selectbox(
                'Electricity (S03f1)',
                options=[1, 0],
                format_func=lambda x: 'Yes — electricity available' if x == 1 else 'No — no electricity'
            )

        st.divider()

        if st.button('🔮 Predict Risk', type='primary', use_container_width=True):

            # Province encoding — match stripped names
            try:
                classes_stripped = [c.strip() for c in le_prov.classes_]
                prov_idx = classes_stripped.index(prov_input.strip())
                province_enc = prov_idx
            except ValueError:
                province_enc = 0
                st.warning(f'Province not found in encoder — using index 0.')

            input_dict = {
                'is_girls_school' : int(gender_input == 'Girls Only'),
                'is_boys_school'  : int(gender_input == 'Boys Only'),
                'is_mixed_school' : int(gender_input == 'Mixed'),
                'is_primary'      : int(level_input == 'Primary'),
                'is_middle'       : int(level_input == 'Middle/Higher'),
                'province_encoded': province_enc,
                'S03a1'           : float(s03a1_input),
                'S03d1'           : float(s03d1_input),
                'S03e1'           : float(s03e1_input),
                'S03f1'           : float(s03f1_input),
            }

            # Build DataFrame with only the columns the model was trained on
            input_df = pd.DataFrame([{k: input_dict[k] for k in feat_names if k in input_dict}])
            # Fill any missing feature columns with 0
            for col in feat_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feat_names]

            prob     = model.predict_proba(input_df)[0][1]
            pred     = model.predict(input_df)[0]
            risk_pct = round(prob * 100, 1)

            col_g, col_r = st.columns(2)

            with col_g:
                color = 'red' if prob > 0.6 else 'orange' if prob > 0.4 else 'green'
                fig = go.Figure(go.Indicator(
                    mode='gauge+number+delta',
                    value=risk_pct,
                    delta={'reference': 50, 'valueformat': '.1f'},
                    title={'text': 'Non-functional risk score (%)'},
                    number={'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100], 'ticksuffix': '%'},
                        'bar' : {'color': color},
                        'steps': [
                            {'range': [0,  40], 'color': '#EAF3DE'},
                            {'range': [40, 60], 'color': '#FAEEDA'},
                            {'range': [60, 100],'color': '#FCEBEB'},
                        ],
                        'threshold': {
                            'line'     : {'color': 'black', 'width': 2},
                            'thickness': 0.75,
                            'value'    : 50
                        }
                    }
                ))
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                st.markdown('**Prediction result**')
                if pred == 1:
                    st.error(f'⚠️ HIGH RISK — {risk_pct}% probability of being non-functional')
                else:
                    st.success(f'✅ LOW RISK — {risk_pct}% probability of being non-functional')

                st.divider()
                st.markdown('**Input summary**')
                st.markdown(f'- Province: **{prov_input}**')
                st.markdown(f'- School type: **{gender_input}**')
                st.markdown(f'- Level: **{level_input}**')
                st.markdown(f'- Teacher present: **{"Yes" if s03a1_input else "No"}**')
                st.markdown(f'- Drinking water: **{"Yes" if s03d1_input else "No"}**')
                st.markdown(f'- Toilet: **{"Yes" if s03e1_input else "No"}**')
                st.markdown(f'- Electricity: **{"Yes" if s03f1_input else "No"}**')

                st.divider()
                st.markdown('**Top predictors (SHAP)**')
                st.markdown('1. 🔴 Province — regional patterns dominate')
                st.markdown('2. 🟠 Teacher present (S03a1) — strongest infra signal')
                st.markdown('3. 🟡 Drinking water availability')
                st.markdown('4. 🟢 School level (primary vs middle)')
                st.markdown('5. 🔵 Toilet + electricity')
                st.markdown('---')
                st.caption('ℹ️ S007P (students present) was removed — it is an attendance proxy that causes data leakage.')

        # SHAP summary plot — loaded from saved image
        st.divider()
        st.subheader('SHAP Feature Importance (from trained model)')

        col_shap1, col_shap2 = st.columns(2)
        with col_shap1:
            if os.path.exists('visuals/shap_bar.png'):
                st.image('visuals/shap_bar.png', caption='Mean absolute SHAP values', use_column_width=True)
            else:
                st.warning(
                    "**SHAP bar chart not generated yet.**\n\n"
                    "Run notebook `03_ML_Model.ipynb` (all cells) to generate "
                    "`visuals/shap_bar.png`."
                )
        with col_shap2:
            if os.path.exists('visuals/shap_beeswarm.png'):
                st.image('visuals/shap_beeswarm.png', caption='SHAP beeswarm (direction of effect)', use_column_width=True)
            else:
                st.warning(
                    "**SHAP beeswarm chart not generated yet.**\n\n"
                    "Run notebook `03_ML_Model.ipynb` (all cells) to generate "
                    "`visuals/shap_beeswarm.png`."
                )

        # Confusion matrix
        st.divider()
        st.subheader('Model Performance')

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric('Model',    'Random Forest')
        col_m2.metric('Accuracy', '71.0%')
        col_m3.metric('F1 Score', '0.727')
        col_m4.metric('ROC-AUC',  '0.771')

        st.caption(
            'Note: These metrics are for the infrastructure-only model (S007P removed). '
            'AUC is lower than the original 0.877 because the attendance leakage has been fixed — '
            'this model makes genuinely predictive assessments.'
        )

        if os.path.exists('visuals/confusion_matrix.png'):
            st.image('visuals/confusion_matrix.png',
                     caption='Confusion matrix — Random Forest',
                     width=400)
        else:
            st.info('📊 Run `03_ML_Model.ipynb` to generate the confusion matrix image.')

        if os.path.exists('visuals/model_comparison.png'):
            st.image('visuals/model_comparison.png',
                     caption='Model comparison — Logistic Regression vs Random Forest vs XGBoost',
                     use_column_width=True)
        else:
            st.info('📊 Run `03_ML_Model.ipynb` to generate the model comparison chart.')

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    'Pakistan Rural Education Dashboard · ASER 2023 · '
    'Vikram Kumar · BS Data Science · Sindh University · 2025'
)
