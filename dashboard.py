import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go
import folium
import base64
from datetime import date
from streamlit_folium import st_folium

site_param_map = {
    "DRDN": {
        "Filtered_G_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_J_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_E_PRNs": ["S1X", "S5X"],
        "Filtered_I_PRNs": ["S1C", "S2W"],
        "Filtered_R_PRNs": ["S1C", "S2C"],
        "Filtered_C_PRNs": [""]
    },
        "HYDN": {
        "Filtered_G_PRNs": ["S1C", "S2X", "S5Q"],
        "Filtered_J_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_E_PRNs": ["S1C", "L2L"],
        "Filtered_I_PRNs": ["S1C"],
        "Filtered_R_PRNs": ["S1C", "L5Q"],
        "Filtered_C_PRNs": ["S2I", "S7I"]
    },
    "IISC": {
        "Filtered_G_PRNs": ["S1C", "S2X", "S5Q"],
        "Filtered_J_PRNs": ["S1C", "S2X", "S5Q"],
        "Filtered_E_PRNs": ["S1C", "C2L"],
        "Filtered_I_PRNs": ["S1C"],
        "Filtered_R_PRNs": ["S1C", "C5Q"],
        "Filtered_C_PRNs": ["C2L", "C2L","C5Q"]
    },
    "IITK": {
"Filtered_G_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_J_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_E_PRNs": ["S1C", "S1X"],
        "Filtered_I_PRNs": ["S1C"],
        "Filtered_R_PRNs": ["S1C", "S2W"],
        "Filtered_C_PRNs": ["S2I", "S7I"]
    },
    "JDPR": {
        "Filtered_G_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_J_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_E_PRNs": ["S1C", "S1X"],
        "Filtered_I_PRNs": ["S1C", "S1X"],
        "Filtered_R_PRNs": ["S1C", "S2W"],
        "Filtered_C_PRNs": ["S2I", "S7I"]
    },
    "LCK4": {
       "Filtered_G_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_J_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_E_PRNs": ["S1C", "S1X"],
       "Filtered_I_PRNs": ["S1C", "S1X"],
        "Filtered_R_PRNs": ["S1C", "S2W"],
        "Filtered_C_PRNs": ["S1X", "S2W"],
    },
    "PBR4": {
        "Filtered_G_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_J_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_E_PRNs": ["S1C", "S1X"],
       "Filtered_I_PRNs": ["S1C", "S1X"],
        "Filtered_R_PRNs": ["S1C", "S2W"],
        "Filtered_C_PRNs": ["..."]
    },
    "SHLG": {
        "Filtered_G_PRNs":["S1C", "S2X", "S5X"],
        "Filtered_J_PRNs": ["S1C", "S2X", "S5X"],
        "Filtered_E_PRNs": ["S1C", "S1X"],
        "Filtered_I_PRNs": ["S1C"],
       "Filtered_R_PRNs": ["S1C", "S2W"],
        "Filtered_C_PRNs": ["S2I", "S7I"]
    }
}
custom_legend_map = {
    ("DRDN", "IRNSS"): {"S1C": "S5A", "S2W": "S9A"},
    ("HYDN", "GALILEO"): {"S1C": "S1C", "L2L": "S5Q"},
    ("IISC", "GALILEO"): {"S1C": "S1C", "C2L": "S5Q"},
    ("IISC", "IRNSS"): {"S1C": "S5A"},
    ("IISC", "GLONASS"): {"S1C": "S1C", "C5Q": "S2C"},
    ("IITK", "GALILEO"): {"S1C": "S1X", "S1X": "S5X"},
    ("IITK", "IRNSS"): {"S1C": "S5A"},
    ("IITK", "GLONASS"): {"S1C": "S1C", "S2W": "S2C"},
    ("JDPR", "GALILEO"): {"S1C": "S1X", "S1X": "S5X"},
    ("JDPR", "IRNSS"): {"S1C": "S5A", "S1X": "S9A"},
    ("JDPR", "GLONASS"): {"S1C": "S1C", "S2W": "S2C"},
    ("LCK4", "BEIDOU"): {"S1X": "S2I", "S1X": "S7I"},
    ("LCK4", "GALILEO"): {"S1C": "S1X", "S1X": "S5X"},
    ("LCK4", "IRNSS"): {"S1C": "S5A", "S1X": "S9A"},
    ("LCK4", "GLONASS"): {"S1C": "S1C", "S2W": "S2C"},
    ("PBR4", "GALILEO"): {"S1C": "S1X", "S1X": "S5X"},
    ("PBR4", "IRNSS"): {"S1C": "S5A", "S1X": "S9A"},
    ("PBR4", "GLONASS"): {"S1C": "S1X", "S2W": "S2C"},
    ("SHLG", "GALILEO"): {"S1C": "S1X", "S1X": "S5X"},
    ("SHLG", "IRNSS"): {"S1C": "S5A"},
    ("SHLG", "GLONASS"): {"S1C": "S1X", "S2W": "S2C"}
}

# --- Initialize session state ---
if "show_content" not in st.session_state:
    st.session_state.show_content = "none"

if "graph_requested" not in st.session_state:
    st.session_state.graph_requested = False

# --- Page Styling ---
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: #DFF0E0 !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 100% !important;
        overflow-x: hidden !important;
    }
    #root, .viewerBadge_link__qRIco, .viewerBadge_container__1QSob {
        background-color: #DFF0E0 !important;
    }
    section.main {
        background-color: #DFF0E0 !important;
        min-height: 100vh !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .block-container {
        background-color: #DFF0E0 !important;
        padding: 2rem 1rem 1rem 1rem !important;
        margin: 0 auto !important;
        max-width: 100% !important;
    }
    iframe, .folium-map {
        background: transparent !important;
        display: block;
        margin: 0 auto;
        width: 100% !important;
        height: 500px !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #87cefa !important;
    }
    footer {
        display: none;
    }
    header, .css-18ni7ap, .css-1dp5vir {
        background-color: #DFF0E0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .main, .block-container {
        background-color: #DFF0E0 !important;
        padding-top: 2rem !important;
        margin-top: 0rem !important;
    }
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #87cefa !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Site Data ---
site_data = pd.DataFrame({
    'site': ['DRDN','HYDN','IISC','IITK','JDPR','LCK4','PBR','SHLG'],
    'lat': [30.34,17.417,13.021,26.521,26.207,26.912,11.637,25.674],
    'lon': [78.041,78.551,77.57,80.232,73.024,80.956,92.712,92.712]
})

# --- Header: Logo + Title ---
image_path = "test.png"  # Your logo image file path
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

st.markdown(f"""
    <div style="display: flex; align-items: center; margin-top: 20px;">
        <img src="data:image/png;base64,{encoded_image}" alt="Logo" style="width: 120px; margin-right: 60px;">
        <h1 style="color: red; font-family: 'Times New Roman', serif; font-size: 38px; margin: 0;">
            QUALITY CHECK FOR IGS STATIONS IN INDIA
        </h1>
    </div>
""", unsafe_allow_html=True)

# --- Scrolling marquee ---
st.markdown(
    """
    <div style="overflow: hidden; white-space: nowrap; width: 100%;">
        <div style="
            display: inline-block;
            padding-left: 100%;
            animation: scroll-left 25s linear infinite;
            font-weight: bold;
            color: red;
        ">
            *IGS Stations:-DRDN:Deharadun, HYDN:Hyderabad, IISc:Indian institute science Bangalore, IITK:Indian institute of technology Kanpur, LCK4:Lucknow, JDPR:Jodhpur, PBR:Port Blair, SHLG:Shilong* 
        </div>
    </div>
    <style>
    @keyframes scroll-left {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-100%); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="italianno-text">This interactive Dashboard is designed to visualize quality metrics for IGS stations located in India. The graphed parameters include Multipath, Cycle slip, SNR, and Data integrity observations for the period from 01-01-2025 to 31-01-2025.This dashboard is developed purely using Python.</p>',
    unsafe_allow_html=True
)
# --- Main button to show map ---
if st.button("Show Site Map"):
    st.session_state.show_content = "map"

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")

    parameter = st.selectbox("Parameter", ['Select','SNR','CYCLE SLIP RATIO','MULTIPATH','DATA INTEGRITY'])
    selected_site = st.selectbox("IGS Station", ['Choose an option'] + site_data['site'].tolist())

    # ❗️Show Constellation unless DATA INTEGRITY is selected
    if parameter != "DATA INTEGRITY":
        constellation = st.multiselect(
            "Constellation", 
            ['GPS', 'GLONASS', 'GALILEO', 'BEIDOU', 'QZSS', 'IRNSS']
        )
    else:
        constellation = []

    date_input = st.date_input(
        "Date Range", 
        value=(date(2025,1,1), date(2025,1,31)),
        min_value=date(2025,1,1), max_value=date(2025,1,31),
        help="Select start and end date"
    )

    if isinstance(date_input, tuple) and len(date_input) == 2:
        start_date, end_date = date_input
    else:
        start_date = end_date = date_input

    # Date range validation
    if start_date >= end_date:
        st.warning("Please select a valid date range (start date should be before end date).")
        can_generate = False
    else:
        can_generate = True

    if st.button("Generate Graph") and can_generate:
        if selected_site == "Choose an option":
            st.warning("Please select a site.")
        elif parameter == "Select":
            st.warning("Please select a parameter.")
        elif parameter != "DATA INTEGRITY" and not constellation:
            st.warning("Please select at least one constellation.")
        else:
            st.session_state.graph_requested = True
            st.session_state.show_content = "graph"

# --- Display content ---

if st.session_state.show_content == "map":
    st.subheader("Site Map View")
    geojson_path = "india_states.geojson"
    if os.path.exists(geojson_path):
        with open(geojson_path, "r") as f:
            india_geo = json.load(f)
        m = folium.Map(location=[20, 80], zoom_start=4)
        folium.GeoJson(india_geo).add_to(m)
        for _, row in site_data.iterrows():
            folium.Marker(
                location=[row.lat, row.lon],
                tooltip=row.site,
                icon=folium.Icon(color='red', icon='map-marker')
            ).add_to(m)
        st_folium(m, width=1300, height=500)
    else:
        st.error("GeoJSON file for India states not found.")

if st.session_state.show_content == "graph" and st.session_state.graph_requested:
    st.session_state.graph_requested = False  # reset flag

    st.subheader(f"{selected_site} — {parameter}")

    # Data Integrity
    if parameter == "DATA INTEGRITY":
        filepath = f"{selected_site}_integrity.xlsx"
        if os.path.exists(filepath):
            df = pd.read_excel(filepath, parse_dates=['DATE'])
            df = df[(df.DATE >= pd.to_datetime(start_date)) & (df.DATE <= pd.to_datetime(end_date))]
            if not df.empty:
                fig = go.Figure([go.Scatter(x=df.DATE, y=df.Percentage, mode='lines+markers')])
                fig.update_layout(xaxis_title="Date", yaxis_title="Integrity (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available in selected date range.")
        else:
            st.warning(f"Integrity file not found for {selected_site}.")

    # Multipath
    elif parameter == "MULTIPATH":
        constellation_files = {
            "GPS":"G_SYS_parameters",
            "GALILEO":"E_SYS_parameters",
            "GLONASS":"R_SYS_parameters",
            "QZSS":"J_SYS_parameters",
            "IRNSS":"I_SYS_parameters",
            "BEIDOU":"C_SYS_parameters"
        }
        for const in constellation:
            fn = constellation_files.get(const)
            path = os.path.join(selected_site, f"{fn}.xlsx")
            if os.path.exists(path):
                df = pd.read_excel(path, parse_dates=['DATE'])
                df = df[(df.DATE >= pd.to_datetime(start_date)) & (df.DATE <= pd.to_datetime(end_date))]
                if not df.empty:
                    fig = go.Figure()
                    for col in ['STD(MP1)', 'STD(MP2)', 'STD(MP5)']:
                        if col in df.columns:
                            fig.add_trace(go.Scatter(x=df.DATE, y=df[col], mode='lines+markers', name=col))
                    fig.update_layout(title=f"{const} Multipath", xaxis_title="Date", yaxis_title="STD")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data for {const} in selected date range.")
            else:
                st.warning(f"No data file for {const} at {selected_site}.")

   

# SNR block
    elif parameter == "SNR":
        st.subheader(f"SNR Daily Mean for {selected_site}")
        if not constellation:
            st.warning("Please select at least one constellation for SNR.")
        else:
            base_path = os.path.join(f"{selected_site}_SNR", selected_site)

            constellation_folder_map = {
                "GPS": "Filtered_G_PRNs",
                "GALILEO": "Filtered_E_PRNs",
                "GLONASS": "Filtered_R_PRNs",
                "QZSS": "Filtered_J_PRNs",
                "IRNSS": "Filtered_I_PRNs",
                "BEIDOU": "Filtered_C_PRNs"
            }

            for const in constellation:
                subfolder = constellation_folder_map.get(const)
                if not subfolder:
                    st.warning(f"Unsupported constellation: {const}")
                    continue

                folder_path = os.path.join(base_path, subfolder)
                if not os.path.exists(folder_path):
                    st.warning(f"NO DATA FOUND FOR CONSTELLATION")
                    continue

                param_list = site_param_map.get(selected_site, {}).get(subfolder, [])
                if not param_list or param_list == [""]:
                    st.warning(f"No SNR columns mapped for constellation '{const}'.")
                    continue

                daily_means = []

                for file in sorted(os.listdir(folder_path)):
                    if not file.endswith(".csv"):
                        continue

                    try:
                        df = pd.read_csv(os.path.join(folder_path, file))
                        df.columns = df.columns.str.strip()

                        if "time" not in df.columns:
                            continue

                        df["time"] = pd.to_datetime(df["time"], errors='coerce')
                        df = df.dropna(subset=["time"])

                        row = {"time": df["time"].dt.date.iloc[0]}
                        for col in param_list:
                            if col in df.columns:
                                row[col] = pd.to_numeric(df[col], errors="coerce").mean()

                        daily_means.append(row)

                    except Exception as e:
                        st.warning(f"Failed to process file {file}: {e}")

                if not daily_means:
                    st.warning(f"No valid data for constellation {const}.")
                    continue

                df_plot = pd.DataFrame(daily_means)
                df_plot["time"] = pd.to_datetime(df_plot["time"])

                df_plot = df_plot[
                    (df_plot["time"] >= pd.to_datetime(start_date)) &
                    (df_plot["time"] <= pd.to_datetime(end_date))
                ]

                if df_plot.empty:
                    st.warning(f"No SNR data in selected range for {const}.")
                    continue

                fig = go.Figure()
                custom_legends = custom_legend_map.get((selected_site, const), {})
                for col in param_list:
                    if col in df_plot.columns:
                        label = custom_legends.get(col, col)
                        fig.add_trace(go.Scatter(
                            x=df_plot["time"],
                            y=df_plot[col],
                            mode='lines+markers',
                            name=label,
                            hovertemplate=f"Date: %{{x|%b %d}}<br>{label}: %{{y:.2f}} dBHz<extra></extra>"
                        ))

                fig.update_layout(
                    title=f"{selected_site} - {const} Signal-to-Noise Ratio",
                    xaxis_title="Date",
                    yaxis_title="SNR (dBHz)",
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

    elif parameter == "CYCLE SLIP RATIO":
        constellation_file_map = {
            "GPS": "G_SYS_parameters",
            "GALILEO": "E_SYS_parameters",
            "GLONASS": "R_SYS_parameters",
            "QZSS": "J_SYS_parameters",
            "IRNSS": "I_SYS_parameters",
            "BEIDOU": "C_SYS_parameters"
        }

        site_folder_path = os.path.join(f"{selected_site}")
        if not os.path.exists(site_folder_path):
            st.error(f"Site folder does not exist: {site_folder_path}")
        else:
            for const in constellation:
                file_name = constellation_file_map.get(const)
                if not file_name:
                    st.warning(f"No file mapping for constellation: {const}")
                    continue

                file_path = os.path.join(site_folder_path, f"{file_name}.xlsx")
                if not os.path.exists(file_path):
                    st.warning(f"File not found: {file_path}")
                    continue

                try:
                    df = pd.read_excel(file_path)
                    if 'DATE' not in df.columns:
                        st.error(f"'DATE' column not found in {file_name}")
                        continue

                    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
                    df = df.dropna(subset=['DATE'])

                    fraction_columns = [
                        '# of slips/nobs (MP1)',
                        '# of slips/nobs (MP2)',
                        '# of slips/nobs (MP5)',
                        '# of slips/nobs (GF)',
                        '# of slips/nobs (MW)',
                        '# of slips/nobs (IOD(L1))'
                    ]

                    legend_name_map = {
                        '# of slips/nobs (MP1)': 'CSR_MP1',
                        '# of slips/nobs (MP2)': 'CSR_MP2',
                        '# of slips/nobs (MP5)': 'CSR_MP5',
                        '# of slips/nobs (GF)': 'CSR_GF',
                        '# of slips/nobs (MW)': 'CSR_MW',
                        '# of slips/nobs (IOD(L1))': 'CSR_IOD'
                    }

                    import numpy as np

                    def transform_fraction_vectorized(series):
                        split_vals = series.astype(str).str.split('/', expand=True)
                        nums = pd.to_numeric(split_vals[0], errors='coerce')
                        denoms = pd.to_numeric(split_vals[1], errors='coerce')

                        swapped = denoms / nums
                        swapped.replace([np.inf, -np.inf, 0], np.nan, inplace=True)

                        result = 1000 / swapped
                        return result

                    for col in fraction_columns:
                        if col in df.columns:
                            df[legend_name_map[col]] = transform_fraction_vectorized(df[col])

                    df_filtered = df[
                        (df['DATE'] >= pd.to_datetime(start_date)) &
                        (df['DATE'] <= pd.to_datetime(end_date))
                    ]

                    if df_filtered.empty:
                        st.warning(f"No cycle slip data in selected date range for {const}.")
                        continue

                    fig = go.Figure()
                    for col in legend_name_map.values():
                        if col in df_filtered.columns:
                            fig.add_trace(go.Scatter(
                                x=df_filtered['DATE'],
                                y=df_filtered[col],
                                mode='lines+markers',
                                name=col
                            ))

                    fig.update_layout(
                        title=f"{selected_site} - {const} Cycle Slip Ratio",
                        xaxis_title="Date",
                        yaxis_title="Cycle Slip Ratio",
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing file {file_path}: {e}")

