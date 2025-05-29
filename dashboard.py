import streamlit as st
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
import os
import csv
import numpy as np
import os




# === Static Setup ===
#DATA_FOLDER = 'StreamlitDashboard'  # Change this if needed


# === Site and location data ===
site_data = pd.DataFrame({
    'site': ['DRDN','HYDN','IISC','IITK', 'JDPR', 'LCK4','PBR','SHLG'],
    'lat': [30.340, 17.417, 13.021,26.521,26.207,26.912,11.637,25.674],
    'lon': [78.041, 78.551, 77.570,80.232,73.024,80.956,92.712,92.712]
})

# === Sidebar filters ===
st.sidebar.title("Filters")
parameter = st.sidebar.selectbox("Parameter", ['SNR', 'CYCLE SLIP RATIO', 'MULTIPATH','DATA INTEGRITY'])
selected_site = st.sidebar.selectbox("IGS Station", site_data['site'])
constellation = st.sidebar.multiselect("Constellation", ['GPS', 'GLONASS', 'GALILEO', 'BEIDOU','QZSS','IRNSS'])

# ✅ Safe date input: single or range
#date_input = st.sidebar.date_input(
    #min_value=date(2025, 1, 1),
    #max_value=date(2025, 1, 31)
#)
# ✅ Safe date input: single or range
date_input = st.sidebar.date_input(
    "Select Date Range",
    value=(date(2025, 1, 1), date(2025, 1, 31)),
    min_value=date(2025, 1, 1),
    max_value=date(2025, 1, 31)
)

if isinstance(date_input, tuple) and len(date_input) == 2:
    start_date, end_date = date_input
    is_range = start_date != end_date
else:
    # Single date selected
    start_date = end_date = date_input if isinstance(date_input, date) else date_input[0]
    is_range = False

# ✅ Global check to block rest of the app if only one date is selected
if not is_range:
    st.warning("PLEASE SELECT THE DATE RANGE")
    st.stop()



if isinstance(date_input, tuple):
    start_date, end_date = date_input
else:
    start_date = end_date = date_input  # Single date selected

# === Page Title ===
st.image("logo.png", width=150)  # Adjust width as needed
st.markdown('<h1 style="color:red;">QUALITY CHECK FOR IGS STATIONs IN INDIA</h1>', unsafe_allow_html=True)


# === Site map ===
st.subheader("Site Map")
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=20.0,
        longitude=80.0,
        zoom=3,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=site_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=50000,
        ),
    ],
))

# === Data Integrity Plot ===
#st.title("DATA INTEGRITY PLOT")

if parameter == "DATA INTEGRITY":
    
    file_name = os.path.join(f"{selected_site}_integrity.xlsx")
    #st.write(f"Looking for file: `{file_name}`")

    if os.path.exists(file_name):
        try:
            df = pd.read_excel(file_name, parse_dates=['DATE'])
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

            mask = (df['DATE'] >= pd.to_datetime(start_date)) & (df['DATE'] <= pd.to_datetime(end_date))
            filtered = df[mask]

            if not filtered.empty:
                fig, ax = plt.subplots()
                ax.plot(filtered['DATE'], filtered['Percentage'], marker='o', linestyle='-')
                ax.set_title(f'{selected_site} - Data Integrity')
                ax.set_xlabel('Date')
                ax.set_ylabel('Percentage (%)')
                ax.grid(True)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                fig.autofmt_xdate()
                st.pyplot(fig)
            else:
                st.warning("No data available for selected date(s).")

        except Exception as e:
            st.error(f"Failed to read or plot data: {e}")
    else:
        st.error("File not found.")


####  MULIPATH

# Mapping constellation to exact file name prefix
constellation_files = {
    "GPS": "G_SYS_parameters",
    "GALILEO": "E_SYS_parameters",
    "QZSS": "J_SYS_parameters",
    "IRNSS": "I_SYS_parameters",
    "GLONASS": "R_SYS_parameters",
    "BEIDOU": "C_SYS_parameters"
}

# Plotting Multipath
if parameter == "MULTIPATH":
    st.subheader(f"Multipath Parameters for {selected_site}")

    # Loop over selected constellations
    for const in constellation:
        # Get the file name based on constellation, skip if unknown
        filename = constellation_files.get(const)
        if filename is None:
            st.warning(f"Unknown constellation: {const}")
            continue

        file_path = os.path.join(f"{selected_site}", f"{filename}.xlsx")
        st.write(f"Looking for file: {file_path}")

        if os.path.exists(file_path):
            df = pd.read_excel(file_path, parse_dates=["DATE"])

            # Filter by selected date range
            df_filtered = df[
                (df['DATE'] >= pd.to_datetime(start_date)) &
                (df['DATE'] <= pd.to_datetime(end_date))
            ]

            if not df_filtered.empty:
                fig, ax = plt.subplots()
                ax.plot(df_filtered['DATE'], df_filtered['STD(MP1)'], label='STD(MP1)')
                ax.plot(df_filtered['DATE'], df_filtered['STD(MP2)'], label='STD(MP2)')
                if 'STD(MP5)' in df_filtered.columns:
                    ax.plot(df_filtered['DATE'], df_filtered['STD(MP5)'], label='STD(MP5)')
                ax.set_title(f"{const} - Multipath Parameters")
                ax.set_xlabel("DATE")
                ax.set_ylabel("Standard Deviation")
                ax.legend()
                ax.grid(True)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                fig.autofmt_xdate()
                st.pyplot(fig)
            else:
                st.warning(f"No data available for {const} in selected range.")
        else:
            st.error(f"File not found for {const}: {file_path}")


#### SNR

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# Folder structure mapping for constellations
constellation_folder_map = {
    "GPS": "Filtered_G_PRNs",
    "GALILEO": "Filtered_E_PRNs",
    "GLONASS": "Filtered_R_PRNs",
    "QZSS": "Filtered_J_PRNs",
    "IRNSS": "Filtered_I_PRNs",
    "BEIDOU": "Filtered_C_PRNs"
}

# Parameter mapping based on site and constellation
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

# Add more sites as needed

# Custom legend mapping
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

# Assume you already have:
# selected_site, constellation (list), start_date, end_date, parameter = "SNR"

if parameter == "SNR":
    st.subheader(f"SNR Daily Mean for {selected_site}")

    base_path = os.path.join(f"{selected_site}_SNR", selected_site)

    for const in constellation:
        subfolder = constellation_folder_map.get(const)
        if not subfolder:
            st.warning(f"Unsupported constellation: {const}")
            continue

        folder_path = os.path.join(base_path, subfolder)
        if not os.path.exists(folder_path):
            st.warning(f"NO DATA FOUND FOR CONSTELLATION")
            continue

        # Get parameters for the site and constellation folder
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

        fig, ax = plt.subplots(figsize=(10, 5))

        # Apply custom legends
        custom_legends = custom_legend_map.get((selected_site, const), {})

        for col in param_list:
            if col in df_plot.columns:
                label = custom_legends.get(col, col)
                ax.plot(df_plot["time"], df_plot[col], label=label)

        ax.set_title(f"{selected_site} - {const} Signal-to-Noise Ratio")
        ax.set_xlabel("Date")
        plt.ylim(30, 50)
        plt.yticks(np.arange(30, 51, 1))
        ax.set_ylabel("SNR(dBHz)")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate()
        st.pyplot(fig)


#### CSR

if parameter == "CYCLE SLIP RATIO":
    st.subheader(f"Cycle Slip Ratio for {selected_site}")

    # Use the constellation selected by the user
    if not constellation:
        st.warning("Please select at least one constellation for Cycle Slip Ratio.")
    else:
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

                    # Column names to transform
                    fraction_columns = [
                        '# of slips/nobs (MP1)',
                        '# of slips/nobs (MP2)',
                        '# of slips/nobs (MP5)',
                        '# of slips/nobs (GF)',
                        '# of slips/nobs (MW)',
                        '# of slips/nobs (IOD(L1))'
                    ]

                    # Legend name mapping
                    legend_name_map = {
                        '# of slips/nobs (MP1)': 'CSR_MP1',
                        '# of slips/nobs (MP2)': 'CSR_MP2',
                        '# of slips/nobs (MP5)': 'CSR_MP5',
                        '# of slips/nobs (GF)': 'CSR_GF',
                        '# of slips/nobs (MW)': 'CSR_MW',
                        '# of slips/nobs (IOD(L1))': 'CSR_IOD'
                    }

                    # Function to compute CSR
                    def transform_fraction(fraction_str):
                        try:
                            num, denom = map(float, str(fraction_str).split('/'))
                            swapped = denom / num if num != 0 else None
                            return 1000 / swapped if swapped else None
                        except:
                            return None

                    # Group by date and calculate transformed CSR
                    all_data = []
                    for date_val, group in df.groupby(df['DATE'].dt.date):
                        row = {'DATE': pd.to_datetime(date_val)}
                        for col in fraction_columns:
                            if col in group.columns:
                                transformed_vals = group[col].apply(transform_fraction)
                                row[col] = transformed_vals.mean()
                        all_data.append(row)

                    df_plot = pd.DataFrame(all_data)

                    # Filter by selected date range
                    df_plot = df_plot[
                        (df_plot["DATE"] >= pd.to_datetime(start_date)) &
                        (df_plot["DATE"] <= pd.to_datetime(end_date))
                    ]

                    if df_plot.empty:
                        st.warning(f"No Cycle Slip Ratio data available for {const} in selected date range.")
                        continue

                    # Plotting with legend name mapping
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for col in fraction_columns:
                        if col in df_plot.columns:
                            legend_label = legend_name_map.get(col, col)
                            ax.plot(df_plot['DATE'], df_plot[col], marker='o', label=legend_label)

                    ax.set_title(f"{selected_site} - {const} Cycle Slip Ratio")
                    ax.set_xlabel("Date")

                    if const != "IRNSS":
                        plt.ylim(0, 110)
                        plt.yticks(np.arange(0, 111, 10))

                    ax.set_ylabel("Cycle Slip Ratio (CSR)")
                    ax.grid(True)
                    ax.legend()
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                    fig.autofmt_xdate()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error processing file {file_name}: {e}")
