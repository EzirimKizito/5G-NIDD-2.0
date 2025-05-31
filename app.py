import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler # Keep for type hinting if needed elsewhere
import time
import os
import re

# --- Page Config ---
st.set_page_config(
    layout="wide",
    page_title="5G NIDD Anomaly Detection System",
    page_icon="📡"
)

# --- Configuration & Constants ---
MODEL_PATH = "nidd_model.h5"
SCALER_PATH = "scaler2.joblib"

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f0f2f6; /* Light grey background */
    }

    /* Title style */
    h1 {
        color: #1E3A5F; /* Dark blue */
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
        margin-bottom: 20px; /* Added margin below title */
    }
    h3 { /* Subheaders like "Flow Characteristics Input" */
        color: #2c3e50; /* Slightly softer dark blue */
        border-bottom: 2px solid #1E3A5F;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
     h5 { /* Section subheaders like "Flow Timing & Packet Stats" */
        color: #34495e;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }


    /* Sidebar styling */
    .css-1d391kg { /* Streamlit's sidebar class might change, inspect if needed */
        background-color: #e8eef7; /* Lighter blue for sidebar */
        border-right: 1px solid #d1d9e6; /* Subtle border for sidebar */
    }
    .st-emotion-cache-16txtl3 { /* Sidebar title*/
        color: #1E3A5F;
        font-weight: bold;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 8px;
        border: 1px solid #3e8e41; /* Darker green border */
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 2px; /* Increased top/bottom margin */
        cursor: pointer;
        transition-duration: 0.3s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green */
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stButton>button:active {
        background-color: #3e8e41;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }


    /* Expander header */
    .st-emotion-cache-ff2z0k { /* Streamlit's expander header class */
        font-weight: bold;
        color: #333;
        background-color: #f9f9f9; /* Light background for expander header */
        border-radius: 5px;
        padding: 8px 12px !important; /* Ensure padding is applied */
    }
    /* Metric labels */
    .st-emotion-cache-1r6slb0{
        font-weight: bold;
        color: #2c3e50; /* Darker text for metric labels */
    }
    /* Metric values */
    .st-emotion-cache-1g6goys{
         color: #1E3A5F; /* Dark blue for metric values */
         font-size: 1.6em !important; /* Slightly larger metric value */
    }

    /* Input field styling - This is a general approach.
       Streamlit's internal structure for input widgets can be complex.
       We target the div that likely wraps the input.
    */
    div[data-testid="stNumberInput"],
    div[data-testid="stSelectbox"] {
        border: 1px solid #ced4da; /* Slim light grey border */
        border-radius: 6px;       /* Rounded corners */
        padding: 8px 10px 2px 10px;    /* Inner padding (top, right, bottom, left) */
        margin-bottom: 10px;      /* Space below each input field container */
        background-color: #ffffff; /* White background for the input area */
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.075); /* Inner shadow for depth */
    }

    /* Adjust label margin for inputs to prevent overlap with new border */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        margin-bottom: 4px !important;
    }

    /* Adjust padding for the actual input element if possible (might need more specific selectors) */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] select { /* This select might not work directly due to Streamlit's rendering */
        padding-top: 5px !important;
        padding-bottom: 5px !important;
    }
    /* Text area specific styling (if needed) */
    div[data-testid="stTextArea"] textarea {
        border: 1px solid #ced4da;
        border-radius: 6px;
        padding: 8px;
    }


</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
st.sidebar.title("👨‍💻 Project & System Info")
st.sidebar.markdown("---")

with st.sidebar.expander("📖 About This System", expanded=False):
    st.markdown(
        """
        **🎯 System Objective:**
        This application leverages a sophisticated machine learning model to
        identify and flag anomalous network traffic patterns within 5G
        Non-IP Data Delivery (NIDD) flows. The goal is to enhance network
        security by detecting potential threats or misconfigurations in real-time.
        """
    )
    st.divider()
    st.subheader("🚀 How to Use:")
    st.markdown(
        """
        1.  **Provide Flow Data:** Accurately input the parameters for the network
            flow you wish to analyze using the designated fields on the main panel.
            A sample benign instance is pre-filled for your convenience.
        2.  **(Optional) Quick Data Entry:** For multiple analyses or pre-saved data,
            you can paste data formatted as `Feature Name: Value` (one entry per line)
            into the 'Paste Instance Data' section. Then, click 'Fill Inputs from
            Pasted Text'.
            *Note: Ensure `Proto`, `Cause`, and `State` values are the **encoded numerical equivalents** if using this method.*
        3.  **Initiate Analysis:** Once all relevant data is entered, click the
            **'Analyze Flow Status'** button below the input fields.
        4.  **Review Analysis:** The system's prediction (Benign/Malicious),
            an associated risk level, and a confidence score will be displayed.
        """
    )
    st.divider()
    st.subheader("🎓 Project Details:")
    st.markdown(
        """
        *   **Researcher:** OLORUKOOBA IBRAHIM KAYODE
        *   **Matric No:** 20/52HA088
        *   **Academic Context:** Final Year Project Submission
        *   **Department:** Department of Computer Science
        *   **Institution:** University of Ilorin
        """
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Powered by TensorFlow & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.divider()
debug_mode = st.sidebar.checkbox("🛠️ Enable Developer Mode", value=False, help="Show detailed processing logs and internal states.")
if debug_mode:
    st.sidebar.warning("Developer Mode is ACTIVE. Verbose logging enabled.")
st.sidebar.markdown("---")
st.sidebar.info(f"Model: `{os.path.basename(MODEL_PATH)}`")
st.sidebar.info(f"Scaler: `{os.path.basename(SCALER_PATH)}`")


# --- Load Model and Preprocessing Objects (Cached) ---
@st.cache_resource
def load_prediction_model(path):
    if debug_mode: st.sidebar.write(f"Attempting to load model from: {path}")
    if not os.path.exists(path):
        st.error(f"Critical Error: Model file not found at `{path}`. Please ensure the file exists in the correct location.")
        return None
    try:
        model = load_model(path)
        print(f"Model loaded successfully from {path}.")
        st.sidebar.success("✅ Neural Network Model Initialized")
        if hasattr(model, 'input_shape'):
            st.sidebar.caption(f"Model Expected Input Shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"Error loading model from `{path}`: {e}")
        st.sidebar.error("❌ Model Loading Failed")
        return None

@st.cache_resource
def load_scaler_object(path):
    if debug_mode: st.sidebar.write(f"Attempting to load scaler from: {path}")
    if not os.path.exists(path):
        st.error(f"Critical Error: Scaler file not found at `{path}`. Please ensure the file exists.")
        return None
    try:
        scaler = joblib.load(path)
        print(f"Scaler loaded successfully from {path}.")
        st.sidebar.success("✅ Data Scaler Initialized")
        if hasattr(scaler, 'n_features_in_'):
            st.sidebar.caption(f"Scaler Trained on Features: {scaler.n_features_in_}")
        if debug_mode:
            with st.sidebar.expander("Loaded Scaler Internals (Dev Mode)"):
                np.set_printoptions(precision=6, suppress=True)
                if hasattr(scaler, 'mean_'): st.write("**Mean:**", scaler.mean_)
                if hasattr(scaler, 'scale_'): st.write("**Scale (Std Dev):**", scaler.scale_)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler from `{path}`: {e}")
        st.sidebar.error("❌ Scaler Loading Failed")
        return None

model = load_prediction_model(MODEL_PATH)
scaler = load_scaler_object(SCALER_PATH)

FINAL_FEATURE_COLS = [
    'Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max', 'Proto', 'sTtl', 'sHops', 'Cause',
    'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes', 'DstBytes', 'Offset',
    'sMeanPktSz', 'dMeanPktSz', 'Load', 'SrcLoad', 'DstLoad', 'Rate', 'SrcRate',
    'DstRate', 'State', 'TcpRtt', 'SynAck', 'AckDat'
]
if model and scaler:
    st.sidebar.caption(f"Model configured for {len(FINAL_FEATURE_COLS)} input features.")
    with st.sidebar.expander("View Expected Model Features"):
        st.dataframe(pd.DataFrame({'Feature Name': FINAL_FEATURE_COLS}), use_container_width=True, height=200)

categorical_features_options = {
    'Proto': ['icmp', 'udp', 'tcp', 'sctp', 'ipv6-icmp'],
    'Cause': ['Start', 'Status', 'Shutdown'],
    'State': ['ECO', 'CON', 'REQ', 'TST', 'RST', 'INT', 'FIN', 'URP', 'RSP', 'NRS', 'ACC'],
    'sTtl': [32, 58, 60, 63, 64, 117, 128, 249, 252, 255]
}
default_benign_instance = {
    'Dur': 0.073378, 'RunTime': 0.073378, 'Mean': 0.073378, 'Sum': 0.073378,
    'Min': 0.073378, 'Max': 0.073378, 'Proto': 'udp', 'sTtl': 64, 'sHops': 0,
    'Cause': 'Status', 'TotPkts': 7, 'SrcPkts': 4, 'DstPkts': 3, 'TotBytes': 960,
    'SrcBytes': 760, 'DstBytes': 200, 'Offset': 47072, 'sMeanPktSz': 190.0,
    'dMeanPktSz': 66.666664, 'Load': 76753.25, 'SrcLoad': 62143.96875,
    'DstLoad': 14609.28418, 'Rate': 81.768379, 'SrcRate': 40.88419,
    'DstRate': 27.256126, 'State': 'CON', 'TcpRtt': 0.0, 'SynAck': 0.0, 'AckDat': 0.0
}

proto_mapping = {'icmp': 0, 'udp': 1, 'tcp': 2, 'sctp': 3, 'ipv6-icmp': 4}
cause_mapping = {'Start': 0, 'Status': 1, 'Shutdown': 2}
state_mapping = {'ECO': 0, 'CON': 1, 'REQ': 2, 'TST': 3, 'RST': 4, 'INT': 5, 'FIN': 6, 'URP': 7, 'RSP': 8, 'NRS': 9, 'ACC': 10}
reverse_proto_mapping = {v: k for k, v in proto_mapping.items()}
reverse_cause_mapping = {v: k for k, v in cause_mapping.items()}
reverse_state_mapping = {v: k for k, v in state_mapping.items()}
feature_mappings_to_apply = {'Proto': proto_mapping, 'Cause': cause_mapping, 'State': state_mapping}
label_mapping = {'Benign': 0, 'Malicious': 1}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def apply_manual_encoding_deploy(df, mappings):
    df_encoded = df.copy()
    cols_processed_for_numeric = []
    for col, mapping in mappings.items():
        if col in df_encoded.columns and isinstance(mapping, dict):
            original_value = df_encoded.iloc[0][col]
            if pd.isna(original_value): df_encoded.loc[0, col] = np.nan
            else:
                mapped_value = mapping.get(original_value)
                if mapped_value is None:
                    if debug_mode: st.sidebar.warning(f"Value '{original_value}' not in mapping for '{col}'. Setting NaN.")
                    df_encoded.loc[0, col] = np.nan
                else: df_encoded.loc[0, col] = mapped_value
            cols_processed_for_numeric.append(col)
    for col in cols_processed_for_numeric: df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    return df_encoded

def preprocess_single_input(input_dict, scaler_obj, feature_cols_list, mappings_to_apply):
    if debug_mode:
        st.sidebar.divider(); st.sidebar.write("--- Dev: Preprocessing Start ---")
        st.sidebar.json({"Input Dictionary (UI)": input_dict})
    input_df = pd.DataFrame([input_dict])
    if debug_mode: st.sidebar.caption("Input DataFrame (1 row) dtypes:"); st.sidebar.code(input_df.dtypes)
    df_processed = apply_manual_encoding_deploy(input_df.copy(), mappings_to_apply)
    if debug_mode:
        st.sidebar.caption("DataFrame AFTER Encoding - dtypes:"); st.sidebar.code(df_processed.dtypes)
        st.sidebar.caption("DataFrame AFTER Encoding - values:"); st.sidebar.code(df_processed.iloc[0].to_dict())
    try:
        missing_cols = [col for col in feature_cols_list if col not in df_processed.columns]
        if missing_cols: return None, f"Critical Error: Columns missing: {missing_cols}."
        df_features = df_processed[feature_cols_list]
    except Exception as e: return None, f"Error during feature selection: {e}"
    if debug_mode:
        st.sidebar.caption(f"DataFrame FEATURES selected ({len(feature_cols_list)}) - dtypes:"); st.sidebar.code(df_features.dtypes)
        st.sidebar.caption("Feature Values before Scaling:"); st.sidebar.code(df_features.iloc[0].to_dict())
    if df_features.isnull().values.any():
        nan_cols = df_features.columns[df_features.isnull().any(axis=0)].tolist()
        return None, f"Data Error: Missing values in: {nan_cols}."
    non_numeric_cols = df_features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols: return None, f"Internal Error: Non-numeric columns: {non_numeric_cols}."
    try:
        if df_features.shape[1] != scaler_obj.n_features_in_:
             return None, f"Shape Mismatch: Scaler needs {scaler_obj.n_features_in_}, data has {df_features.shape[1]}."
        scaled_data = scaler_obj.transform(df_features)
        if np.isnan(scaled_data).any(): return None, "Data Error: NaNs after scaling."
    except Exception as e: return None, f"Scaling Error: {e}"
    if debug_mode: st.sidebar.write("--- Dev: Preprocessing End ---"); st.sidebar.divider()
    return scaled_data, None

if 'initialized' not in st.session_state:
    for key, value in default_benign_instance.items(): st.session_state[f"input_{key}"] = value
    st.session_state.pasted_text = ""; st.session_state.initialized = True

st.title("📡 5G NIDD Anomaly Detection System")
st.markdown("Welcome! Input network flow parameters or paste data, then click 'Analyze Flow Status' for prediction.")
st.markdown("---")

with st.expander("📋 Paste Instance Data (Optional - Advanced Users)", expanded=False):
    st.markdown("""Paste data: `FeatureName: Value` (one per line). **Encoded numerical values** for `Proto`, `Cause`, `State`. `sTtl` must be from predefined list.""")
    pasted_text_area = st.text_area("Paste encoded data here:", value=st.session_state.pasted_text, height=250, key="paste_area_key", label_visibility="collapsed")
    if st.button("📄 Apply Pasted Data to Inputs", key="paste_button_key"):
        st.session_state.pasted_text = pasted_text_area
        parsed_values_from_paste, parsing_errors = {}, []
        if st.session_state.pasted_text:
            for i, line in enumerate(st.session_state.pasted_text.strip().split('\n')):
                if not (line := line.strip()): continue
                if not (match := re.match(r"^\s*([a-zA-Z0-9_]+)\s*:\s*(.*)$", line)):
                    parsing_errors.append(f"L{i+1}: Invalid format '{line}'"); continue
                feature_name, value_str = match.group(1).strip(), match.group(2).strip()
                if feature_name not in FINAL_FEATURE_COLS: continue
                try:
                    if feature_name == 'Proto': parsed_val = reverse_proto_mapping.get(int(float(value_str)))
                    elif feature_name == 'Cause': parsed_val = reverse_cause_mapping.get(int(float(value_str)))
                    elif feature_name == 'State': parsed_val = reverse_state_mapping.get(int(float(value_str)))
                    elif feature_name == 'sTtl': parsed_val = int(float(value_str)); assert parsed_val in categorical_features_options['sTtl']
                    else: parsed_val = float(value_str); int_cols = ['sHops', 'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes', 'DstBytes', 'Offset'];_ = [parsed_val := int(parsed_val) for f in int_cols if feature_name == f]
                    if parsed_val is None: raise ValueError(f"Invalid code/value for {feature_name}")
                    parsed_values_from_paste[feature_name] = parsed_val
                except Exception as e_parse: parsing_errors.append(f"L{i+1} ('{feature_name}'): Invalid '{value_str}' - {e_parse}")
            updated_count = 0
            for key, value in parsed_values_from_paste.items():
                if f"input_{key}" in st.session_state and value is not None: st.session_state[f"input_{key}"] = value; updated_count +=1
            if updated_count > 0: st.success(f"Applied {updated_count} values.")
            if parsing_errors: st.warning("Pasting Issues:\n" + "\n".join(parsing_errors))
            st.rerun()
        else: st.info("Paste area is empty.")
st.markdown("---")

user_inputs = {}
st.subheader("📊 Flow Characteristics Input")

st.markdown("##### **Flow Timing & Packet Stats**")
cols = st.columns(4)
with cols[0]: user_inputs['Dur'] = st.number_input('Duration (s)', 0.0, float(st.session_state.input_Dur), format="%.6f", key="dur_ui", help="Total duration of the flow.")
with cols[1]: user_inputs['RunTime'] = st.number_input('Run Time (s)', 0.0, float(st.session_state.input_RunTime), format="%.6f", key="runtime_ui", help="Effective runtime.")
with cols[2]: user_inputs['Mean'] = st.number_input('Mean IPT (s)', 0.0, float(st.session_state.input_Mean), format="%.6f", key="mean_ui", help="Mean inter-packet time.")
with cols[3]: user_inputs['Sum'] = st.number_input('Sum IPT (s)', 0.0, float(st.session_state.input_Sum), format="%.6f", key="sum_ui", help="Sum of inter-packet times.")
cols = st.columns(4)
with cols[0]: user_inputs['Min'] = st.number_input('Min IPT (s)', 0.0, float(st.session_state.input_Min), format="%.6f", key="min_ui", help="Minimum inter-packet time.")
with cols[1]: user_inputs['Max'] = st.number_input('Max IPT (s)', 0.0, float(st.session_state.input_Max), format="%.6f", key="max_ui", help="Maximum inter-packet time.")
with cols[2]: user_inputs['TotPkts'] = st.number_input('Total Packets', 0, int(st.session_state.input_TotPkts), 1, key="totpkts_ui", help="Total packets.")
with cols[3]: user_inputs['Offset'] = st.number_input('Offset (bytes)', 0, int(st.session_state.input_Offset), 1, key="offset_ui", help="Data offset.")

st.markdown("---")
st.markdown("##### **Packet & Byte Counts**")
cols = st.columns(3)
with cols[0]:
    user_inputs['SrcPkts'] = st.number_input('Source Packets', 0, int(st.session_state.input_SrcPkts), 1, key="srcpkts_ui", help="Packets from source.")
    user_inputs['DstPkts'] = st.number_input('Dest Packets', 0, int(st.session_state.input_DstPkts), 1, key="dstpkts_ui", help="Packets to destination.")
with cols[1]:
    user_inputs['TotBytes'] = st.number_input('Total Bytes', 0, int(st.session_state.input_TotBytes), 1, key="totbytes_ui", help="Total bytes in flow.")
    user_inputs['SrcBytes'] = st.number_input('Source Bytes', 0, int(st.session_state.input_SrcBytes), 1, key="srcbytes_ui", help="Bytes from source.")
with cols[2]:
    user_inputs['DstBytes'] = st.number_input('Dest Bytes', 0, int(st.session_state.input_DstBytes), 1, key="dstbytes_ui", help="Bytes to destination.")
    user_inputs['sMeanPktSz'] = st.number_input('Src Mean Pkt Size', 0.0, float(st.session_state.input_sMeanPktSz), format="%.2f", key="smeansz_ui", help="Avg source packet size.")

st.markdown("---")
st.markdown("##### **Protocol, Routing & State Information**")
cols = st.columns(4)
with cols[0]: user_inputs['Proto'] = st.selectbox('Protocol', categorical_features_options['Proto'], index=categorical_features_options['Proto'].index(st.session_state.input_Proto) if st.session_state.input_Proto in categorical_features_options['Proto'] else 0, key="proto_ui", help="Network protocol.")
with cols[1]: user_inputs['sTtl'] = st.selectbox('Source TTL', categorical_features_options['sTtl'], index=categorical_features_options['sTtl'].index(st.session_state.input_sTtl) if st.session_state.input_sTtl in categorical_features_options['sTtl'] else 0, key="sttl_ui", help="Source Time-To-Live.")
with cols[2]: user_inputs['sHops'] = st.number_input('Source Hops', 0, int(st.session_state.input_sHops), 1, key="shops_ui", help="Hops from source.")
with cols[3]: user_inputs['State'] = st.selectbox('Flow State', categorical_features_options['State'], index=categorical_features_options['State'].index(st.session_state.input_State) if st.session_state.input_State in categorical_features_options['State'] else 0, key="state_ui", help="Current flow state.")

st.markdown("---")
st.markdown("##### **Network Load, Rate & Flow Termination**")
cols = st.columns(3)
with cols[0]:
     user_inputs['Load'] = st.number_input('Overall Load (bps)', 0.0, float(st.session_state.input_Load), format="%.2f", key="load_ui", help="Total network load (bps).")
     user_inputs['SrcLoad'] = st.number_input('Source Load (bps)', 0.0, float(st.session_state.input_SrcLoad), format="%.2f", key="srcload_ui", help="Source load (bps).")
     user_inputs['DstLoad'] = st.number_input('Dest Load (bps)', 0.0, float(st.session_state.input_DstLoad), format="%.2f", key="dstload_ui", help="Destination load (bps).")
with cols[1]:
    user_inputs['Rate'] = st.number_input('Overall Rate (pps)', 0.0, float(st.session_state.input_Rate), format="%.2f", key="rate_ui", help="Total packet rate (pps).")
    user_inputs['SrcRate'] = st.number_input('Source Rate (pps)', 0.0, float(st.session_state.input_SrcRate), format="%.2f", key="srcrate_ui", help="Source packet rate (pps).")
    user_inputs['DstRate'] = st.number_input('Dest Rate (pps)', 0.0, float(st.session_state.input_DstRate), format="%.2f", key="dstrate_ui", help="Destination packet rate (pps).")
with cols[2]:
    user_inputs['dMeanPktSz'] = st.number_input('Dest Mean Pkt Size', 0.0, float(st.session_state.input_dMeanPktSz), format="%.6f", key="dmeansz_ui", help="Avg dest packet size.")
    user_inputs['Cause'] = st.selectbox('Flow Termination Cause', categorical_features_options['Cause'], index=categorical_features_options['Cause'].index(st.session_state.input_Cause) if st.session_state.input_Cause in categorical_features_options['Cause'] else 0, key="cause_ui", help="Reason for flow termination.")

st.markdown("---")
st.markdown("##### **TCP Specific Metrics** (Enter 0 if N/A)")
cols = st.columns(3)
with cols[0]: user_inputs['TcpRtt'] = st.number_input('TCP RTT (s)', 0.0, float(st.session_state.input_TcpRtt), format="%.6f", key="tcprtt_ui", help="TCP Round Trip Time.")
with cols[1]: user_inputs['SynAck'] = st.number_input('TCP SYN-ACK Time (s)', 0.0, float(st.session_state.input_SynAck), format="%.6f", key="synack_ui", help="Time between SYN and ACK.")
with cols[2]: user_inputs['AckDat'] = st.number_input('TCP ACK Data Time (s)', 0.0, float(st.session_state.input_AckDat), format="%.6f", key="ackdat_ui", help="Time for data acknowledgment.")

st.markdown("---")
col_btn_left, col_btn_mid, col_btn_right = st.columns([1.5, 2, 1.5])
with col_btn_mid:
    predict_button_clicked = st.button("🔍 Analyze Flow Status", key="predict_button_key_main", type="primary", use_container_width=True, help="Click to process input data and get a prediction.")

output_placeholder = st.container()
with output_placeholder:
    if predict_button_clicked:
        st.markdown("---")
        if model and scaler:
            for key_ui, value_ui in user_inputs.items(): st.session_state[f"input_{key_ui}"] = value_ui
            current_inputs_for_model_dict, all_features_present = {}, True
            for key_model in FINAL_FEATURE_COLS:
                if key_model in user_inputs: current_inputs_for_model_dict[key_model] = user_inputs[key_model]
                else: st.error(f"Input for '{key_model}' missing."); all_features_present = False; break
            if all_features_present:
                with st.spinner('🔬 Analyzing network flow data...'):
                    preprocessed_data, error_msg = preprocess_single_input(current_inputs_for_model_dict, scaler, FINAL_FEATURE_COLS, feature_mappings_to_apply)
                    time.sleep(1)
                st.subheader("📈 Analysis Results")
                if error_msg: st.error(f"**Analysis Error:** {error_msg}")
                elif preprocessed_data is not None:
                    try:
                        with st.spinner('🧠 Applying neural network model...'):
                            prediction_proba = model.predict(preprocessed_data)[0][0]
                            time.sleep(0.5)
                        prediction_label_num = 1 if prediction_proba > 0.5 else 0
                        prediction_label_str = reverse_label_mapping.get(prediction_label_num, "Unknown")
                        if prediction_label_num == 1: confidence, risk, delta_c, icon, msg_help = prediction_proba, "High Risk ❗", "inverse", "🚨", "Confidence: Malicious"
                        else: confidence, risk, delta_c, icon, msg_help = 1.0 - prediction_proba, "Low Risk ✅", "normal", "🛡️", "Confidence: Benign"
                        res_col1, res_col2 = st.columns(2); res_col1.metric(f"{icon} Predicted Flow Type", prediction_label_str, risk, delta_color=delta_c); res_col2.metric("Prediction Confidence", f"{confidence*100:.2f}%", help=msg_help)
                        st.progress(float(confidence))
                        st.info(f"**Status:** Flow classified as **{prediction_label_str}**.")
                        if prediction_label_str == 'Malicious':
                            if confidence > 0.85: st.error("Action Recommended: Investigate immediately (High Confidence Malicious).")
                            elif confidence > 0.6: st.warning("Caution: Shows malicious characteristics. Further monitoring advised.")
                        elif prediction_label_str == 'Benign' and confidence <= 0.85 and confidence > 0.5 : st.info("Observation: Benign, but confidence moderate. Standard monitoring applies.")
                        elif prediction_label_str == 'Benign': st.success("Observation: Flow appears normal with high confidence.")
                    except Exception as e_pred:
                        st.error(f"Model prediction failed: {e_pred}")
                        if debug_mode: import traceback; st.sidebar.error("Dev: Prediction Traceback:"); st.sidebar.code(traceback.format_exc())
                else: st.error("Unknown preprocessing error.")
        else: st.error("🔴 System Error: Model or Scaler not loaded. Cannot analyze.")
    else: st.info("ℹ️ Enter flow details and click 'Analyze Flow Status'.")

st.markdown("---")
st.caption(f"© {time.strftime('%Y')} OLORUKOOBA IBRAHIM KAYODE (20/52HA088) - 5G NIDD Anomaly Detection System")
