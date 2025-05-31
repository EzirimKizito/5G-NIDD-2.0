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
    page_icon="📡" # Added a page icon
)

# --- Configuration & Constants ---
# !!! IMPORTANT: UPDATE THESE PATHS IF NECESSARY !!!
MODEL_PATH = "nidd_model.h5"  # Assuming it's in the same directory as the script
SCALER_PATH = "scaler2.joblib" # Assuming it's in the same directory

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
    }

    /* Sidebar styling */
    .css-1d391kg { /* Streamlit's sidebar class might change, inspect if needed */
        background-color: #e8eef7; /* Lighter blue for sidebar */
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
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green */
    }

    /* Expander header */
    .st-emotion-cache-ff2z0k { /* Streamlit's expander header class */
        font-weight: bold;
        color: #333;
    }
    /* Metric labels */
    .st-emotion-cache-1r6slb0{
        font-weight: bold;
        color: #2c3e50; /* Darker text for metric labels */
    }
    /* Metric values */
    .st-emotion-cache-1g6goys{
         color: #1E3A5F; /* Dark blue for metric values */
         font-size: 1.5em !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
st.sidebar.title("👨‍💻 Project & System Info")
st.sidebar.markdown("---") # Visual separator

# Expander for App Information (collapsed by default)
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
        *   **Institution:** [Insert University Name Here - Placeholder]
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
        model = load_model(path) # Keras load_model handles .h5
        print(f"Model loaded successfully from {path}.") # For server-side logging
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
        print(f"Scaler loaded successfully from {path}.") # For server-side logging
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

# --- Load Artifacts ---
model = load_prediction_model(MODEL_PATH)
scaler = load_scaler_object(SCALER_PATH)

# --- Feature Definitions and Constraints ---
FINAL_FEATURE_COLS = [ # This order MUST match the order used for training the scaler and model
    'Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max', 'Proto', 'sTtl', 'sHops', 'Cause',
    'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes', 'DstBytes', 'Offset',
    'sMeanPktSz', 'dMeanPktSz', 'Load', 'SrcLoad', 'DstLoad', 'Rate', 'SrcRate',
    'DstRate', 'State', 'TcpRtt', 'SynAck', 'AckDat'
]
if model and scaler: # Only show if artifacts loaded
    st.sidebar.caption(f"Model configured for {len(FINAL_FEATURE_COLS)} input features.")
    with st.sidebar.expander("View Expected Model Features"):
        st.dataframe(pd.DataFrame({'Feature Name': FINAL_FEATURE_COLS}), use_container_width=True, height=200)


categorical_features_options = {
    'Proto': ['icmp', 'udp', 'tcp', 'sctp', 'ipv6-icmp'], # Order matters if mapping relies on index
    'Cause': ['Start', 'Status', 'Shutdown'],
    'State': ['ECO', 'CON', 'REQ', 'TST', 'RST', 'INT', 'FIN', 'URP', 'RSP', 'NRS', 'ACC'],
    'sTtl': [32, 58, 60, 63, 64, 117, 128, 249, 252, 255] # Treat as categorical for UI
}

# Default instance (using all 29 features, same as before but mapping might change if options changed)
default_benign_instance = {
    'Dur': 0.073378, 'RunTime': 0.073378, 'Mean': 0.073378, 'Sum': 0.073378,
    'Min': 0.073378, 'Max': 0.073378, 'Proto': 'udp', 'sTtl': 64, 'sHops': 0,
    'Cause': 'Status', 'TotPkts': 7, 'SrcPkts': 4, 'DstPkts': 3, 'TotBytes': 960,
    'SrcBytes': 760, 'DstBytes': 200, 'Offset': 47072, 'sMeanPktSz': 190.0,
    'dMeanPktSz': 66.666664, 'Load': 76753.25, 'SrcLoad': 62143.96875,
    'DstLoad': 14609.28418, 'Rate': 81.768379, 'SrcRate': 40.88419,
    'DstRate': 27.256126, 'State': 'CON', 'TcpRtt': 0.0, 'SynAck': 0.0, 'AckDat': 0.0
}

# --- Preprocessing Definitions ---
# Mappings should correspond to how the model was trained
proto_mapping = {'icmp': 0, 'udp': 1, 'tcp': 2, 'sctp': 3, 'ipv6-icmp': 4}
cause_mapping = {'Start': 0, 'Status': 1, 'Shutdown': 2}
state_mapping = {'ECO': 0, 'CON': 1, 'REQ': 2, 'TST': 3, 'RST': 4, 'INT': 5, 'FIN': 6, 'URP': 7, 'RSP': 8, 'NRS': 9, 'ACC': 10}
# sTtl is also treated as categorical for input, but will be a direct numerical value for the model
# No specific mapping dictionary needed for sTtl if it's already numeric in FINAL_FEATURE_COLS

reverse_proto_mapping = {v: k for k, v in proto_mapping.items()}
reverse_cause_mapping = {v: k for k, v in cause_mapping.items()}
reverse_state_mapping = {v: k for k, v in state_mapping.items()}

feature_mappings_to_apply = {
    'Proto': proto_mapping,
    'Cause': cause_mapping,
    'State': state_mapping,
}
label_mapping = {'Benign': 0, 'Malicious': 1} # Assuming this was your training encoding
reverse_label_mapping = {v: k for k, v in label_mapping.items()}


# --- Preprocessing Functions ---
def apply_manual_encoding_deploy(df, mappings):
    df_encoded = df.copy()
    cols_processed_for_numeric = []
    for col, mapping in mappings.items():
        if col in df_encoded.columns and isinstance(mapping, dict):
            original_value = df_encoded.iloc[0][col] # Works for single row DataFrame
            if pd.isna(original_value):
                df_encoded.loc[0, col] = np.nan
            else:
                mapped_value = mapping.get(original_value)
                if mapped_value is None:
                    if debug_mode: st.sidebar.warning(f"Value '{original_value}' not in mapping for '{col}'. Setting NaN.")
                    df_encoded.loc[0, col] = np.nan
                else:
                    df_encoded.loc[0, col] = mapped_value # Assign mapped numeric value
            cols_processed_for_numeric.append(col)

    # Ensure columns that were mapped are numeric
    for col in cols_processed_for_numeric:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    return df_encoded

def preprocess_single_input(input_dict, scaler_obj, feature_cols_list, mappings_to_apply):
    if debug_mode:
        st.sidebar.divider()
        st.sidebar.write("--- Dev: Preprocessing Start ---")
        st.sidebar.json({"Input Dictionary (UI)": input_dict})

    input_df = pd.DataFrame([input_dict])
    if debug_mode: st.sidebar.caption("Input DataFrame (1 row) dtypes:"); st.sidebar.code(input_df.dtypes)

    # Apply manual encoding for string-based categoricals
    df_processed = apply_manual_encoding_deploy(input_df.copy(), mappings_to_apply)
    if debug_mode:
        st.sidebar.caption("DataFrame AFTER Encoding - dtypes:"); st.sidebar.code(df_processed.dtypes)
        st.sidebar.caption("DataFrame AFTER Encoding - values:"); st.sidebar.code(df_processed.iloc[0].to_dict())

    # Ensure all feature columns are present and in the correct order
    try:
        missing_cols = [col for col in feature_cols_list if col not in df_processed.columns]
        if missing_cols:
            return None, f"Critical Error: Columns missing after internal processing: {missing_cols}. This indicates a mismatch in feature definitions."
        df_features = df_processed[feature_cols_list] # Reorder to match training
    except KeyError as e:
        return None, f"Critical Error: Required feature column '{e}' not found after processing. Check FINAL_FEATURE_COLS consistency."
    except Exception as e:
        return None, f"Error during feature selection/ordering: {e}"

    if debug_mode:
        st.sidebar.caption(f"DataFrame FEATURES selected ({len(feature_cols_list)}) - dtypes:"); st.sidebar.code(df_features.dtypes)
        st.sidebar.caption("Feature Order Used for Scaling:"); st.sidebar.code(df_features.columns.tolist())
        st.sidebar.caption("Feature Values before Scaling:"); st.sidebar.code(df_features.iloc[0].to_dict())

    # Check for NaNs before scaling (critical after encoding)
    if df_features.isnull().values.any():
        nan_cols = df_features.columns[df_features.isnull().any(axis=0)].tolist()
        return None, f"Data Error: Missing or invalid values detected before scaling in features: {nan_cols}. Please check inputs or encoding logic."

    # Final check for non-numeric types (should not happen if encoding is correct)
    non_numeric_cols = df_features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        return None, f"Internal Error: Non-numeric columns found before scaling: {non_numeric_cols}. This indicates an issue with the encoding process."

    # Apply scaling
    try:
        if debug_mode:
            st.sidebar.write(f"Scaler expects {scaler_obj.n_features_in_} features. Data has {df_features.shape[1]} features.")
        if df_features.shape[1] != scaler_obj.n_features_in_:
             return None, f"Shape Mismatch Error: Scaler expects {scaler_obj.n_features_in_} features, but received data with {df_features.shape[1]} features."

        scaled_data = scaler_obj.transform(df_features)
        if debug_mode:
            st.sidebar.caption("Scaled Data (NumPy array):"); st.sidebar.code(scaled_data)
            st.sidebar.write(f"Scaled Data Shape: {scaled_data.shape}")

        if np.isnan(scaled_data).any():
            return None, "Data Error: NaNs were generated during the scaling process. This might be due to extreme values or issues with the scaler's learned parameters."
    except ValueError as ve:
        return None, f"Scaling Error: {ve}. This often means the data being scaled is not purely numeric or has an unexpected shape."
    except Exception as e:
        return None, f"An unexpected error occurred during data scaling: {e}"

    if debug_mode: st.sidebar.write("--- Dev: Preprocessing End ---"); st.sidebar.divider()
    return scaled_data, None

# --- Initialize Session State ---
# Ensure session state keys are initialized from default_benign_instance
# This allows UI elements to pick up these values on first load or after paste-fill
if 'initialized' not in st.session_state:
    for key, value in default_benign_instance.items():
        st.session_state[f"input_{key}"] = value
    st.session_state.pasted_text = ""
    st.session_state.initialized = True # Mark as initialized

# --- UI Layout and Input Fields ---
st.title("📡 5G NIDD Anomaly Detection System")
st.markdown("Welcome! Please input the network flow parameters below or use the paste option. Click 'Analyze Flow Status' for a prediction.")
st.markdown("---")

# --- Paste-and-Fill Section ---
with st.expander("📋 Paste Instance Data (Optional - Advanced Users)", expanded=False):
    st.markdown("""
    You can paste data where each line is `FeatureName: Value`.
    *   **Important:** For features like `Proto`, `Cause`, and `State`, you must provide their **encoded numerical values** (e.g., `Proto: 1` for 'udp').
    *   `sTtl` should be one of the predefined numerical values.
    *   Only features listed under "View Expected Model Features" in the sidebar will be parsed.
    """)
    pasted_text_area = st.text_area("Paste encoded data here:", value=st.session_state.pasted_text, height=250, key="paste_area_key", label_visibility="collapsed")

    if st.button("📄 Apply Pasted Data to Inputs", key="paste_button_key"):
        st.session_state.pasted_text = pasted_text_area # Update state with current text
        parsed_values_from_paste = {}
        parsing_errors = []
        if st.session_state.pasted_text:
            lines = st.session_state.pasted_text.strip().split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*:\s*(.*)$", line)
                if match:
                    feature_name_from_paste = match.group(1).strip()
                    value_str_from_paste = match.group(2).strip()

                    if feature_name_from_paste not in FINAL_FEATURE_COLS: continue # Skip if not a model feature

                    try:
                        # For selectbox-driven features, try to map back if numeric code is pasted
                        if feature_name_from_paste == 'Proto':
                            parsed_val = reverse_proto_mapping.get(int(float(value_str_from_paste)))
                            if parsed_val is None: raise ValueError("Invalid Proto code")
                        elif feature_name_from_paste == 'Cause':
                            parsed_val = reverse_cause_mapping.get(int(float(value_str_from_paste)))
                            if parsed_val is None: raise ValueError("Invalid Cause code")
                        elif feature_name_from_paste == 'State':
                            parsed_val = reverse_state_mapping.get(int(float(value_str_from_paste)))
                            if parsed_val is None: raise ValueError("Invalid State code")
                        elif feature_name_from_paste == 'sTtl':
                            parsed_val = int(float(value_str_from_paste))
                            if parsed_val not in categorical_features_options['sTtl']: raise ValueError("Invalid sTtl value")
                        else: # For other numeric features
                            parsed_val = float(value_str_from_paste)
                            # Convert to int if it's an integer-like feature
                            int_cols = ['sHops', 'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes', 'DstBytes', 'Offset']
                            if feature_name_from_paste in int_cols: parsed_val = int(parsed_val)
                        parsed_values_from_paste[feature_name_from_paste] = parsed_val
                    except ValueError as e_parse:
                        parsing_errors.append(f"L{i+1} ('{feature_name_from_paste}'): Invalid value '{value_str_from_paste}' - {e_parse}")
                else:
                    parsing_errors.append(f"L{i+1}: Invalid format '{line}'")

            updated_count = 0
            for key, value in parsed_values_from_paste.items():
                if f"input_{key}" in st.session_state and value is not None:
                    st.session_state[f"input_{key}"] = value
                    updated_count +=1
            if updated_count > 0: st.success(f"Successfully applied {updated_count} values from pasted text.")
            if parsing_errors: st.warning("Pasting Issues:\n" + "\n".join(parsing_errors))
            st.rerun() # Force UI update
        else:
            st.info("Paste area is empty.")
st.markdown("---")


user_inputs = {} # Collect current widget values

# --- UI Input Sections ---
st.subheader("📊 Flow Characteristics Input")
st.markdown("Fill in the parameters of the network flow you wish to analyze.")

# Row 1: Timing
c1, c2, c3, c4 = st.columns(4)
with c1: user_inputs['Dur'] = st.number_input('Duration (s)', min_value=0.0, value=float(st.session_state.input_Dur), format="%.6f", key="dur_ui", help="Total duration of the flow.")
with c2: user_inputs['RunTime'] = st.number_input('Run Time (s)', min_value=0.0, value=float(st.session_state.input_RunTime), format="%.6f", key="runtime_ui", help="Effective runtime of the flow.")
with c3: user_inputs['Mean'] = st.number_input('Mean Inter-Packet Time (s)', min_value=0.0, value=float(st.session_state.input_Mean), format="%.6f", key="mean_ui", help="Mean time between packets.")
with c4: user_inputs['Sum'] = st.number_input('Sum of Inter-Packet Times (s)', min_value=0.0, value=float(st.session_state.input_Sum), format="%.6f", key="sum_ui", help="Total of all inter-packet times.")

# Row 2: Min/Max Time & Packet Counts
c1, c2, c3, c4 = st.columns(4)
with c1: user_inputs['Min'] = st.number_input('Min Inter-Packet Time (s)', min_value=0.0, value=float(st.session_state.input_Min), format="%.6f", key="min_ui", help="Minimum time between packets.")
with c2: user_inputs['Max'] = st.number_input('Max Inter-Packet Time (s)', min_value=0.0, value=float(st.session_state.input_Max), format="%.6f", key="max_ui", help="Maximum time between packets.")
with c3: user_inputs['TotPkts'] = st.number_input('Total Packets', min_value=0, value=int(st.session_state.input_TotPkts), step=1, key="totpkts_ui", help="Total number of packets in the flow.")
with c4: user_inputs['Offset'] = st.number_input('Offset (bytes)', min_value=0, value=int(st.session_state.input_Offset), step=1, key="offset_ui", help="Data offset, if applicable.")


st.markdown("---")
st.markdown("##### **Packet & Byte Counts**")
c1, c2, c3 = st.columns(3)
with c1:
    user_inputs['SrcPkts'] = st.number_input('Source Packets', min_value=0, value=int(st.session_state.input_SrcPkts), step=1, key="srcpkts_ui", help="Number of packets from source.")
    user_inputs['DstPkts'] = st.number_input('Destination Packets', min_value=0, value=int(st.session_state.input_DstPkts), step=1, key="dstpkts_ui", help="Number of packets to destination.")
with c2:
    user_inputs['TotBytes'] = st.number_input('Total Bytes', min_value=0, value=int(st.session_state.input_TotBytes), step=1, key="totbytes_ui", help="Total bytes in the flow.")
    user_inputs['SrcBytes'] = st.number_input('Source Bytes', min_value=0, value=int(st.session_state.input_SrcBytes), step=1, key="srcbytes_ui", help="Bytes sent by source.")
with c3:
    user_inputs['DstBytes'] = st.number_input('Destination Bytes', min_value=0, value=int(st.session_state.input_DstBytes), step=1, key="dstbytes_ui", help="Bytes received by destination.")
    user_inputs['sMeanPktSz'] = st.number_input('Source Mean Packet Size', min_value=0.0, value=float(st.session_state.input_sMeanPktSz), format="%.2f", key="smeansz_ui", help="Average size of source packets.")


st.markdown("---")
st.markdown("##### **Protocol, Routing & State Information**")
c1, c2, c3, c4 = st.columns(4)
with c1:
    proto_options = categorical_features_options['Proto']
    current_proto_ui = st.session_state.input_Proto
    proto_index_ui = proto_options.index(current_proto_ui) if current_proto_ui in proto_options else 0
    user_inputs['Proto'] = st.selectbox('Protocol', options=proto_options, index=proto_index_ui, key="proto_ui", help="Network protocol used.")
with c2:
    sttl_options = categorical_features_options['sTtl']
    current_sttl_ui = st.session_state.input_sTtl
    sttl_index_ui = sttl_options.index(current_sttl_ui) if current_sttl_ui in sttl_options else (sttl_options.index(64) if 64 in sttl_options else 0)
    user_inputs['sTtl'] = st.selectbox('Source TTL', options=sttl_options, index=sttl_index_ui, key="sttl_ui", help="Source Time-To-Live value.")
with c3:
    user_inputs['sHops'] = st.number_input('Source Hops', min_value=0, value=int(st.session_state.input_sHops), step=1, key="shops_ui", help="Number of hops from source.")
with c4:
    state_options = categorical_features_options['State']
    current_state_ui = st.session_state.input_State
    state_index_ui = state_options.index(current_state_ui) if current_state_ui in state_options else 0
    user_inputs['State'] = st.selectbox('Flow State', options=state_options, index=state_index_ui, key="state_ui", help="Current state of the flow.")

st.markdown("---")
st.markdown("##### **Network Load, Rate & Flow Termination**")
c1, c2, c3 = st.columns(3)
with c1:
     user_inputs['Load'] = st.number_input('Overall Load (bps)', min_value=0.0, value=float(st.session_state.input_Load), format="%.2f", key="load_ui", help="Total network load in bits per second.")
     user_inputs['SrcLoad'] = st.number_input('Source Load (bps)', min_value=0.0, value=float(st.session_state.input_SrcLoad), format="%.2f", key="srcload_ui", help="Load generated by the source.")
     user_inputs['DstLoad'] = st.number_input('Destination Load (bps)', min_value=0.0, value=float(st.session_state.input_DstLoad), format="%.2f", key="dstload_ui", help="Load received by the destination.")
with c2:
    user_inputs['Rate'] = st.number_input('Overall Rate (pps)', min_value=0.0, value=float(st.session_state.input_Rate), format="%.2f", key="rate_ui", help="Total packet rate in packets per second.")
    user_inputs['SrcRate'] = st.number_input('Source Rate (pps)', min_value=0.0, value=float(st.session_state.input_SrcRate), format="%.2f", key="srcrate_ui", help="Packet rate from the source.")
    user_inputs['DstRate'] = st.number_input('Destination Rate (pps)', min_value=0.0, value=float(st.session_state.input_DstRate), format="%.2f", key="dstrate_ui", help="Packet rate to the destination.")
with c3:
    user_inputs['dMeanPktSz'] = st.number_input('Destination Mean Packet Size', min_value=0.0, value=float(st.session_state.input_dMeanPktSz), format="%.6f", key="dmeansz_ui", help="Average size of destination packets.")
    cause_options = categorical_features_options['Cause']
    current_cause_ui = st.session_state.input_Cause
    cause_index_ui = cause_options.index(current_cause_ui) if current_cause_ui in cause_options else 0
    user_inputs['Cause'] = st.selectbox('Flow Termination Cause', options=cause_options, index=cause_index_ui, key="cause_ui", help="Reason for flow termination.")

st.markdown("---")
st.markdown("##### **TCP Specific Metrics** (Enter 0 if not applicable, e.g., for UDP/ICMP)")
c1, c2, c3 = st.columns(3)
with c1: user_inputs['TcpRtt'] = st.number_input('TCP Round Trip Time (s)', min_value=0.0, value=float(st.session_state.input_TcpRtt), format="%.6f", key="tcprtt_ui", help="TCP Round Trip Time.")
with c2: user_inputs['SynAck'] = st.number_input('TCP SYN-ACK Time (s)', min_value=0.0, value=float(st.session_state.input_SynAck), format="%.6f", key="synack_ui", help="Time between SYN and ACK in TCP handshake.")
with c3: user_inputs['AckDat'] = st.number_input('TCP ACK Data Time (s)', min_value=0.0, value=float(st.session_state.input_AckDat), format="%.6f", key="ackdat_ui", help="Time for data acknowledgment in TCP.")


# --- Prediction Button and Output Area ---
st.markdown("---")
# Centering the button using columns trick
col_btn_left, col_btn_mid, col_btn_right = st.columns([1.5, 2, 1.5])
with col_btn_mid:
    predict_button_clicked = st.button("🔍 Analyze Flow Status", key="predict_button_key_main", type="primary", use_container_width=True, help="Click to process the input data and get a prediction.")

output_placeholder = st.container()

with output_placeholder:
    if predict_button_clicked:
        st.markdown("---") # Separator before results
        if model and scaler:
            # Update session state from current widget values before preprocessing
            for key_ui, value_ui in user_inputs.items():
                 st.session_state[f"input_{key_ui}"] = value_ui

            current_inputs_for_model_dict = {}
            all_features_present = True
            for key_model in FINAL_FEATURE_COLS: # Ensure all model features are accounted for
                if key_model in user_inputs:
                    current_inputs_for_model_dict[key_model] = user_inputs[key_model]
                else:
                    # This case should ideally not happen if UI is comprehensive
                    st.error(f"Input for feature '{key_model}' is missing from the UI form.")
                    all_features_present = False
                    break # Stop processing if a feature is missing

            if all_features_present:
                with st.spinner('🔬 Analyzing network flow data... This may take a moment.'):
                    preprocessed_data, error_msg = preprocess_single_input(
                        current_inputs_for_model_dict, scaler, FINAL_FEATURE_COLS, feature_mappings_to_apply
                    )
                    time.sleep(1) # Simulate some processing time

                st.subheader("📈 Analysis Results")
                if error_msg:
                    st.error(f"**Analysis Error:** {error_msg}")
                elif preprocessed_data is not None:
                    try:
                        with st.spinner('🧠 Applying neural network model...'):
                            if debug_mode: st.sidebar.caption(f"Dev: Data shape to model: {preprocessed_data.shape}")
                            prediction_proba = model.predict(preprocessed_data)[0][0]
                            if debug_mode: st.sidebar.caption(f"Dev: Raw prediction probability: {prediction_proba:.8f}")
                            time.sleep(0.5)

                        prediction_label_num = 1 if prediction_proba > 0.5 else 0
                        prediction_label_str = reverse_label_mapping.get(prediction_label_num, "Unknown")

                        if prediction_label_num == 1: # Malicious
                            confidence_in_prediction = prediction_proba
                            risk_assessment = "High Risk ❗"
                            delta_color = "inverse"
                            status_icon = "🚨"
                            status_message = f"**Alert:** Flow classified as **{prediction_label_str}**."
                            confidence_help = "This is the model's confidence that the flow is Malicious."
                        else: # Benign
                            confidence_in_prediction = 1.0 - prediction_proba
                            risk_assessment = "Low Risk ✅"
                            delta_color = "normal"
                            status_icon = "🛡️"
                            status_message = f"**Status:** Flow classified as **{prediction_label_str}**."
                            confidence_help = "This is the model's confidence that the flow is Benign."

                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric(label="Predicted Flow Type", value=f"{status_icon} {prediction_label_str}", delta=risk_assessment, delta_color=delta_color)
                        with res_col2:
                            st.metric(label="Prediction Confidence", value=f"{confidence_in_prediction*100:.2f}%", help=confidence_help)

                        st.progress(float(confidence_in_prediction))
                        st.info(status_message)

                        if prediction_label_str == 'Malicious':
                            if confidence_in_prediction > 0.85:
                                st.error("Action Recommended: Investigate this flow immediately due to high confidence malicious prediction.")
                            elif confidence_in_prediction > 0.6:
                                st.warning("Caution: This flow shows characteristics of malicious activity. Further monitoring is advised.")
                        elif prediction_label_str == 'Benign':
                             if confidence_in_prediction > 0.85:
                                 st.success("Observation: Flow appears normal with high confidence.")
                             else:
                                 st.info("Observation: Flow classified as benign, but confidence is moderate. Standard monitoring practices apply.")


                    except Exception as e_pred:
                        st.error(f"Model prediction failed: {e_pred}")
                        if debug_mode:
                             import traceback
                             st.sidebar.error("Dev: Prediction Traceback:")
                             st.sidebar.code(traceback.format_exc())
                else:
                    st.error("An unknown error occurred during data preprocessing.")
        else:
            st.error("🔴 System Error: Model or Scaler not loaded. Cannot perform analysis. Please check file paths and logs.")
    else:
        st.info("ℹ️ Enter flow details and click 'Analyze Flow Status' to get a prediction.")

# Footer
st.markdown("---")
st.caption(f"© {time.strftime('%Y')} OLORUKOOBA IBRAHIM KAYODE (20/52HA088) - 5G NIDD Anomaly Detection System")
