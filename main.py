"""
main.py: Training Script for Campaign Recommendation GNN

This script performs the end-to-end training pipeline for a GraphSAGE-based
recommender system. It takes user profile and campaign data as input,
preprocesses features (including SBERT embeddings), builds a heterogeneous graph,
creates interaction edges based on heuristics, trains a link prediction model,
evaluates it using AUC, Precision@k, and NDCG@k, and saves the final model
artifacts, preprocessed data, and graph structure.

Usage:
    python main.py

Requires:
    - data/UserProfiles_Data.csv
    - data/Campaigns_Data.csv
    - Correctly installed dependencies (see requirements.txt)
    - Manually verified/customized LOCATION_* maps in the script.
"""
# -*- coding: utf-8 -*-
# === Imports ===
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import logging
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, ReLU, Dropout, BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau # For LR scheduling
from tqdm.auto import tqdm
import collections
import gc # Garbage collector
import random # Needed for simulating device targeting


# === Dependency Checks ===
try: from sentence_transformers import SentenceTransformer
except ImportError as e: print(f"CRITICAL ERROR: sentence-transformers library not found: {e}\nPlease install it."); exit(1)
try: from torch_geometric.nn import SAGEConv; from torch_geometric.data import Data; from torch_geometric.utils import negative_sampling, train_test_split_edges; import torch_scatter
except ImportError as e: print(f"CRITICAL ERROR: torch-geometric or dependencies not found: {e}\nPlease install carefully."); exit(1)

# === Configuration ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-[%(filename)s:%(lineno)d]-%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
tqdm.pandas(desc="Pandas Apply")

# --- File/Directory Paths ---
try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()
logger.info(f"Using base directory: {BASE_DIR}")
DATA_DIR = os.path.join(BASE_DIR, "data"); PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed_data_gnn")
# Using final artifact names
MODEL_ARTIFACTS_FILE = "recommender_gnn_artifacts_hackathon_final_v2.pkl"; CAMPAIGNS_PREPROCESSED_FILE = "campaigns_preprocessed_gnn_hackathon_final_v2.csv"; GRAPH_DATA_FILE = "graph_data_hackathon_final_v2.pt"
USER_CSV_FILE = 'UserProfiles_Data.csv'; CAMPAIGN_CSV_FILE = 'Campaigns_Data.csv'

# --- Dataset Column Names ---
USER_ID_COL = 'user_id'; USER_AGE_COL = 'age_group'; USER_LOCATION_COL = 'location'; USER_LOCATION_TIER_COL = 'location_tier'; USER_DEVICE_COL = 'device_type'; USER_INTERESTS_COL = 'interests'; USER_ACTIVITY_COL = 'activity_level'; USER_MONETARY_COL = 'monetary_level'; USER_PAST_REWARDS_COL = 'past_rewards_list'; USER_WATCH_CATEGORIES_COL = 'recent_watch_categories'; USER_WEATHER_COL = 'simulated_weather'; USER_CREATION_DATE_COL = 'profile_creation_date'
CAMPAIGN_ID_COL = 'campaign_id'; CAMPAIGN_BUSINESS_NAME_COL = 'business_name'; CAMPAIGN_CATEGORY_COL = 'category'; CAMPAIGN_LOCATION_COL = 'business_location'; CAMPAIGN_PROMO_COL = 'promo'; CAMPAIGN_START_TIME_COL = 'start_time'; CAMPAIGN_END_TIME_COL = 'end_time'; CAMPAIGN_TARGET_GROUP_COL = 'target_group'; CAMPAIGN_BUDGET_COL = 'budget'; CAMPAIGN_TARGET_DEVICE_COL = 'target_device' # Added target device column name used internally

# --- Device Types ---
DEVICE_TYPES = ["android smartphone (mid-range)", "android smartphone (high-end)", "ios smartphone", "tablet (android)", "tablet (ios)", "windows desktop/laptop", "macos desktop/laptop", "unknown"] # Removed feature phone

# --- Hardware Setup ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); logger.info(f"Selected device: {DEVICE}")
if DEVICE.type == 'cuda':
    try: torch.cuda.empty_cache(); gpu_name = torch.cuda.get_device_name(0); total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9; logger.info(f"CUDA GPU: {gpu_name}, Total Memory: {total_mem:.2f} GB")
    except Exception as e: logger.warning(f"CUDA check failed: {e}.")
else: logger.warning("CUDA not available, running on CPU.")

# --- Embedding Setup ---
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
try: sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE); EMBEDDING_DIM = sbert_model.get_sentence_embedding_dimension(); logger.info(f"SBERT model '{SBERT_MODEL_NAME}' loaded. Dim: {EMBEDDING_DIM}")
except Exception as e: logger.error(f"CRITICAL SBERT load error: {e}", exc_info=True); raise RuntimeError(f"Could not load SBERT model.")
zero_embedding_count = 0

# --- !!! Essential Location Processing Info (CUSTOMIZE!) !!! ---
logger.info("Initializing Location Maps - PLEASE VERIFY AND CUSTOMIZE THESE MANUALLY!")
LOCATION_STANDARDIZATION_MAP = {"delhi (ncr)": "delhi", "new delhi": "delhi", "delhi": "delhi", "bangalore (bengaluru)": "bangalore", "bengaluru": "bangalore", "mumbai": "mumbai", "varanasi (benaras)": "varanasi", "benaras": "varanasi", "chennai": "chennai", "kolkata": "kolkata", "hyderabad": "hyderabad", "pune": "pune", "ahmedabad": "ahmedabad", "jaipur": "jaipur", "surat": "surat", "lucknow": "lucknow", "kanpur": "kanpur", "nagpur": "nagpur", "indore": "indore", "thane": "thane", "bhopal": "bhopal", "visakhapatnam": "visakhapatnam", "patna": "patna", "vadodara": "vadodara", "ludhiana": "ludhiana", "agra": "agra", "nashik": "nashik", "coimbatore": "coimbatore", "kochi": "kochi", "chandigarh": "chandigarh", "mysore": "mysore", "amritsar": "amritsar", "guwahati": "guwahati", "shimla": "shimla", "goa": "goa", "rishikesh": "rishikesh", "udaipur": "udaipur", "darjeeling": "darjeeling", "madurai": "madurai", "jodhpur": "jodhpur", "puducherry": "puducherry", "aurangabad": "aurangabad", "dehradun": "dehradun", "bhubaneswar": "bhubaneswar", "raipur": "raipur", "gurgaon": "gurugram", "gurugram": "gurugram", "unknown": "unknown", "nan": "unknown", "": "unknown"}
LOCATION_TIERS_MAP = {"mumbai": "Tier 1", "delhi": "Tier 1", "bangalore": "Tier 1", "chennai": "Tier 1", "kolkata": "Tier 1", "hyderabad": "Tier 1", "pune": "Tier 1", "ahmedabad": "Tier 1", "gurugram": "Tier 1", "noida": "Tier 1", "jaipur": "Tier 2", "surat": "Tier 2", "lucknow": "Tier 2", "kanpur": "Tier 2", "nagpur": "Tier 2", "indore": "Tier 2", "thane": "Tier 2", "bhopal": "Tier 2", "visakhapatnam": "Tier 2", "patna": "Tier 2", "vadodara": "Tier 2", "ludhiana": "Tier 2", "agra": "Tier 2", "nashik": "Tier 2", "coimbatore": "Tier 2", "kochi": "Tier 2", "chandigarh": "Tier 2", "mysore": "Tier 2", "varanasi": "Tier 3", "amritsar": "Tier 3", "guwahati": "Tier 3", "madurai": "Tier 3", "jodhpur": "Tier 3", "aurangabad": "Tier 3", "dehradun": "Tier 3", "bhubaneswar": "Tier 3", "raipur": "Tier 3", "shimla": "Specialty", "goa": "Specialty", "rishikesh": "Specialty", "udaipur": "Specialty", "darjeeling": "Specialty", "puducherry": "Specialty", "unknown": "Unknown"}
LOCATION_REGIONS_MAP = {"NCR": ["delhi", "gurugram", "noida", "faridabad", "ghaziabad"], "Mumbai MMR": ["mumbai", "thane", "navi mumbai"], "South Tier 1": ["bangalore", "chennai", "hyderabad"], "West Tier 1/2": ["pune", "ahmedabad", "surat", "vadodara", "indore", "nagpur", "nashik", "bhopal", "aurangabad"], "North Tier 2/3/S": ["jaipur", "lucknow", "kanpur", "ludhiana", "chandigarh", "agra", "amritsar", "shimla", "dehradun", "rishikesh", "jodhpur"], "East Tier 1/2/3/S": ["kolkata", "patna", "bhubaneswar", "guwahati", "darjeeling", "raipur"], "South Tier 2/3/S": ["visakhapatnam", "coimbatore", "kochi", "mysore", "madurai", "puducherry", "goa", "varanasi"], "Unknown Region": ["unknown"]}

# === Helper Functions ===
def clear_cuda_cache(context=""):
     if DEVICE.type == 'cuda': gc.collect(); torch.cuda.empty_cache(); logger.debug(f"Cleared CUDA cache ({context}).")

def get_sbert_batch_embeddings(texts, model, batch_size=64):
    global EMBEDDING_DIM # Rely on global EMBEDDING_DIM
    if not model: logger.error("SBERT model not loaded!"); return [np.zeros(EMBEDDING_DIM)] * len(texts)
    if not isinstance(texts, list): texts = list(texts)
    processed_texts = [str(t).strip() if pd.notna(t) and str(t).strip() else " " for t in texts]
    original_indices = [i for i, t in enumerate(processed_texts) if t != " "]
    embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    valid_texts = [processed_texts[i] for i in original_indices]
    if not valid_texts: return list(embeddings) # No need to track zero count globally here
    try:
        show_bar = False # No progress bar for internal calls
        valid_embeddings = model.encode(valid_texts, convert_to_numpy=True, device=DEVICE, batch_size=batch_size, show_progress_bar=show_bar)
        if valid_embeddings.shape[0] != len(valid_texts): logger.warning(f"SBERT encode returned {valid_embeddings.shape[0]} embeds for {len(valid_texts)} inputs!")
        for i, embed_idx in enumerate(original_indices):
             if i < len(valid_embeddings): embeddings[embed_idx] = valid_embeddings[i]
        return list(embeddings)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e): logger.warning(f"CUDA OOM during batch SBERT encoding."); clear_cuda_cache("SBERT OOM")
        else: logger.warning(f"Runtime error during batch SBERT encoding: {e}.");
    except Exception as e: logger.warning(f"General error during batch SBERT encoding: {e}.");
    return list(np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)) # Return zeros on error

def get_sbert_embedding(text, model): return get_sbert_batch_embeddings([text], model, batch_size=1)[0]
def get_sbert_list_embedding(list_text, model):
    global EMBEDDING_DIM
    if not model: return np.zeros(EMBEDDING_DIM)
    if not isinstance(list_text, str) or not list_text.strip(): return np.zeros(EMBEDDING_DIM)
    items = [item.strip() for item in list_text.split(',') if item.strip()]
    if not items: return np.zeros(EMBEDDING_DIM)
    item_embeddings_list = get_sbert_batch_embeddings(items, model, batch_size=len(items))
    item_embeddings = np.array(item_embeddings_list); valid_mask = np.any(item_embeddings != 0, axis=1)
    if not np.any(valid_mask): return np.zeros(EMBEDDING_DIM)
    avg_embedding = np.mean(item_embeddings[valid_mask], axis=0)
    if np.any(~np.isfinite(avg_embedding)): logger.warning(f"Non-finite value after averaging list: '{list_text[:50]}...'."); return np.zeros(EMBEDDING_DIM)
    return avg_embedding

def standardize_location(loc_str):
    loc_str = str(loc_str).lower().strip()
    return LOCATION_STANDARDIZATION_MAP.get(loc_str, loc_str) # Default to cleaned original if not mapped
def get_location_tier(standardized_location):
    return LOCATION_TIERS_MAP.get(standardized_location, "Unknown")
def get_location_region(standardized_location): # Used only in prediction but keep here for consistency
    for region, locations in LOCATION_REGIONS_MAP.items():
        if standardized_location in locations: return region
    return "Other"
def validate_schema(df, required_cols, df_type, func_name="validate_schema"):
    missing = [col for col in required_cols if col not in df.columns]
    if missing: raise ValueError(f"[{func_name}] Missing columns in {df_type}: {missing}. Check CSV headers/paths.")
    logger.debug(f"[{func_name}] Schema validation passed for {df_type}.")
    return True

# === Core Functions ===

# --- Preprocessing ---
def preprocess_data_gnn(users_df_orig: pd.DataFrame, campaigns_df_orig: pd.DataFrame):
    global zero_embedding_count, sbert_model, EMBEDDING_DIM # Need zero_embedding_count for final log
    zero_embedding_count = 0; func_name = "preprocess_data_gnn"; logger.info(f"[{func_name}] Starting...")
    users_df = users_df_orig.copy(); campaigns_df = campaigns_df_orig.copy()
    logger.info(f"[{func_name}] Initial shapes - Users: {users_df.shape}, Campaigns: {campaigns_df.shape}")

    # --- User Preprocessing ---
    logger.info(f"[{func_name}] --- Processing User Data ---")
    # Define columns expected based on variable names
    expected_user_cols = [USER_ID_COL, USER_AGE_COL, USER_LOCATION_COL, USER_LOCATION_TIER_COL, USER_DEVICE_COL, USER_INTERESTS_COL, USER_ACTIVITY_COL, USER_MONETARY_COL, USER_PAST_REWARDS_COL, USER_WATCH_CATEGORIES_COL, USER_WEATHER_COL, USER_CREATION_DATE_COL]
    user_defaults = {USER_AGE_COL: 'unknown', USER_LOCATION_COL: 'unknown', USER_INTERESTS_COL: 'general', USER_ACTIVITY_COL: 'medium (weekly)', USER_MONETARY_COL: 'medium spender', USER_PAST_REWARDS_COL: '', USER_WATCH_CATEGORIES_COL: 'general', USER_WEATHER_COL: 'clear skies', USER_CREATION_DATE_COL: datetime.now().strftime('%Y-%m-%d'), USER_LOCATION_TIER_COL: 'Unknown', USER_DEVICE_COL: 'unknown'}
    missing_user_cols_count = 0;
    for col in expected_user_cols:
        if col not in users_df.columns: default_val = user_defaults.get(col, 'unknown'); logger.warning(f"[{func_name}] User column '{col}' missing. Adding default '{default_val}'."); users_df[col] = default_val; missing_user_cols_count += 1
    if missing_user_cols_count > 0: logger.info(f"[{func_name}] Added {missing_user_cols_count} missing user columns.")
    users_df.fillna(user_defaults, inplace=True)
    user_cols_to_standardize = [USER_INTERESTS_COL, USER_PAST_REWARDS_COL, USER_WATCH_CATEGORIES_COL, USER_LOCATION_COL, USER_WEATHER_COL, USER_AGE_COL, USER_ACTIVITY_COL, USER_MONETARY_COL, USER_DEVICE_COL, USER_LOCATION_TIER_COL]
    logger.info(f"[{func_name}] Standardizing user text/categorical columns...")
    for col in user_cols_to_standardize:
        if col in users_df.columns: users_df[col] = users_df[col].astype(str).str.lower().str.strip()
    logger.info(f"[{func_name}] Standardizing user locations & deriving tiers...")
    if USER_LOCATION_COL not in users_df.columns: raise ValueError(f"User location column '{USER_LOCATION_COL}' not found!")
    users_df['location_standardized'] = users_df[USER_LOCATION_COL].progress_apply(standardize_location)
    users_df[USER_LOCATION_TIER_COL] = users_df['location_standardized'].progress_apply(get_location_tier)

    logger.info(f"[{func_name}] Generating user SBERT embeddings (batch)...")
    if USER_INTERESTS_COL not in users_df.columns: raise ValueError(f"User interests column '{USER_INTERESTS_COL}' not found!")
    if USER_WATCH_CATEGORIES_COL not in users_df.columns: raise ValueError(f"User watch categories column '{USER_WATCH_CATEGORIES_COL}' not found!")
    with tqdm(total=2, desc=f"[{func_name}] User Embeddings", ncols=100, leave=False) as pbar:
        users_df['interest_embed'] = get_sbert_batch_embeddings(users_df[USER_INTERESTS_COL].tolist(), sbert_model); pbar.update(1)
        users_df['recent_watch_embed'] = get_sbert_batch_embeddings(users_df[USER_WATCH_CATEGORIES_COL].tolist(), sbert_model); pbar.update(1)

    # Date Features
    datetime_format = '%Y-%m-%d'; logger.info(f"[{func_name}] Processing user date features ('{USER_CREATION_DATE_COL}')...")
    numerical_user_cols = []
    if USER_CREATION_DATE_COL in users_df.columns:
        try:
            users_df[USER_CREATION_DATE_COL] = users_df[USER_CREATION_DATE_COL].astype(str).str.strip().replace('', pd.NA) # Use NA for replace
            creation_times = pd.to_datetime(users_df[USER_CREATION_DATE_COL], errors='coerce')
            creation_times.fillna(pd.Timestamp.now(), inplace=True)
            users_df['creation_hour'] = creation_times.dt.hour
            users_df['creation_dayofweek'] = creation_times.dt.dayofweek
            numerical_user_cols = ['creation_hour', 'creation_dayofweek']
        except Exception as date_e: logger.error(f"[{func_name}] Error parsing '{USER_CREATION_DATE_COL}': {date_e}. Setting defaults."); users_df['creation_hour'] = 0; users_df['creation_dayofweek'] = 0; numerical_user_cols = ['creation_hour', 'creation_dayofweek']
    else: logger.warning(f"[{func_name}] '{USER_CREATION_DATE_COL}' not found."); users_df['creation_hour'] = 0; users_df['creation_dayofweek'] = 0; numerical_user_cols = ['creation_hour', 'creation_dayofweek']

    # User Encoders
    logger.info(f"[{func_name}] Encoding user categorical features...")
    encoders = { USER_AGE_COL: LabelEncoder(), USER_WEATHER_COL: LabelEncoder(), USER_ACTIVITY_COL: LabelEncoder(), USER_MONETARY_COL: LabelEncoder(), USER_LOCATION_TIER_COL: LabelEncoder(), USER_DEVICE_COL: LabelEncoder() }
    categorical_encoded_user_cols = []
    fitted_user_encoders = {}
    for col, encoder in encoders.items():
        if col in users_df.columns:
            encoded_col_name = f"{col}_encoded"
            try: users_df[encoded_col_name] = encoder.fit_transform(users_df[col]); categorical_encoded_user_cols.append(encoded_col_name); fitted_user_encoders[col] = encoder
            except Exception as enc_e: logger.error(f"[{func_name}] Failed to encode user '{col}': {enc_e}. Skipping.")
        else: logger.warning(f"[{func_name}] User column '{col}' missing for encoding.")

    # --- Campaign Preprocessing ---
    logger.info(f"[{func_name}] --- Processing Campaign Data ---")
    expected_campaign_cols = [CAMPAIGN_ID_COL, CAMPAIGN_PROMO_COL, CAMPAIGN_CATEGORY_COL, CAMPAIGN_BUDGET_COL, CAMPAIGN_LOCATION_COL, CAMPAIGN_BUSINESS_NAME_COL, CAMPAIGN_TARGET_GROUP_COL, CAMPAIGN_START_TIME_COL, CAMPAIGN_END_TIME_COL]
    campaign_defaults = { CAMPAIGN_PROMO_COL: 'general offer', CAMPAIGN_CATEGORY_COL: 'general', CAMPAIGN_BUDGET_COL: 0, CAMPAIGN_LOCATION_COL: 'unknown', CAMPAIGN_BUSINESS_NAME_COL: 'unknown', CAMPAIGN_TARGET_GROUP_COL: 'general', CAMPAIGN_START_TIME_COL: datetime.now().strftime('%Y-%m-%d %H:%M:%S'), CAMPAIGN_END_TIME_COL: (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')}
    missing_camp_cols_count = 0 # Handle missing cols and fillna
    for col in expected_campaign_cols:
        if col not in campaigns_df.columns: default_val = campaign_defaults.get(col, 'unknown'); logger.warning(f"[{func_name}] Campaign column '{col}' missing. Adding default '{default_val}'."); campaigns_df[col] = default_val; missing_camp_cols_count += 1
    if missing_camp_cols_count > 0: logger.info(f"[{func_name}] Added {missing_camp_cols_count} missing campaign columns.")
    campaigns_df.fillna(campaign_defaults, inplace=True)
    campaign_cols_to_standardize = [CAMPAIGN_PROMO_COL, CAMPAIGN_CATEGORY_COL, CAMPAIGN_LOCATION_COL, CAMPAIGN_BUSINESS_NAME_COL, CAMPAIGN_TARGET_GROUP_COL]
    logger.info(f"[{func_name}] Standardizing campaign text columns...")
    for col in campaign_cols_to_standardize:
        if col in campaigns_df.columns: campaigns_df[col] = campaigns_df[col].astype(str).str.lower().str.strip()
    logger.info(f"[{func_name}] Validating campaign IDs ('{CAMPAIGN_ID_COL}')...") # ID validation
    if CAMPAIGN_ID_COL not in campaigns_df.columns: logger.warning(f"[{func_name}] '{CAMPAIGN_ID_COL}' missing. Generating IDs."); campaigns_df[CAMPAIGN_ID_COL] = [f'cmp_gen_{i:05d}' for i in range(1, len(campaigns_df) + 1)]
    campaigns_df[CAMPAIGN_ID_COL] = campaigns_df[CAMPAIGN_ID_COL].astype(str)
    if campaigns_df[CAMPAIGN_ID_COL].duplicated().any(): logger.warning(f"[{func_name}] Duplicate campaign IDs found. Keeping first."); campaigns_df = campaigns_df.drop_duplicates(subset=[CAMPAIGN_ID_COL], keep='first').reset_index(drop=True); logger.info(f"[{func_name}] Campaigns shape after dedupe: {campaigns_df.shape}")
    if CAMPAIGN_LOCATION_COL not in campaigns_df.columns: raise ValueError(f"Campaign location column '{CAMPAIGN_LOCATION_COL}' not found!")
    logger.info(f"[{func_name}] Standardizing campaign locations ('{CAMPAIGN_LOCATION_COL}')...")
    campaigns_df['location_standardized'] = campaigns_df[CAMPAIGN_LOCATION_COL].progress_apply(standardize_location)
    logger.info(f"[{func_name}] Generating campaign SBERT embeddings (batch)...")
    if CAMPAIGN_PROMO_COL not in campaigns_df.columns: raise ValueError(f"Campaign promo column '{CAMPAIGN_PROMO_COL}' not found!")
    if CAMPAIGN_CATEGORY_COL not in campaigns_df.columns: raise ValueError(f"Campaign category column '{CAMPAIGN_CATEGORY_COL}' not found!")
    with tqdm(total=2, desc=f"[{func_name}] Campaign Embeddings", ncols=100, leave=False) as pbar:
        campaigns_df['promo_embed'] = get_sbert_batch_embeddings(campaigns_df[CAMPAIGN_PROMO_COL].tolist(), sbert_model); pbar.update(1)
        campaigns_df['category_embed'] = get_sbert_batch_embeddings(campaigns_df[CAMPAIGN_CATEGORY_COL].tolist(), sbert_model); pbar.update(1)

    # Temporal Features
    campaign_temporal_num_cols = []; scaler_campaign_temporal = None
       # --- Campaign Preprocessing ---
    # ... (previous campaign processing steps) ...

    # Temporal Features
    campaign_temporal_num_cols = []; scaler_campaign_temporal = None
    if CAMPAIGN_START_TIME_COL in campaigns_df.columns and CAMPAIGN_END_TIME_COL in campaigns_df.columns:
        # --- CORRECTED try...except block ---
        try: # Start try block
            # Indent the code that might fail under the try
            logger.info(f"[{func_name}] Processing campaign temporal features...")
            start_times = pd.to_datetime(campaigns_df[CAMPAIGN_START_TIME_COL], errors='coerce').fillna(pd.Timestamp.now())
            # Handle potential NaT in start_times before adding timedelta
            end_times_calculated = start_times + timedelta(days=30)
            end_times_from_col = pd.to_datetime(campaigns_df[CAMPAIGN_END_TIME_COL], errors='coerce')
            # Use COALESCE logic: Use end_times_from_col if valid, else use calculated fallback
            end_times = end_times_from_col.fillna(end_times_calculated)
            # Ensure end_time is at least one day after start_time
            end_times = np.maximum(start_times + timedelta(days=1), end_times)

            campaigns_df['campaign_duration_days']=(end_times - start_times).dt.days
            campaigns_df['start_dayofweek']=start_times.dt.dayofweek
            campaigns_df['start_month']=start_times.dt.month
            campaigns_df['end_dayofweek']=end_times.dt.dayofweek
            campaigns_df['end_month']=end_times.dt.month
            campaign_temporal_num_cols=['campaign_duration_days', 'start_dayofweek', 'start_month', 'end_dayofweek', 'end_month']

            # Ensure columns are numeric and handle potential NaNs introduced by date diffs
            for col in campaign_temporal_num_cols:
                campaigns_df[col]=pd.to_numeric(campaigns_df[col], errors='coerce').fillna(0).astype(int)

            scaler_campaign_temporal=StandardScaler()
            campaigns_df[campaign_temporal_num_cols]=scaler_campaign_temporal.fit_transform(campaigns_df[campaign_temporal_num_cols])
            logger.info(f"[{func_name}] Temporal features created and scaled.")

        except Exception as temp_e: # Add the except block aligned with try
            # Indent the code inside the except block
            logger.error(f"[{func_name}] Error processing temporal features: {temp_e}. Skipping temporal features.")
            # Reset variables if an error occurred
            campaign_temporal_num_cols = []
            scaler_campaign_temporal = None
            # Remove potentially partially created columns if error occurred mid-way
            cols_to_drop_on_error = ['campaign_duration_days', 'start_dayofweek', 'start_month', 'end_dayofweek', 'end_month']
            for col in cols_to_drop_on_error:
                 if col in campaigns_df.columns:
                     campaigns_df = campaigns_df.drop(columns=[col])
        # --- End CORRECTED try...except block ---

    else: # This else corresponds to the initial if check for column existence
        logger.warning(f"[{func_name}] Temporal columns ('{CAMPAIGN_START_TIME_COL}', '{CAMPAIGN_END_TIME_COL}') missing. Skipping temporal features.")
        # Ensure variables are defined even if columns missing
        campaign_temporal_num_cols = []
        scaler_campaign_temporal = None

    # --- Budget Scaling ---
    # ... (rest of the preprocess_data_gnn function) ...
    # Budget Scaling
    if CAMPAIGN_BUDGET_COL in campaigns_df.columns: logger.info(f"[{func_name}] Scaling campaign budget ('{CAMPAIGN_BUDGET_COL}')...")
    else: logger.warning(f"[{func_name}] Campaign budget column '{CAMPAIGN_BUDGET_COL}' missing.")
    campaigns_df[CAMPAIGN_BUDGET_COL]=pd.to_numeric(campaigns_df.get(CAMPAIGN_BUDGET_COL, 0), errors='coerce').fillna(0); scaler_campaign_budget=StandardScaler(); campaigns_df['budget_scaled']=scaler_campaign_budget.fit_transform(campaigns_df[[CAMPAIGN_BUDGET_COL]])
    # Campaign Encoders
    logger.info(f"[{func_name}] Encoding campaign categorical features...")
    target_encoder=LabelEncoder(); encoded_target_col=f"{CAMPAIGN_TARGET_GROUP_COL}_encoded"; categorical_encoded_campaign_cols_extra = []
    if CAMPAIGN_TARGET_GROUP_COL in campaigns_df.columns: campaigns_df[encoded_target_col]=target_encoder.fit_transform(campaigns_df[CAMPAIGN_TARGET_GROUP_COL]); categorical_encoded_campaign_cols_extra.append(encoded_target_col)
    else: logger.warning(f"[{func_name}] Campaign target group column '{CAMPAIGN_TARGET_GROUP_COL}' missing."); target_encoder = None
    # Simulate device targeting
    campaign_device_target_col = CAMPAIGN_TARGET_DEVICE_COL; campaign_device_target_encoded_col = f'{campaign_device_target_col}_encoded'; device_targeting_fraction = 0.05; num_device_targets = int(len(campaigns_df) * device_targeting_fraction)
    campaigns_df[campaign_device_target_col] = 'all'; target_indices = np.random.choice(campaigns_df.index, size=num_device_targets, replace=False); possible_targets = [d for d in DEVICE_TYPES if d != 'unknown']; random_targets = np.random.choice(possible_targets, size=num_device_targets, replace=True); campaigns_df.loc[target_indices, campaign_device_target_col] = random_targets
    campaign_device_encoder = LabelEncoder(); all_possible_campaign_devices = ['all'] + DEVICE_TYPES; campaign_device_encoder.fit(all_possible_campaign_devices); campaigns_df[campaign_device_target_encoded_col] = campaign_device_encoder.transform(campaigns_df[campaign_device_target_col]); categorical_encoded_campaign_cols_extra.append(campaign_device_target_encoded_col)

    # --- Unified Location Encoding ---
    logger.info(f"[{func_name}] Performing unified location encoding...")
    location_encoder = LabelEncoder(); all_locations = pd.concat([users_df['location_standardized'], campaigns_df['location_standardized']], ignore_index=True).astype(str).unique()
    location_encoder.fit(all_locations); users_df['location_encoded'] = location_encoder.transform(users_df['location_standardized']); campaigns_df['location_encoded'] = location_encoder.transform(campaigns_df['location_standardized'])
    logger.info(f"[{func_name}] Unified Location Encoder Classes: {len(location_encoder.classes_)}")
    categorical_encoded_user_cols.append('location_encoded'); categorical_encoded_campaign_cols = ['location_encoded'] + categorical_encoded_campaign_cols_extra

    # --- Combine Features ---
    logger.info(f"[{func_name}] Combining final feature vectors...")
    # (Using robust feature combination logic)
    user_features_num = users_df[numerical_user_cols].values if numerical_user_cols else np.zeros((len(users_df), 0))
    user_features_cat = users_df[[col for col in categorical_encoded_user_cols if col in users_df.columns]].values
    user_embed_interests = np.vstack(users_df['interest_embed'].values)
    user_embed_watch = np.vstack(users_df['recent_watch_embed'].values)
    user_features_list = [user_embed_interests, user_embed_watch, user_features_num, user_features_cat]
    user_features = np.hstack([feat for feat in user_features_list if feat.ndim == 2 and feat.shape[1] > 0])
    user_feature_dim = user_features.shape[1] if user_features.size > 0 else 0

    campaign_features_temporal = campaigns_df[campaign_temporal_num_cols].values if campaign_temporal_num_cols else np.zeros((len(campaigns_df), 0))
    campaign_features_cat = campaigns_df[[col for col in categorical_encoded_campaign_cols if col in campaigns_df.columns]].values
    campaign_embed_promo = np.vstack(campaigns_df['promo_embed'].values)
    campaign_embed_cat = np.vstack(campaigns_df['category_embed'].values)
    campaign_budget_scaled = campaigns_df[['budget_scaled']].values
    campaign_features_list = [campaign_embed_promo, campaign_embed_cat, campaign_budget_scaled, campaign_features_temporal, campaign_features_cat]
    campaign_features = np.hstack([feat for feat in campaign_features_list if feat.ndim == 2 and feat.shape[1] > 0])
    campaign_feature_dim = campaign_features.shape[1] if campaign_features.size > 0 else 0
    logger.debug(f"[{func_name}] User feature dim: {user_feature_dim}, Campaign feature dim: {campaign_feature_dim}")
    if user_feature_dim == 0 or campaign_feature_dim == 0: raise ValueError("Feature generation resulted in empty arrays.")

    # Padding
    max_dim = max(user_feature_dim, campaign_feature_dim); final_node_feature_dim = max_dim
    logger.info(f"[{func_name}] Padding features to dimension: {final_node_feature_dim}")
    if user_features.size == 0: user_features = np.zeros((len(users_df), final_node_feature_dim))
    elif user_feature_dim < max_dim: user_features = np.hstack([user_features, np.zeros((user_features.shape[0], max_dim - user_feature_dim), dtype=user_features.dtype)])
    if campaign_features.size == 0: campaign_features = np.zeros((len(campaigns_df), final_node_feature_dim))
    elif campaign_feature_dim < max_dim: campaign_features = np.hstack([campaign_features, np.zeros((campaign_features.shape[0], max_dim - campaign_feature_dim), dtype=campaign_features.dtype)])

    # --- Mappings & Final Features (Robust Alignment) ---
    logger.info(f"[{func_name}] Creating final node mappings and aligning features...")
    unique_user_ids=users_df[USER_ID_COL].astype(str).unique(); unique_campaign_ids=campaigns_df[CAMPAIGN_ID_COL].astype(str).unique()
    user_map={orig_id: i for i, orig_id in enumerate(unique_user_ids)}; campaign_map={orig_id: i + len(user_map) for i, orig_id in enumerate(unique_campaign_ids)}
    reverse_user_map={v: k for k, v in user_map.items()}; reverse_campaign_map={v: k for k, v in campaign_map.items()}; num_users=len(user_map); num_campaigns=len(campaign_map); num_nodes=num_users + num_campaigns
    logger.info(f"[{func_name}] Final counts - Users={num_users}, Campaigns={num_campaigns}, Total Nodes={num_nodes}")

    aligned_user_features = np.zeros((num_users, final_node_feature_dim), dtype=user_features.dtype)
    user_reindexer = pd.Series(range(len(users_df)), index=users_df[USER_ID_COL])
    target_user_indices = user_reindexer.reindex(unique_user_ids).values
    valid_user_mask = ~np.isnan(target_user_indices); valid_indices = target_user_indices[valid_user_mask].astype(int); valid_mask_indices = np.where(valid_user_mask)[0]
    if len(valid_indices) == len(valid_mask_indices): aligned_user_features[valid_mask_indices] = user_features[valid_indices]
    else: logger.error("Mismatch user mask lengths!"); raise ValueError("User alignment failed")

    aligned_campaign_features = np.zeros((num_campaigns, final_node_feature_dim), dtype=campaign_features.dtype)
    campaign_reindexer = pd.Series(range(len(campaigns_df)), index=campaigns_df[CAMPAIGN_ID_COL])
    target_campaign_indices = campaign_reindexer.reindex(unique_campaign_ids).values
    valid_campaign_mask = ~np.isnan(target_campaign_indices); valid_indices_camp = target_campaign_indices[valid_campaign_mask].astype(int); valid_mask_indices_camp = np.where(valid_campaign_mask)[0]
    if len(valid_indices_camp) == len(valid_mask_indices_camp): aligned_campaign_features[valid_mask_indices_camp] = campaign_features[valid_indices_camp]
    else: logger.error("Mismatch campaign mask lengths!"); raise ValueError("Campaign alignment failed")

    node_features=np.vstack((aligned_user_features, aligned_campaign_features)); node_features_tensor=torch.tensor(node_features, dtype=torch.float)
    logger.info(f"[{func_name}] Final node features tensor shape: {node_features_tensor.shape}")
    assert node_features_tensor.shape[0] == num_nodes, f"Row count mismatch"; assert node_features_tensor.shape[1] == final_node_feature_dim, f"Column count mismatch"

    # --- Store Artifacts ---
    logger.info(f"[{func_name}] Storing preprocessing artifacts...")
    artifacts = {
        'user_map': user_map, 'campaign_map': campaign_map, 'reverse_user_map': reverse_user_map, 'reverse_campaign_map': reverse_campaign_map,
        f'{USER_AGE_COL}_encoder': fitted_user_encoders.get(USER_AGE_COL), f'{USER_WEATHER_COL}_encoder': fitted_user_encoders.get(USER_WEATHER_COL), f'{USER_ACTIVITY_COL}_encoder': fitted_user_encoders.get(USER_ACTIVITY_COL), f'{USER_MONETARY_COL}_encoder': fitted_user_encoders.get(USER_MONETARY_COL), 'location_encoder': location_encoder, f'{USER_LOCATION_TIER_COL}_encoder': fitted_user_encoders.get(USER_LOCATION_TIER_COL), f'{USER_DEVICE_COL}_encoder': fitted_user_encoders.get(USER_DEVICE_COL),
        f'{CAMPAIGN_TARGET_GROUP_COL}_encoder': target_encoder, 'campaign_device_encoder': campaign_device_encoder,
        'scaler_campaign_budget': scaler_campaign_budget, 'scaler_campaign_temporal': scaler_campaign_temporal,
        'num_users': num_users, 'num_campaigns': num_campaigns, 'node_feature_dim': final_node_feature_dim, 'standardized_locations': list(location_encoder.classes_) }
    os.makedirs(PREPROCESSED_DIR, exist_ok=True); campaigns_df_preprocessed_path = os.path.join(PREPROCESSED_DIR, CAMPAIGNS_PREPROCESSED_FILE)
    cols_to_save = [CAMPAIGN_ID_COL, CAMPAIGN_BUSINESS_NAME_COL, CAMPAIGN_PROMO_COL, CAMPAIGN_CATEGORY_COL, 'location_standardized', campaign_device_target_col]
    campaigns_df_to_save = campaigns_df[[col for col in cols_to_save if col in campaigns_df.columns]].copy()
    campaigns_df_to_save.rename(columns={'location_standardized': 'location'}, inplace=True)
    campaigns_df_to_save.to_csv(campaigns_df_preprocessed_path, index=False)
    logger.info(f"[{func_name}] Preprocessed campaign info saved.")
    logger.info(f"[{func_name}] Preprocessing finished. Zero embeddings: {zero_embedding_count}.")
    return users_df, campaigns_df, node_features_tensor, artifacts

# --- Interaction / Edge Creation (With fixed all_device_encoded_val init) ---
def create_interaction_edges(users_df, campaigns_df, user_map, campaign_map, artifacts):
    func_name="create_interaction_edges"; logger.info(f"[{func_name}] Starting...")
    required_user_cols_edge=['interest_embed', 'recent_watch_embed', 'location_encoded', f'{USER_DEVICE_COL}_encoded', USER_ID_COL]
    required_camp_cols_edge=['promo_embed', 'category_embed', 'location_encoded', 'target_device_encoded', CAMPAIGN_ID_COL]
    validate_schema(users_df, required_user_cols_edge, "Users DF for Edge Creation", func_name)
    validate_schema(campaigns_df, required_camp_cols_edge, "Campaigns DF for Edge Creation", func_name)
    num_users=artifacts.get('num_users'); num_campaigns=artifacts.get('num_campaigns');
    if num_users is None or num_campaigns is None: raise ValueError(f"[{func_name}] User/Campaign count missing.")
    try:
        logger.info(f"[{func_name}] Extracting & normalizing embeddings...")
        try: user_interest_embeddings=np.vstack(users_df['interest_embed'].values); campaign_promo_embeddings=np.vstack(campaigns_df['promo_embed'].values); user_recent_embeddings=np.vstack(users_df['recent_watch_embed'].values); campaign_category_embeddings=np.vstack(campaigns_df['category_embed'].values);
        except ValueError as stack_e: logger.error(f"[{func_name}] Error stacking embeddings: {stack_e}"); raise
        user_interest_norm=normalize(user_interest_embeddings); campaign_promo_norm=normalize(campaign_promo_embeddings); user_recent_norm=normalize(user_recent_embeddings); campaign_category_norm=normalize(campaign_category_embeddings)
        logger.info(f"[{func_name}] Calculating similarities...")
        interest_promo_sim=cosine_similarity(user_interest_norm, campaign_promo_norm); recent_category_sim=cosine_similarity(user_recent_norm, campaign_category_norm);
        user_loc_encoded=users_df['location_encoded'].values; camp_loc_encoded=campaigns_df['location_encoded'].values; location_match=(user_loc_encoded[:, np.newaxis]==camp_loc_encoded[np.newaxis, :]).astype(float)
        logger.info(f"[{func_name}] Calculating device match..."); user_device_encoded=users_df[f'{USER_DEVICE_COL}_encoded'].values; camp_device_encoded=campaigns_df['target_device_encoded'].values; campaign_device_encoder=artifacts.get('campaign_device_encoder');
        if campaign_device_encoder is None: raise ValueError("Campaign device encoder missing.");
        # --- FIX: Initialize BEFORE the try block ---
        all_device_encoded_val = -99 # Default non-matching value
        try: all_device_encoded_val = list(campaign_device_encoder.classes_).index('all'); logger.debug(f"Encoded value for 'all' devices found: {all_device_encoded_val}")
        except ValueError: logger.warning("Could not find 'all' in campaign device encoder classes. Using default non-match value.")
        # --- End FIX ---
        device_match=((camp_device_encoded[np.newaxis, :]==all_device_encoded_val)|(user_device_encoded[:, np.newaxis]==camp_device_encoded[np.newaxis, :])).astype(float)
        location_weight=2.0; interest_weight=1.0; watch_weight=1.0; device_weight=0.75; logger.info(f"[{func_name}] Calculating weighted scores (Loc:{location_weight}, Dev:{device_weight})...");
        interaction_scores=(interest_weight*interest_promo_sim+watch_weight*recent_category_sim+location_weight*location_match+device_weight*device_match)
        threshold_percentile=85; fallback_threshold=0.1; positive_scores=interaction_scores[interaction_scores > 1e-6]; threshold=max(np.percentile(positive_scores, threshold_percentile), fallback_threshold) if positive_scores.size > 0 else fallback_threshold; logger.info(f"[{func_name}] Interaction threshold: {threshold:.4f}")
        positive_indices=np.where(interaction_scores > threshold); user_df_indices=positive_indices[0]; campaign_df_indices=positive_indices[1]; logger.info(f"[{func_name}] Found {len(user_df_indices)} potential interactions.")
        if len(user_df_indices) == 0: logger.warning(f"[{func_name}] No interactions found above threshold."); return torch.empty((2, 0), dtype=torch.long)
        logger.info(f"[{func_name}] Mapping heuristic edges..."); source_nodes=[]; target_nodes=[]; skipped_heur_edges=0; existing_edges_set=set(); user_ids_at_indices=users_df[USER_ID_COL].iloc[user_df_indices].values; campaign_ids_at_indices=campaigns_df[CAMPAIGN_ID_COL].iloc[campaign_df_indices].values
        for u_orig_id, c_orig_id in tqdm(zip(user_ids_at_indices, campaign_ids_at_indices), total=len(user_ids_at_indices), desc=f"[{func_name}] Mapping Edges", ncols=100, leave=False):
             user_node_idx=user_map.get(str(u_orig_id)); campaign_node_idx=campaign_map.get(str(c_orig_id));
             if user_node_idx is None or campaign_node_idx is None: skipped_heur_edges+=1; continue
             edge_tuple=(user_node_idx, campaign_node_idx);
             if edge_tuple not in existing_edges_set: source_nodes.append(user_node_idx); target_nodes.append(campaign_node_idx); existing_edges_set.add(edge_tuple)
        if skipped_heur_edges > 0: logger.warning(f"[{func_name}] Skipped {skipped_heur_edges} edges."); logger.info(f"[{func_name}] Created {len(source_nodes)} edges from heuristics.")
        logger.info(f"[{func_name}] Checking/connecting isolated campaigns..."); campaign_node_indices_set=set(target_nodes); all_campaign_node_ids=set(campaign_map.values()); isolated_campaign_nodes=list(all_campaign_node_ids - campaign_node_indices_set); num_isolated_connected=0; CONNECT_TOP_N_USERS=max(2, num_users // 1000 if num_users > 0 else 2)
        if isolated_campaign_nodes: # Isolate connection logic
             logger.warning(f"[{func_name}] Found {len(isolated_campaign_nodes)} isolated campaigns. Connecting."); interest_category_sim=cosine_similarity(user_interest_norm, campaign_category_norm); campaign_id_to_df_idx={cid: idx for idx, cid in enumerate(campaigns_df[CAMPAIGN_ID_COL])}; campaign_node_to_df_idx={node_id: campaign_id_to_df_idx.get(artifacts['reverse_campaign_map'].get(node_id)) for node_id in isolated_campaign_nodes if artifacts['reverse_campaign_map'].get(node_id) in campaign_id_to_df_idx}
             for camp_node_id in tqdm(isolated_campaign_nodes, desc=f"[{func_name}] Connecting Isolates", ncols=100, leave=False):
                  camp_df_idx=campaign_node_to_df_idx.get(camp_node_id);
                  if camp_df_idx is None or camp_df_idx >= interest_category_sim.shape[1]: continue
                  try: sim_scores_for_camp=interest_category_sim[:, camp_df_idx]; num_users_to_consider=min(CONNECT_TOP_N_USERS, len(users_df)); top_user_df_indices=np.argsort(sim_scores_for_camp)[-num_users_to_consider:][::-1]
                  except IndexError as idx_e: logger.warning(f"Index error sim calc iso {camp_df_idx}: {idx_e}"); continue
                  for user_df_idx in top_user_df_indices:
                      user_orig_id=users_df.iloc[user_df_idx][USER_ID_COL]; user_node_idx=user_map.get(str(user_orig_id));
                      if user_node_idx is not None: edge_tuple=(user_node_idx, camp_node_id);
                      if edge_tuple not in existing_edges_set: source_nodes.append(user_node_idx); target_nodes.append(camp_node_id); existing_edges_set.add(edge_tuple); num_isolated_connected+=1
             logger.info(f"[{func_name}] Added {num_isolated_connected} edges for isolated campaigns.")
        else: logger.info(f"[{func_name}] No isolated campaigns found.")
        if not source_nodes: logger.warning(f"[{func_name}] No edges created!"); return torch.empty((2, 0), dtype=torch.long)
        edge_index=torch.tensor([source_nodes, target_nodes], dtype=torch.long); logger.info(f"[{func_name}] Final edge count: {edge_index.shape[1]}")
        if edge_index.numel() > 0: # Validate edge indices
            max_idx = edge_index.max().item(); num_nodes_total = len(user_map) + len(campaign_map);
            if max_idx >= num_nodes_total: raise ValueError(f"[{func_name}] Edge index max ({max_idx}) >= num_nodes ({num_nodes_total})")
        logger.info(f"[{func_name}] Edge creation finished.")
        return edge_index
    except Exception as e: logger.error(f"[{func_name}] Error: {e}", exc_info=True); clear_cuda_cache("edge creation error"); raise

# --- GNN Model Definitions ---
# (Keep GraphSAGE and LinkPredictor)
class GraphSAGE_GNN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5): super().__init__(); self.conv1 = SAGEConv(in_channels, hidden_channels); self.conv2 = SAGEConv(hidden_channels, out_channels); self.dropout = Dropout(dropout); logger.info(f"Init GraphSAGE_GNN: In={in_channels}, Hid={hidden_channels}, Out={out_channels}, Drop={dropout}")
    def forward(self, x, edge_index): x = F.relu(self.conv1(x, edge_index)); x = self.dropout(x); x = self.conv2(x, edge_index); return x
class LinkPredictor(Module):
    def __init__(self, in_channels): super().__init__(); predictor_hidden_dim = max(32, in_channels // 2); self.lin1 = Linear(in_channels * 2, predictor_hidden_dim); self.lin2 = Linear(predictor_hidden_dim, 1); self.dropout = Dropout(0.3); logger.info(f"Init Predictor: In={in_channels*2}, Hid={predictor_hidden_dim}, Out=1, Drop={0.3}")
    def forward(self, x_i, x_j): x = torch.cat([x_i, x_j], dim=-1); x = F.relu(self.lin1(x)); x = self.dropout(x); x = self.lin2(x); return x.squeeze(-1)


# --- Evaluation Function ---
# (Keep test_model function with robust checks)
# Make sure this function definition is at the correct indentation level
# (usually top-level, not indented under another function or class)

@torch.no_grad()
def test_model(data, model, predictor, artifacts, device):
    """Evaluates the model on the test set (AUC, P@K, NDCG@K) with robust checks."""
    func_name = "test_model"; logger.info(f"[{func_name}] Starting final evaluation...")
    model.eval(); predictor.eval(); message_passing_edges = torch.empty((2,0), dtype=torch.long, device=device)

    # --- Get Edges for Message Passing ---
    # Prefer train edges to avoid info leakage from test set into embeddings
    if hasattr(data, 'train_pos_edge_index') and data.train_pos_edge_index is not None and data.train_pos_edge_index.numel() > 0:
        message_passing_edges = data.train_pos_edge_index.to(device)
        logger.info(f"[{func_name}] Using train_pos_edge_index for message passing.")
    elif hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
         logger.warning(f"[{func_name}] train_pos_edge_index not found, using full edge_index for message passing (potential minor leakage).")
         message_passing_edges = data.edge_index.to(device)
    else:
        logger.error(f"[{func_name}] No suitable edges found for message passing during testing.")
        return 0.0, 0.0, 0.0 # Return defaults if no edges

    # --- Generate Embeddings ---
    try:
        x_feat = data.x.to(device);
        if not x_feat.is_contiguous(): x_feat = x_feat.contiguous()
        if not message_passing_edges.is_contiguous(): message_passing_edges = message_passing_edges.contiguous()
        h = model(x_feat, message_passing_edges)
        logger.info(f"[{func_name}] Generated test node embeddings shape: {h.shape}")
    except RuntimeError as rt_e:
         logger.error(f"[{func_name}] Runtime error generating test embeddings: {rt_e}")
         clear_cuda_cache("test embed error - runtime")
         return 0.0, 0.0, 0.0
    except Exception as e:
        logger.error(f"[{func_name}] General error generating test embeddings: {e}")
        clear_cuda_cache("test embed error - general")
        return 0.0, 0.0, 0.0 # Return defaults if embedding fails

    # --- Get Test Edges ---
    pos_edge_index=getattr(data, 'test_pos_edge_index', torch.empty((2,0))).to(device)
    neg_edge_index=getattr(data, 'test_neg_edge_index', torch.empty((2,0))).to(device)

    # Check if there are any positive test edges
    if pos_edge_index.numel() == 0:
        logger.warning(f"[{func_name}] No positive test edges found to evaluate.")
        del h, x_feat, message_passing_edges, pos_edge_index, neg_edge_index # Cleanup
        clear_cuda_cache(f"end {func_name} - no pos test edges")
        return 0.0, 0.0, 0.0 # Return defaults if no positive edges

    # --- Check Index Bounds ---
    max_test_idx = pos_edge_index.max().item()
    if neg_edge_index.numel() > 0: max_test_idx = max(max_test_idx, neg_edge_index.max().item())
    if max_test_idx >= h.shape[0]:
        logger.error(f"[{func_name}] Max test edge index {max_test_idx} >= embedding dim {h.shape[0]}. Aborting test.")
        del h, x_feat, message_passing_edges, pos_edge_index, neg_edge_index # Cleanup
        clear_cuda_cache(f"end {func_name} - index OOB")
        return 0.0, 0.0, 0.0

    # --- Calculate Predictions ---
    try:
        pos_out = predictor(h[pos_edge_index[0]], h[pos_edge_index[1]])
        neg_out = predictor(h[neg_edge_index[0]], h[neg_edge_index[1]]) if neg_edge_index.numel() > 0 else torch.empty(0, device=device)
    except IndexError as idx_e:
         logger.error(f"[{func_name}] IndexError during prediction: {idx_e}. h shape: {h.shape}")
         del h, x_feat, message_passing_edges, pos_edge_index, neg_edge_index # Cleanup
         clear_cuda_cache("test prediction index error")
         return 0.0, 0.0, 0.0 # Cannot calculate scores
    except Exception as pred_e:
         logger.error(f"[{func_name}] Error during prediction calculation: {pred_e}")
         del h, x_feat, message_passing_edges, pos_edge_index, neg_edge_index # Cleanup
         clear_cuda_cache("test prediction error")
         return 0.0, 0.0, 0.0

    # --- AUC Calculation ---
    auc = 0.0
    out = None # Initialize to prevent reference before assignment if first block skipped
    y = None
    if pos_out.numel() > 0 or neg_out.numel() > 0:
        out = torch.cat([pos_out, neg_out], dim=0)
        y = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)], dim=0)

        # --- CORRECTED try...except for AUC ---
        try: # Start try block
             # Indent code under try
             y_np=y.cpu().numpy()
             out_sigmoid_np=torch.sigmoid(out).cpu().numpy()
             if len(np.unique(y_np)) > 1 and np.all(np.isfinite(out_sigmoid_np)):
                  auc=roc_auc_score(y_np, out_sigmoid_np)
             # else: auc remains 0.0
        except Exception as e: # Add except block aligned with try
             # Indent code under except
             logger.warning(f"[{func_name} - AUC Calc] Error: {e}")
             auc = 0.0 # Explicitly set to 0 on error
        # --- End Correction ---

    # --- P@K, NDCG@K Calculation ---
    k = 10; avg_precision_at_k = 0.0; avg_ndcg_at_k = 0.0;
    num_users = artifacts.get('num_users', 0); num_campaigns = artifacts.get('num_campaigns', 0);
    # Ensure node indices are valid before attempting P@K/NDCG@K
    if num_users > 0 and num_campaigns > 0 and h.shape[0] >= (num_users + num_campaigns):
        user_indices_test=pos_edge_index[0].unique(); precision_list=[]; ndcg_list=[];
        all_campaign_indices=torch.arange(num_users, num_users + num_campaigns, device=device)

        # Check if campaign indices are valid before trying to access embeddings
        if all_campaign_indices.numel() > 0 and all_campaign_indices.max().item() < h.shape[0]:
             all_campaign_embeddings = h[all_campaign_indices]
             logger.info(f"[{func_name}] Calculating P@K/NDCG@K for {len(user_indices_test)} test users...")
             for user_node_idx_tensor in tqdm(user_indices_test, desc=f"[{func_name}] User Eval", leave=False, ncols=100):
                 user_node_idx=user_node_idx_tensor.item();
                 if user_node_idx >= h.shape[0]: logger.warning(f"User index {user_node_idx} OOB for h"); continue

                 user_embedding=h[user_node_idx].unsqueeze(0);
                 user_embedding_repeated=user_embedding.repeat(len(all_campaign_indices), 1)

                 try:
                     all_scores=predictor(user_embedding_repeated, all_campaign_embeddings);
                     all_scores_sigmoid=torch.sigmoid(all_scores)
                 except Exception as pred_err:
                     logger.warning(f"Error predicting scores for P@K/NDCG user {user_node_idx}: {pred_err}"); continue

                 num_to_rank=len(all_campaign_indices);
                 actual_pos_mask=(pos_edge_index[0]==user_node_idx_tensor);
                 actual_pos_campaign_indices=pos_edge_index[1][actual_pos_mask];
                 actual_pos_set=set(actual_pos_campaign_indices.cpu().numpy())

                 # Precision@K
                 num_topk=min(k, num_to_rank);
                 if num_topk > 0:
                      _, top_k_relative_indices=torch.topk(all_scores_sigmoid, num_topk);
                      top_k_campaign_indices=all_campaign_indices[top_k_relative_indices]
                      hits=sum(1 for pred_idx in top_k_campaign_indices.cpu().numpy() if pred_idx in actual_pos_set);
                      precision_list.append(hits/k)
                 else:
                      precision_list.append(0.0)


                 # NDCG@K
                 true_relevance=np.zeros(num_to_rank);
                 # Get relative indices of true positives within the 'all_campaign_indices' tensor
                 all_campaign_indices_np = all_campaign_indices.cpu().numpy()
                 relative_indices_pos = [idx for idx, node_id in enumerate(all_campaign_indices_np) if node_id in actual_pos_set];
                 if relative_indices_pos: true_relevance[relative_indices_pos]=1.0 # Assign relevance score (binary 1.0)

                 if np.sum(true_relevance) > 0: # Only calculate if there are true positives
                     predicted_scores_np=all_scores_sigmoid.cpu().numpy()
                     if np.all(np.isfinite(predicted_scores_np)): # Check scores are finite
                          try:
                              # Ensure inputs have the right shape: (n_samples, n_labels) -> (1, num_to_rank)
                              ndcg_val=ndcg_score([true_relevance], [predicted_scores_np], k=k)
                              ndcg_list.append(ndcg_val)
                          except ValueError as ndcg_ve:
                              logger.warning(f"NDCG ValueError for user {user_node_idx}: {ndcg_ve}")
                          except Exception as ndcg_e:
                              logger.warning(f"NDCG calc failed user {user_node_idx}: {ndcg_e}")
                     else:
                          logger.warning(f"Skipping NDCG for user {user_node_idx} due to non-finite scores.")

             # Calculate Averages
             if precision_list: avg_precision_at_k = np.mean(precision_list)
             if ndcg_list: avg_ndcg_at_k = np.mean(ndcg_list)

             del all_campaign_embeddings # Cleanup inside condition
        else:
            logger.warning(f"[{func_name}] Cannot calculate P@K/NDCG@K: Invalid campaign indices range or count.")

        # Cleanup outside condition
        if 'all_campaign_indices' in locals(): del all_campaign_indices


    # Final Cleanup
    del h, x_feat, message_passing_edges, pos_edge_index, neg_edge_index, pos_out, neg_out
    # Delete combined tensors if they were created
    if out is not None: del out
    if y is not None: del y
    clear_cuda_cache(f"end {func_name}")

    logger.info(f"[{func_name}] Evaluation finished: AUC={auc:.4f}, P@10={avg_precision_at_k:.4f}, NDCG@10={avg_ndcg_at_k:.4f}")
    return auc, avg_precision_at_k, avg_ndcg_at_k

# --- Training Function ---
# (Keep robust train_gnn_model function with LR Scheduling and corrected sub-functions)
def train_gnn_model(data, artifacts, n_epochs=150, lr=0.003, hidden_channels=128, out_channels=64, dropout=0.4, weight_decay=1e-5, model_type='SAGE'):
    func_name="train_gnn_model"; logger.info(f"[{func_name}] Starting Training (Model: {model_type})..."); logger.info("Splitting edges...")
    try: # Edge splitting
        if not hasattr(data, 'edge_index') or data.edge_index is None: raise ValueError("Edge index missing.");
        with np.testing.suppress_warnings() as sup: sup.filter(UserWarning); data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1);
        train_count=getattr(data,'train_pos_edge_index',torch.empty((2,0))).shape[1]; val_count=getattr(data,'val_pos_edge_index',torch.empty((2,0))).shape[1]; test_count=getattr(data,'test_pos_edge_index',torch.empty((2,0))).shape[1]
        logger.info(f"[{func_name}] Edges: Train={train_count}, Val={val_count}, Test={test_count}")
        if val_count==0 or test_count==0: logger.warning(f"[{func_name}] Val or Test set has 0 pos edges!")
        if train_count==0: logger.error(f"[{func_name}] Train set has 0 pos edges! Stop."); return None, None, artifacts, data
    except Exception as e: logger.error(f"[{func_name}] Edge split error: {e}.", exc_info=True); raise
    logger.info(f"[{func_name}] Setting up model, predictor, optimizer, scheduler...") # Model setup
    node_feature_dim = artifacts.get('node_feature_dim');
    if node_feature_dim is None: raise ValueError(f"[{func_name}] node_feature_dim missing.")
    try:
        if model_type.upper()=='SAGE': model = GraphSAGE_GNN(node_feature_dim, hidden_channels, out_channels, dropout).to(DEVICE)
        else: raise ValueError(f"Unsupported model_type: {model_type}.")
        predictor = LinkPredictor(out_channels).to(DEVICE)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
        criterion = BCEWithLogitsLoss(); logger.info(f"[{func_name}] Model setup complete.")
    except Exception as e: logger.error(f"[{func_name}] Model setup error: {e}", exc_info=True); raise

    # --- Sub-Functions (train_epoch, validate) ---
    def train_epoch(current_epoch):
        model.train(); predictor.train(); optimizer.zero_grad(); total_loss = 0;
        train_pos_edge_index = getattr(data, 'train_pos_edge_index', None)
        if train_pos_edge_index is None or train_pos_edge_index.numel() == 0: return 0.0, 0.0
        try:
            train_pos_edge_index = train_pos_edge_index.to(DEVICE); x_feat = data.x.to(DEVICE);
            if not x_feat.is_contiguous(): x_feat = x_feat.contiguous()
            if not train_pos_edge_index.is_contiguous(): train_pos_edge_index = train_pos_edge_index.contiguous()
            h = model(x_feat, train_pos_edge_index)
            if train_pos_edge_index.max().item() >= h.shape[0]: raise IndexError(f"Pos index {train_pos_edge_index.max().item()} OOB for h {h.shape[0]}")
            pos_out = predictor(h[train_pos_edge_index[0]], h[train_pos_edge_index[1]])
            num_neg_samples = train_pos_edge_index.shape[1]; neg_edge_index = negative_sampling(edge_index=train_pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=num_neg_samples).to(DEVICE)
            if neg_edge_index.numel() > 0 and neg_edge_index.max().item() >= h.shape[0]: raise IndexError(f"Neg index {neg_edge_index.max().item()} OOB for h {h.shape[0]}")
            neg_out = predictor(h[neg_edge_index[0]], h[neg_edge_index[1]])
            out = torch.cat([pos_out, neg_out], dim=0); y = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)], dim=0); loss = criterion(out, y)
            if torch.isnan(loss) or torch.isinf(loss): logger.error(f"[{func_name}] NaN/Inf loss: {loss.item()}."); return np.nan, 0.0
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0); optimizer.step()
            total_loss = loss.item()
        except RuntimeError as e: logger.error(f"[{func_name}] Runtime Error train: {e}", exc_info=False); clear_cuda_cache("after train OOM"); return np.nan, 0.0
        except IndexError as idx_e: logger.error(f"[{func_name}] Index Error train: {idx_e}. Max pos idx: {train_pos_edge_index.max().item()}, Max neg idx: {neg_edge_index.max().item()}, h shape: {h.shape[0]}"); return np.nan, 0.0
        except Exception as e: logger.error(f"[{func_name}] Error train: {e}", exc_info=True); return np.nan, 0.0
        train_auc = 0.0
        try:
            with torch.no_grad():
                y_np = y.cpu().numpy(); out_sigmoid_np = torch.sigmoid(out).cpu().numpy()
                if len(np.unique(y_np)) > 1 and np.all(np.isfinite(out_sigmoid_np)): train_auc = roc_auc_score(y_np, out_sigmoid_np)
        except Exception as e: logger.warning(f"[{func_name} - Train AUC Calc] Error: {e}"); train_auc = 0.0
        del h, pos_out, neg_out, out, y, neg_edge_index, x_feat; return total_loss, train_auc

    @torch.no_grad()
    def validate():
        # (Keep corrected validate function from previous response)
        model.eval(); predictor.eval(); val_auc = 0.0; val_loss = 0.0
        func_name_val = f"{func_name} - Validate"
        eval_edge_index = getattr(data, 'train_pos_edge_index', getattr(data, 'edge_index', None))
        if eval_edge_index is None or eval_edge_index.numel() == 0: logger.warning(f"[{func_name_val}] No edges for message passing."); return 0.0, 0.0
        try:
             eval_edge_index=eval_edge_index.to(DEVICE); x_feat=data.x.to(DEVICE);
             if not x_feat.is_contiguous(): x_feat = x_feat.contiguous()
             if not eval_edge_index.is_contiguous(): eval_edge_index = eval_edge_index.contiguous()
             h=model(x_feat, eval_edge_index); logger.debug(f"[{func_name_val}] Val embeddings shape: {h.shape}")
        except RuntimeError as rt_e: logger.error(f"[{func_name_val}] Runtime error generating val embeddings: {rt_e}"); clear_cuda_cache("validate embed error - runtime"); return 0.0, 0.0
        except Exception as e: logger.error(f"[{func_name_val}] General error generating val embeddings: {e}"); clear_cuda_cache("validate embed error - general"); return 0.0, 0.0
        pos_edge_index=getattr(data, 'val_pos_edge_index', torch.empty((2,0))).to(DEVICE); neg_edge_index=getattr(data, 'val_neg_edge_index', torch.empty((2,0))).to(DEVICE)
        if pos_edge_index.numel() == 0 and neg_edge_index.numel() == 0: logger.warning(f"[{func_name_val}] No val edges."); del h, x_feat, eval_edge_index; clear_cuda_cache(f"end validate - no edges"); return 0.0, 0.0
        max_val_idx = -1
        if pos_edge_index.numel() > 0: max_val_idx = max(max_val_idx, pos_edge_index.max().item())
        if neg_edge_index.numel() > 0: max_val_idx = max(max_val_idx, neg_edge_index.max().item())
        if max_val_idx >= h.shape[0]: logger.error(f"[{func_name_val}] Max val edge index {max_val_idx} >= h dim {h.shape[0]}."); del h,x_feat,eval_edge_index,pos_edge_index,neg_edge_index; return 0.0, 0.0
        try:
            pos_out=predictor(h[pos_edge_index[0]], h[pos_edge_index[1]]) if pos_edge_index.numel() > 0 else torch.empty(0, device=DEVICE)
            neg_out=predictor(h[neg_edge_index[0]], h[neg_edge_index[1]]) if neg_edge_index.numel() > 0 else torch.empty(0, device=DEVICE)
        except IndexError as idx_e: logger.error(f"[{func_name_val}] IndexError prediction: {idx_e}. h shape: {h.shape}"); del h,x_feat,eval_edge_index,pos_edge_index,neg_edge_index; return 0.0, 0.0
        except Exception as pred_e: logger.error(f"[{func_name_val}] Error prediction calc: {pred_e}"); del h,x_feat,eval_edge_index,pos_edge_index,neg_edge_index; return 0.0, 0.0
        if pos_out.numel() > 0 or neg_out.numel() > 0:
            out=torch.cat([pos_out, neg_out], dim=0); y=torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)], dim=0)
            if out.numel() > 0:
                try: val_loss = criterion(out, y).item()
                except Exception as e: logger.warning(f"[{func_name_val} - Loss Calc] Error: {e}"); val_loss = 0.0
            else: val_loss = 0.0
            try:
                y_np=y.cpu().numpy(); out_sigmoid_np=torch.sigmoid(out).cpu().numpy()
                if len(np.unique(y_np)) > 1 and np.all(np.isfinite(out_sigmoid_np)): val_auc=roc_auc_score(y_np, out_sigmoid_np)
            except ValueError as ve: logger.warning(f"[{func_name_val} - AUC Calc] ValueError: {ve}."); val_auc = 0.0
            except Exception as e: logger.warning(f"[{func_name_val} - AUC Calc] General Error: {e}"); val_auc = 0.0
            del out, y
        else: logger.warning(f"[{func_name_val}] Both pos/neg outputs empty."); val_loss = 0.0; val_auc = 0.0
        del h, x_feat, eval_edge_index, pos_edge_index, neg_edge_index, pos_out, neg_out; clear_cuda_cache(f"end validate epoch")
        return val_loss, val_auc

    # --- Training Loop ---
    best_val_auc = -1.0; epochs_without_improvement = 0; patience = 10; saved_artifact_path = os.path.join(PREPROCESSED_DIR, MODEL_ARTIFACTS_FILE)
    logger.info(f"[{func_name}] --- Starting Training Loop ({n_epochs} epochs) ---")
    final_artifacts = artifacts.copy()
    epoch_pbar = tqdm(range(1, n_epochs + 1), desc=f"[{func_name}] Training", ncols=100, dynamic_ncols=True)
    for epoch in epoch_pbar:
        try:
            train_loss, train_auc = train_epoch(epoch)
            if np.isnan(train_loss): logger.error(f"[{func_name}] NaN loss. Stop."); break
            val_loss, val_auc = validate()
            current_lr = optimizer.param_groups[0]['lr']
            mem_allocated = torch.cuda.memory_allocated(DEVICE)/1e6 if DEVICE.type=='cuda' else 0
            epoch_pbar.set_postfix({'TrL':f'{train_loss:.3f}','VaAUC':f'{val_auc:.3f}','Best':f'{best_val_auc:.3f}','LR':f'{current_lr:.1E}','Mem':f'{mem_allocated:.0f}MB'})
            if epoch % 10 == 1 or epoch == n_epochs: logger.info(f"E:{epoch:03d}|TrL:{train_loss:.4f}|TrAUC:{train_auc:.4f}|VaL:{val_loss:.4f}|VaAUC:{val_auc:.4f}|LR:{current_lr:.1E}")
            scheduler.step(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc; logger.info(f"[{func_name}] *** New best VaAUC: {best_val_auc:.4f} at epoch {epoch}. Saving... ***")
                final_artifacts['best_epoch'] = epoch; final_artifacts['best_val_auc'] = best_val_auc; final_artifacts['model_state_dict'] = model.state_dict(); final_artifacts['predictor_state_dict'] = predictor.state_dict()
                final_artifacts['training_hyperparams'] = {'lr': lr, 'hidden': hidden_channels, 'out': out_channels, 'dropout': dropout, 'weight_decay': weight_decay, 'epochs_run': epoch, 'model_type': model_type}
                try: os.makedirs(PREPROCESSED_DIR, exist_ok=True); joblib.dump(final_artifacts, saved_artifact_path); logger.debug(f"[{func_name}] Artifacts saved.")
                except Exception as e: logger.error(f"[{func_name}] Error saving artifacts: {e}", exc_info=True)
                epochs_without_improvement = 0
            else: epochs_without_improvement += 1
            if epochs_without_improvement >= patience: logger.info(f"[{func_name}] Stopping early (patience {patience}). Best VaAUC: {best_val_auc:.4f}"); break
            if optimizer.param_groups[0]['lr'] < 1e-6: logger.info(f"[{func_name}] LR too low. Stopping early."); break
        except Exception as e: logger.error(f"[{func_name}] Error epoch {epoch}: {e}", exc_info=True); clear_cuda_cache(f"error epoch {epoch}"); break
    epoch_pbar.close()
    logger.info(f"[{func_name}] --- Training Loop Finished. Best Val AUC: {best_val_auc:.4f} ---")

    # --- Final Test Evaluation ---
    logger.info(f"[{func_name}] Loading best model for final test evaluation...")
    test_auc, test_p10, test_ndcg10 = 0.0, 0.0, 0.0
    try:
        if os.path.exists(saved_artifact_path):
            best_artifacts = joblib.load(saved_artifact_path)
            if 'model_state_dict' in best_artifacts and 'predictor_state_dict' in best_artifacts:
                logger.info(f"[{func_name}] Re-init best model..."); loaded_hp=best_artifacts.get('training_hyperparams',{}); node_feature_dim_test=best_artifacts.get('node_feature_dim');
                loaded_hidden=loaded_hp.get('hidden', 128); loaded_out=loaded_hp.get('out', 64); loaded_dropout=loaded_hp.get('dropout', 0.4); loaded_model_type=loaded_hp.get('model_type', 'SAGE');
                if node_feature_dim_test is None: raise ValueError(f"node_feature_dim missing.")
                if loaded_model_type.upper() == 'SAGE': model = GraphSAGE_GNN(node_feature_dim_test, loaded_hidden, loaded_out, dropout=loaded_dropout).to(DEVICE)
                else: raise ValueError(f"Untrained model type {loaded_model_type} loaded")
                predictor = LinkPredictor(loaded_out).to(DEVICE)
                model.load_state_dict(best_artifacts['model_state_dict']); predictor.load_state_dict(best_artifacts['predictor_state_dict']); logger.info(f"[{func_name}] Running final evaluation...");
                # Pass the split data object 'data' which contains test edges
                test_auc, test_p10, test_ndcg10 = test_model(data, model, predictor, best_artifacts, DEVICE)
                logger.info(f"[{func_name}] --- Test Results --- | AUC: {test_auc:.4f} | P@10: {test_p10:.4f} | NDCG@10: {test_ndcg10:.4f}")
                best_artifacts['test_auc']=test_auc; best_artifacts['test_precision_at_10']=test_p10; best_artifacts['test_ndcg_at_10']=test_ndcg10
                final_artifacts = best_artifacts; joblib.dump(final_artifacts, saved_artifact_path); logger.info(f"[{func_name}] Updated artifacts.")
            else: logger.warning(f"[{func_name}] Model states missing. Skip final test.")
        else: logger.warning(f"[{func_name}] Artifact file not found. Skip final test.")
    except Exception as e: logger.error(f"[{func_name}] Error during final test: {e}", exc_info=True); clear_cuda_cache("test eval error")

    return model, predictor, final_artifacts, data # Return split data

# --- Main Execution Logic (Hackathon Training Mode Only) ---
def run_main_logic(mode='t'):
    func_name = "run_main_logic"; logger.info(f"[{func_name}] Starting in mode '{mode}'...")
    if mode != 't': logger.error(f"[{func_name}] Invalid mode '{mode}'. Only 't' (training) supported."); return

    artifacts = {}
    try:
        # --- Setup & Data Loading ---
        os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(PREPROCESSED_DIR, exist_ok=True)
        users_path = os.path.join(DATA_DIR, USER_CSV_FILE); campaigns_path = os.path.join(DATA_DIR, CAMPAIGN_CSV_FILE)
        logger.info(f"[{func_name}] Checking for data files...");
        if not os.path.exists(users_path): raise FileNotFoundError(f"User data missing: {users_path}")
        if not os.path.exists(campaigns_path): raise FileNotFoundError(f"Campaign data missing: {campaigns_path}")
        logger.info(f"[{func_name}] Loading CSVs...");
        users_df_orig = pd.read_csv(users_path, low_memory=False); campaigns_df_orig = pd.read_csv(campaigns_path, low_memory=False);
        logger.info(f"[{func_name}] Data loaded. Users: {users_df_orig.shape}, Campaigns: {campaigns_df_orig.shape}")

        # --- Robust Schema Validation ---
        logger.info(f"[{func_name}] Validating schemas...");
        user_cols_needed = [USER_ID_COL, USER_AGE_COL, USER_LOCATION_COL, USER_DEVICE_COL, USER_INTERESTS_COL, USER_ACTIVITY_COL, USER_MONETARY_COL, USER_WATCH_CATEGORIES_COL, USER_CREATION_DATE_COL, USER_LOCATION_TIER_COL, USER_WEATHER_COL, USER_PAST_REWARDS_COL] # Updated expected cols
        camp_cols_needed = [CAMPAIGN_ID_COL, CAMPAIGN_BUSINESS_NAME_COL, CAMPAIGN_CATEGORY_COL, CAMPAIGN_LOCATION_COL, CAMPAIGN_PROMO_COL, CAMPAIGN_START_TIME_COL, CAMPAIGN_END_TIME_COL, CAMPAIGN_TARGET_GROUP_COL, CAMPAIGN_BUDGET_COL]
        validate_schema(users_df_orig, user_cols_needed, "User Profiles", func_name); validate_schema(campaigns_df_orig, camp_cols_needed, "Campaigns", func_name);
        logger.info(f"[{func_name}] Schema validation passed.")

        # --- Training Pipeline ---
        logger.info(f"[{func_name}] === Starting Training Pipeline ===")
        logger.info(f"[{func_name}] Step 1: Preprocessing...");
        users_df_proc, campaigns_df_proc, node_features, artifacts = preprocess_data_gnn(users_df_orig, campaigns_df_orig)
        del users_df_orig, campaigns_df_orig; gc.collect(); clear_cuda_cache("after preprocess")

        logger.info(f"[{func_name}] Step 2: Creating Edges...");
        edge_index = create_interaction_edges(users_df_proc, campaigns_df_proc, artifacts['user_map'], artifacts['campaign_map'], artifacts)
        del users_df_proc, campaigns_df_proc; gc.collect(); clear_cuda_cache("after edge creation")

        # === Explicit Edge Check ===
        if not edge_index.numel() > 0: logger.error(f"[{func_name}] No edges created."); raise ValueError("Graph has no edges.")
        logger.info(f"[{func_name}] Edge creation successful. Num edges: {edge_index.shape[1]}")

        logger.info(f"[{func_name}] Step 3: Creating Graph Data...");
        graph_data = Data(x=node_features, edge_index=edge_index); graph_data.num_users = artifacts.get('num_users', 0); graph_data.num_campaigns = artifacts.get('num_campaigns', 0); graph_data.num_nodes = graph_data.x.shape[0]
        logger.info(f"[{func_name}] Graph Data: Nodes={graph_data.num_nodes}, Feats={graph_data.x.shape[1]}, Edges={graph_data.edge_index.shape[1]}")
        if graph_data.num_nodes == 0: raise ValueError("Graph has no nodes.")
        del node_features, edge_index; gc.collect(); clear_cuda_cache("after graph creation")

        logger.info(f"[{func_name}] Step 4: Training Model...");
        model_to_train = 'SAGE'
        # Use potentially optimized hyperparameters
        model, predictor, final_artifacts, split_data = train_gnn_model(data=graph_data, artifacts=artifacts, n_epochs=150, lr=0.005, hidden_channels=64, out_channels=32, dropout=0.5, weight_decay=5e-5, model_type=model_to_train)
        if model is None: logger.error(f"[{func_name}] Training failed."); return
        logger.info(f"[{func_name}] Training Complete!")

        # --- Save Final Outputs ---
        graph_split_save_path = os.path.join(PREPROCESSED_DIR, GRAPH_DATA_FILE); torch.save(split_data, graph_split_save_path); logger.info(f"[{func_name}] Split graph data saved: {graph_split_save_path}")
        # Artifacts saved during best epoch in train_gnn_model

    # --- Error Handling ---
    except FileNotFoundError as e: logger.error(f"FILE NOT FOUND: {e}", exc_info=True); print(f"\nERROR: Ensure CSV files are in {DATA_DIR}.")
    except ValueError as e: logger.error(f"VALUE ERROR: {e}", exc_info=True); print(f"\nERROR: Invalid data/config/graph: {e}")
    except RuntimeError as e: logger.error(f"RUNTIME ERROR: {e}", exc_info=True); print(f"\nERROR: Runtime error (CUDA OOM?): {e}"); clear_cuda_cache("main runtime error")
    except Exception as e: logger.error(f"UNEXPECTED ERROR: {str(e)}", exc_info=True); print(f"\nUnexpected error: {str(e)}"); clear_cuda_cache("main unexpected error")
    finally: logger.info(f"[{func_name}] Execution finished.") ; clear_cuda_cache("final cleanup")

# === Colab Execution ===
if __name__ == "__main__":
    # --- Instructions ---
    # 1. UPLOAD UserProfiles...csv and Campaigns_Data.csv to /content/data/
    # 2. INSTALL dependencies.
    # 3. !!! CUSTOMIZE Location Maps !!!
    # 4. Runs TRAINING ('t' mode) only. Prediction logic defined in app.py.
    # ------------------
    print("\n" + "="*30 + " GNN Training Script (Hackathon - Robust Final) " + "="*30)
    run_main_logic(mode='t')
    print("="*30 + " GNN Training Finished " + "="*30 + "\n")
    print(f"Check logs. Artifacts saved in: {PREPROCESSED_DIR}")
    # Print final artifact names
    print(f"Model/Results: {os.path.join(PREPROCESSED_DIR, MODEL_ARTIFACTS_FILE)}")
    print(f"Split Graph: {os.path.join(PREPROCESSED_DIR, GRAPH_DATA_FILE)}")
    print(f"Preprocessed Campaigns: {os.path.join(PREPROCESSED_DIR, CAMPAIGNS_PREPROCESSED_FILE)}")