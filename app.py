import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ---- Visual style ----
plt.style.use('dark_background')
sns.set_palette("Set1")

# ---- Paths and model ----
BASE_DIR = Path(__file__).resolve().parent
working_dir = str(BASE_DIR)
model_path = BASE_DIR / "RF_Crop.sav"
data_path = BASE_DIR / "Data" / "Crop_recommendation.csv"
images_dir = BASE_DIR / "Images"

# load model if present
if model_path.exists():
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        model = None
        print("Failed to load model:", e)
else:
    model = None

# load dataset if present
if data_path.exists():
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        data = None
        print("Failed to read dataset:", e)
else:
    data = None

# ---- Streamlit config & CSS ----
st.set_page_config(
    page_title="Crop Recommender",
    layout="wide",
    page_icon="🍀",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #ffffff, #e6ffe6);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to right, #ffffff, #e6ffe6);
    }
    .highlight {
        color: #FF4500;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---- helper to show images safely ----
def show_image(name, caption=None, use_column_width=True):
    path = images_dir / name
    if not path.exists():
        st.info(f"(Image not found: {path})")
        return
    try:
        img = Image.open(path)
        st.image(img, caption=caption, use_column_width=use_column_width)
    except Exception as e:
        st.error(f"Failed to open image {path}: {e}")

# ---- session state & sidebar navigation ----
if 'page' not in st.session_state:
    st.session_state.page = "Overview"

st.sidebar.title("Crop Recommender")
if st.sidebar.button("Overview"):
    st.session_state.page = "Overview"
if st.sidebar.button("Get Recommendation"):
    st.session_state.page = "Get Recommendation"
if st.sidebar.button("Analysis"):
    st.session_state.page = "Analysis"

# ---- Overview page ----
if st.session_state.page == "Overview":
    st.title("Overview")
    st.subheader("Welcome to the Crop Recommendation App!")
    show_image("image1.jpg", caption="Healthy Crops")
    st.write("""
        This application assists farmers in selecting the optimal crop to cultivate, considering soil composition
        and environmental conditions. Provide N, P, K, temperature, humidity, pH, and rainfall to get recommendations.
    """)
    show_image("mod_comparison.png", caption="Model comparison (if available)")

# ---- Get Recommendation page ----
elif st.session_state.page == "Get Recommendation":
    st.title("Crop Recommendation")
    st.write("Enter the soil & environmental values to get a crop recommendation.")
    N = st.number_input('Nitrogen (N)', min_value=0, max_value=500, value=0)
    P = st.number_input('Phosphorus (P)', min_value=0, max_value=500, value=0)
    K = st.number_input('Potassium (K)', min_value=0, max_value=500, value=0)
    temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=0.0)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=500.0, value=0.0)
    pH = st.number_input('pH', min_value=0.0, max_value=14.0, value=0.0)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=0.0)

    user_input = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    if st.button('Predict'):
        if np.all(user_input == 0):
            st.warning("Please enter valid values.")
        else:
            if model is None:
                st.error("Model file not found or failed to load. Prediction unavailable.")
            else:
                prediction = model.predict(user_input)
                crop = prediction[0]
                st.markdown(f"🌾 Based on your input, the recommended crop is: <span class='highlight'>{crop}</span>.", unsafe_allow_html=True)

# ---- Analysis page with plot options ----
elif st.session_state.page == "Analysis":
    st.title("Data Analysis & Plots")
    st.write("Choose which plot to view. For Feature Distribution, choose the feature you want to inspect.")

    # allow uploading dataset or fallback to loaded dataset
    uploaded_file = st.file_uploader("Upload CSV for analysis (optional)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Uploaded CSV loaded.")
        except Exception as e:
            st.error(f"Failed to load uploaded CSV: {e}")
            df = data  # fallback
    else:
        df = data

    if df is None:
        st.error("No dataset available. Place 'Crop_recommendation.csv' in Data/ or upload a CSV.")
    else:
        st.write("Dataset preview:")
        st.dataframe(df.head())

        plot_options = [
            "Feature Distribution (Histogram + Violin + Box)",
            "Correlation Heatmap",
            "Crop Count (label/crop/target column required)",
            "Model Comparison Image"
        ]
        choice = st.selectbox("Select plot type:", plot_options)

        # ---------- Feature Distribution: user's 3-panel chart ----------
        if choice == "Feature Distribution (Histogram + Violin + Box)":
            # choose feature (skip final column if it's the label)
            feature_candidates = list(df.columns)
            # if last column is likely label, exclude it (common pattern)
            if len(feature_candidates) > 1:
                feature_candidates_display = feature_candidates[:-1]
            else:
                feature_candidates_display = feature_candidates

            feature = st.selectbox("Choose a feature to visualize:", feature_candidates_display)

            if feature:
                # create 3-panel figure (histogram + violin + box)
                fig, ax = plt.subplots(1, 3, figsize=(18, 4))
                try:
                    sns.histplot(data=df, x=feature, kde=True, bins=20, ax=ax[0])
                    sns.violinplot(data=df, x=feature, ax=ax[1])
                    sns.boxplot(data=df, x=feature, ax=ax[2])
                    plt.suptitle(f"Visualizing {feature}", size=20)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Failed to render plots for feature '{feature}': {e}")
                finally:
                    plt.close(fig)

                st.write(f"""
                    ### Analysis of {feature}
                    - **Histogram + KDE**: Shows how values of {feature} are distributed and their overall shape.
                    - **Violin Plot**: Reveals density and multi-modality (thick regions = more observations).
                    - **Box Plot**: Summarizes median, interquartile range, and highlights outliers.
                """)
                st.info("Tip: If the histogram is heavily skewed, consider log or power transforms before modeling.")

        # ---------- Correlation heatmap ----------
        elif choice == "Correlation Heatmap":
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                st.warning("Not enough numeric columns to compute correlations.")
            else:
                st.subheader("📈🔗📉 Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = numeric.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("**Analysis:** Strong correlations (positive or negative) show features that move together. Consider removing or combining highly correlated features.")

        # ---------- Crop count ----------
        elif choice == "Crop Count (label/crop/target column required)":
            label_col = None
            for c in ['label', 'crop', 'target']:
                if c in df.columns:
                    label_col = c
                    break
            if label_col is None:
                st.warning("No label/crop/target column found. Upload a labelled dataset to view crop counts.")
            else:
                counts = df[label_col].value_counts().reset_index()
                counts.columns = ['Crop', 'Count']
                counts['Percentage (%)'] = (counts['Count'] / counts['Count'].sum() * 100).round(2)

                # Add rank or color styling
                st.subheader("🌾 Crop Count Summary")
                st.markdown("Below is a detailed table of crop frequencies with their relative percentages in the dataset.")

                # Display as a styled dataframe
                styled_counts = (
                    counts.style
                    .background_gradient(subset=['Count'], cmap='Greens')
                    .format({'Percentage (%)': '{:.2f}%'})
                    .set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#2e2e2e'),
                                                           ('color', 'white'),
                                                           ('font-weight', 'bold'),
                                                           ('text-align', 'center')]},
                        {'selector': 'tbody td', 'props': [('text-align', 'center')]}
                    ])
                )

                st.dataframe(styled_counts, use_container_width=True)
                st.markdown("**Analysis:** Crops with very low counts may need balancing techniques such as oversampling or class weighting before model training.")
        # ---------- Model comparison image ----------
        elif choice == "Model Comparison Image":
            st.write("Model comparison (image from Images/ if available):")
            show_image("mod_comparison.png", caption="Model comparison", use_column_width=True)
            st.markdown("**Analysis:** Use this visual to compare model performance metrics across candidate algorithms.")

# ---- end ----
