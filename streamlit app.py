import streamlit as st
import jwt

SECRET_KEY = "YOUR_SECRET_KEY"

# --- JWT validation ---
query_params = st.experimental_get_query_params()
token = query_params.get("token", [None])[0]

if not token:
    st.error("🚫 Unauthorized. Please login first.")
    st.stop()

try:
    decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    st.sidebar.success(f"✅ Logged in as {decoded['user']}")
except Exception as e:
    st.error("🚫 Invalid or expired token. Please login again.")
    st.stop()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import boxcox, zscore, chi2_contingency

# --------------------------
# App Config
# --------------------------
st.set_page_config(page_title="📊 Data Analysis Dashboard", layout="wide")
st.title("📊 Your Data Analysis Dashboard")

# Sidebar Navigation
page = st.sidebar.radio("📌 Navigate", [
    "Upload Data",
    "Dataset Overview",
    "Visualization",
    "Outlier Detection & Fixing",
    "Statistical Tests",
    "Download Data"
])

# Global dataset
if "df" not in st.session_state:
    st.session_state.df = None

# --------------------------
# Upload Data Page
# --------------------------
if page == "Upload Data":
    uploaded_file = st.file_uploader("Upload CSV, TXT, or MAT file", type=['csv', 'txt', 'mat'])
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type in ['csv', 'txt']:
            st.session_state.df = pd.read_csv(uploaded_file)

         elif file_type == 'mat':
             mat_data = loadmat(uploaded_file)
             for key in mat_data:
                 if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                     signal = mat_data[key].squeeze()  # remove singleton dims

                     if signal.ndim == 1:  # ✅ 1D signal (e.g., PPG, single-lead ECG)
                        st.session_state.df = pd.DataFrame({"Signal": signal})

                     elif signal.ndim == 2:  # ✅ Multi-channel (e.g., 12-lead ECG)
                         n_channels = signal.shape[1] if signal.shape[0] > signal.shape[1] else signal.shape[0]
                         if signal.shape[0] < signal.shape[1]:
                             signal = signal.T  # transpose if samples are in wrong axis
                         col_names = [f"Channel_{i+1}" for i in range(signal.shape[1])]
                         st.session_state.df = pd.DataFrame(signal, columns=col_names)

                     break  # take the first valid array


        st.success("✅ File successfully loaded!")

# --------------------------
# Dataset Overview
# --------------------------
if page == "Dataset Overview" and st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("👀 Dataset Preview & Info")

    st.dataframe(df.head())
    st.write("**Shape:**", df.shape)
    st.write("**Column Types:**")
    st.write(df.dtypes)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include='all'))

# --------------------------
# Visualization
# --------------------------
# --------------------------
# Visualization
# --------------------------
if page == "Visualization" and st.session_state.df is not None:
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Histograms",
        "📦 Boxplots",
        "🔥 Correlation",
        "📑 Categorical Counts",
        "🎨 Custom Plots"
    ])

    # --- Histograms ---
    with tab1:
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols, key="hist_col")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

    # --- Boxplots ---
    with tab2:
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols, key="box_col")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

    # --- Correlation ---
    with tab3:
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # --- Categorical counts ---
    with tab4:
        if cat_cols:
            col = st.selectbox("Select categorical column", cat_cols, key="cat_col")
            fig, ax = plt.subplots()
            sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # --- Custom Plots ---
    with tab5:
        st.subheader("🎨 Custom Visualization")

        all_cols = df.columns.tolist()
        x_axis = st.selectbox("Select X-axis", ["Samples"] + all_cols, key="x_axis")
        y_axis = st.selectbox("Select Y-axis", numeric_cols, key="y_axis")

        plot_type = st.selectbox("Select Plot Type", [
            "Line Plot", "Scatter Plot", "Bar Plot"
        ])

        # Range selection
        start_idx, end_idx = st.slider(
            "Select data range", 0, len(df)-1, (0, min(100, len(df)-1))
        )

        plot_df = df.iloc[start_idx:end_idx+1]

        fig, ax = plt.subplots()

        if plot_type == "Line Plot":
            if x_axis == "Samples":
                ax.plot(np.arange(len(plot_df)), plot_df[y_axis], label=y_axis)
                ax.set_xlabel(f"Samples (0-{len(plot_df)-1})")
            else:
                ax.plot(plot_df[x_axis], plot_df[y_axis], label=y_axis)
                ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.legend()

        elif plot_type == "Scatter Plot":
            if x_axis == "Samples":
                ax.scatter(np.arange(len(plot_df)), plot_df[y_axis])
                ax.set_xlabel(f"Samples (0-{len(plot_df)-1})")
            else:
                ax.scatter(plot_df[x_axis], plot_df[y_axis])
                ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)

        elif plot_type == "Bar Plot":
            if x_axis == "Samples":
                ax.bar(np.arange(len(plot_df)), plot_df[y_axis])
                ax.set_xlabel(f"Samples (0-{len(plot_df)-1})")
            else:
                ax.bar(plot_df[x_axis], plot_df[y_axis])
                ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)

        st.pyplot(fig)

# --------------------------
# Outlier Detection & Fixing
# --------------------------
if page == "Outlier Detection & Fixing" and st.session_state.df is not None:
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("🧹 Missing Value Handling")

    missing_counts = df.isnull().sum()
    st.write("**Missing values per column:**")
    st.write(missing_counts[missing_counts > 0])

    if missing_counts.sum() > 0:
        action = st.selectbox(
            "Choose how to handle missing values",
            ["Do nothing", "Drop rows", "Drop columns",
             "Fill values", "Forward Fill", "Backward Fill"]
        )

        if action == "Drop rows":
            df = df.dropna()
            st.success("✅ Rows with missing values dropped!")

        elif action == "Drop columns":
            df = df.dropna(axis=1)
            st.success("✅ Columns with missing values dropped!")

        elif action == "Fill values":
            fill_method = st.selectbox("Choose fill method", ["Mean", "Median", "Mode", "Custom"])
            if fill_method == "Mean":
                df = df.fillna(df.mean(numeric_only=True))
            elif fill_method == "Median":
                df = df.fillna(df.median(numeric_only=True))
            elif fill_method == "Mode":
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])
            elif fill_method == "Custom":
                custom_value = st.text_input("Enter value to fill missing values")
                if custom_value:
                    df = df.fillna(custom_value)
            st.success("✅ Missing values filled!")

        elif action == "Forward Fill":
            df = df.ffill()
            st.success("✅ Forward filled missing values!")

        elif action == "Backward Fill":
            df = df.bfill()
            st.success("✅ Backward filled missing values!")

    else:
        st.success("🎉 No missing values found.")

    st.dataframe(df.head())



    st.subheader("⚠️ Outlier Detection")
    method = st.radio("Choose Method", ["IQR", "Z-Score"])

    if method == "IQR":
        if st.button("Detect Outliers (IQR)"):
            outliers = {}
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                outliers[col] = ((df[col] < lower) | (df[col] > upper)).sum()
            st.write(outliers)

    elif method == "Z-Score":
        if st.button("Detect Outliers (Z-Score)"):
            outliers = {}
            for col in numeric_cols:
                z_scores = np.abs(zscore(df[col].dropna()))
                outliers[col] = (z_scores > 3).sum()
            st.write(outliers)

    st.subheader("⚡ Fix Outliers")
    if numeric_cols:
        col = st.selectbox("Select column", numeric_cols)
        fix_method = st.selectbox("Choose Fix Method", [
            "Remove", "Cap", "Log Transform", "Square Root Transform",
            "Box-Cox Transform", "Impute with Median",
            "Linear Regression", "K-Means Clustering"
        ])

        if st.button("Apply Fix"):
            try:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                outlier_idx = df.index[(df[col] < lower) | (df[col] > upper)]

                if fix_method == "Remove":
                    df = df.drop(outlier_idx)
                elif fix_method == "Cap":
                    df[col] = np.where(df[col] > upper, upper,
                                np.where(df[col] < lower, lower, df[col]))
                elif fix_method == "Log Transform":
                    df[col] = np.log1p(df[col].clip(lower=0))
                elif fix_method == "Square Root Transform":
                    df[col] = np.sqrt(df[col].clip(lower=0))
                elif fix_method == "Box-Cox Transform":
                    if (df[col] <= 0).any():
                        shift = abs(df[col].min()) + 1
                        df[col] = df[col] + shift
                    df[col], _ = boxcox(df[col])
                elif fix_method == "Impute with Median":
                    median_val = df[col].median()
                    df.loc[outlier_idx, col] = median_val
                elif fix_method == "Linear Regression":
                    X = df.drop(columns=[col]).select_dtypes(include=np.number)
                    if not X.empty:
                        y = df[col]
                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        df.loc[outlier_idx, col] = y_pred[outlier_idx]
                elif fix_method == "K-Means Clustering":
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(df[[col]])
                    centers = kmeans.cluster_centers_
                    df.loc[outlier_idx, col] = [centers[label][0] for label in kmeans.labels_[outlier_idx]]

                st.session_state.df = df
                st.success(f"✅ {fix_method} applied to {col}")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --------------------------
# Statistical Tests
# --------------------------
if page == "Statistical Tests" and st.session_state.df is not None:
    df = st.session_state.df
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.subheader("📊 Chi-Square Test")
    if len(cat_cols) >= 2:
        col1 = st.selectbox("Select first categorical column", cat_cols, key="chi1")
        col2 = st.selectbox("Select second categorical column", [c for c in cat_cols if c != col1], key="chi2")

        if st.button("Run Chi-Square Test"):
            contingency_table = pd.crosstab(df[col1], df[col2])
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            st.write("**Contingency Table:**")
            st.write(contingency_table)
            st.write(f"Chi-Square Statistic: {chi2:.4f}, p-value: {p:.4f}")

            if p < 0.05:
                st.error("❌ Variables are likely dependent (reject H0)")
            else:
                st.success("✅ Variables are likely independent (fail to reject H0)")
    else:
        st.warning("⚠️ Need at least 2 categorical columns.")

# --------------------------
# Download Data
# --------------------------
if page == "Download Data" and st.session_state.df is not None:
    st.subheader("💾 Download Cleaned Dataset")
    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")
