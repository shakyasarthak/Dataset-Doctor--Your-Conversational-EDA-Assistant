import pandas as pd
from ydata_profiling import ProfileReport
import numpy as np

def get_summary(df):
    summary = df.describe(include='all').T
    summary['missing'] = df.isnull().sum()
    summary['missing_pct'] = (df.isnull().mean() * 100).round(2)
    summary['dtype'] = df.dtypes
    return summary

def get_missing_report(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (df.isnull().mean() * 100).round(2)
    return pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_pct
    })

def generate_profile(df):
    profile = ProfileReport(df,
                            explorative=True,
                            correlations={"auto": {"calculate": True}},
                            interactions={"continuous": True})
    report_path = "eda_report.html"
    profile.to_file(report_path)
    return report_path

def clean_data(df):
    cleaned_df = df.copy()
    for col in cleaned_df.select_dtypes(include='number'):
        if cleaned_df[col].isnull().sum() > 0:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    for col in cleaned_df.select_dtypes(exclude='number'):
        if cleaned_df[col].isnull().sum() > 0:
            mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else ''
            cleaned_df[col] = cleaned_df[col].fillna(mode_val)
    cleaned_df = cleaned_df.drop_duplicates()
    for col in cleaned_df.select_dtypes(include='number'):
        if len(cleaned_df[col].unique()) > 10:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df[col] = np.where(
                cleaned_df[col] < lower_bound,
                lower_bound,
                np.where(
                    cleaned_df[col] > upper_bound,
                    upper_bound,
                    cleaned_df[col]
                )
            )
    return cleaned_df

def suggest_features(df):
    suggestions = []
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].nunique() > 10:
            suggestions.append(f"Consider log transformation for {col}")
        if abs(df[col].skew()) > 1.0:
            suggestions.append(f"Consider transformation for {col} (skew={df[col].skew():.2f})")
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    for col in categorical_cols:
        if df[col].nunique() < 20:
            suggestions.append(f"Consider one-hot encoding for {col}")
        elif df[col].nunique() < 100:
            suggestions.append(f"Consider target encoding for {col}")
    if len(numeric_cols) >= 2:
        suggestions.append(f"Create interaction feature between {numeric_cols[0]} and {numeric_cols[1]}")
    time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if time_cols:
        suggestions.append(f"Extract datetime features from {time_cols[0]} (e.g., day, month, year)")
    return suggestions
