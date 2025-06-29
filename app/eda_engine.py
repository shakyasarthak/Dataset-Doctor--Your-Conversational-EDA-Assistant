import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
import numpy as np
from scipy import stats

def get_summary(df):
    """Generate statistical summary with enhanced metrics"""
    summary = df.describe(include='all').T
    summary['missing'] = df.isnull().sum()
    summary['missing_pct'] = (df.isnull().mean() * 100).round(2)
    summary['dtype'] = df.dtypes
    return summary

def get_missing_report(df):
    """Generate comprehensive missing value report"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (df.isnull().mean() * 100).round(2)
    return pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_pct
    })

def generate_profile(df):
    """Generate interactive EDA report"""
    profile = ProfileReport(df,
        explorative=True,
        correlations={"auto": {"calculate": True}},
        interactions={"continuous": True})
    report_path = "eda_report.html"
    profile.to_file(report_path)
    return report_path

def clean_data(df):
    """Auto-clean data with intelligent handling"""
    # Handle missing values
    for col in df.select_dtypes(include='number'):
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in df.select_dtypes(exclude='number'):
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle outliers using IQR
    for col in df.select_dtypes(include='number'):
        if len(df[col].unique()) > 10:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    return df
