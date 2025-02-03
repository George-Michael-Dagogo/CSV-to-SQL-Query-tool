import streamlit as st
import pandas as pd
import sqlite3
import pyarrow.parquet as pq
from io import StringIO, BytesIO
import xlrd
import openpyxl
import numpy as np
import re
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="SQL Query Tool",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox {
        margin-bottom: 20px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def clean_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a pandas DataFrame to make it more SQL-friendly"""
    cleaned_df = df.copy()
    
    def clean_column_name(name):
        """Clean column names to be SQL-friendly"""
        name = str(name).lower()
        name = re.sub(r'[^a-z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        if name[0].isdigit():
            name = 'col_' + name
        return name
    
    # Clean column names
    cleaned_df.columns = [clean_column_name(col) for col in cleaned_df.columns]

    # First pass: Detect and convert string columns with numbers and commas
    for column in cleaned_df.columns:
        # Check if column is string/object type
        if cleaned_df[column].dtype == 'object':
            # Check if all non-null values in column could be numeric after removing commas
            sample = cleaned_df[column].dropna().astype(str).str.replace(',', '')
            try:
                # Try converting to numeric - if successful, this is a numeric column
                pd.to_numeric(sample)
                # If we get here, conversion worked, so clean the actual column
                cleaned_df[column] = cleaned_df[column].astype(str).str.replace(',', '')
                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
            except:
                # Not a numeric column, skip it
                continue
    
    for column in cleaned_df.columns:
        dtype = cleaned_df[column].dtype
        
        # Handle string/object columns
        if dtype == 'object':
            cleaned_df[column] = cleaned_df[column].apply(
                lambda x: None if isinstance(x, str) and not x.strip() else x
            )
            cleaned_df[column] = cleaned_df[column].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            cleaned_df[column] = cleaned_df[column].replace(
                ['nan', 'NaN', 'null', 'NULL', 'None', 'NA', ''], None
            )
            
        # Handle numeric columns
        elif np.issubdtype(dtype, np.number):
            cleaned_df[column] = cleaned_df[column].replace([np.inf, -np.inf], None)
            if np.issubdtype(dtype, np.floating):
                cleaned_df[column] = cleaned_df[column].round(6)
        
        elif np.issubdtype(dtype, np.number):
            # Add this new section here:
            # Remove commas and convert to numeric if possible
            if cleaned_df[column].dtype == 'object':
                cleaned_df[column] = cleaned_df[column].astype(str).str.replace(',', '').replace('', None)
                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
            
            # Then continue with existing code:
            cleaned_df[column] = cleaned_df[column].replace([np.inf, -np.inf], None)
            if np.issubdtype(dtype, np.floating):
                cleaned_df[column] = cleaned_df[column].round(6)

        # Handle date/time columns
        elif np.issubdtype(dtype, np.datetime64):
            cleaned_df[column] = cleaned_df[column].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else None
            )
    
    return cleaned_df

def validate_data(df: pd.DataFrame) -> dict:
    """Validate the DataFrame and return potential issues"""
    issues = {}
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        issues['duplicate_columns'] = [
            col for col in df.columns if list(df.columns).count(col) > 1
        ]
    
    # Check for columns with mixed data types
    mixed_type_cols = []
    for col in df.columns:
        types = df[col].apply(type).unique()
        if len(types) > 1 and pd.notnull(df[col]).any():
            mixed_type_cols.append({
                'column': col,
                'types': [str(t) for t in types]
            })
    if mixed_type_cols:
        issues['mixed_types'] = mixed_type_cols
    
    return issues

def read_file(uploaded_file, file_type):
    """Read different file formats into a pandas DataFrame"""
    try:
        if file_type == "csv":
            return pd.read_csv(uploaded_file)
        elif file_type == "excel":
            return pd.read_excel(uploaded_file)
        elif file_type == "parquet":
            parquet_bytes = BytesIO(uploaded_file.read())
            return pd.read_parquet(parquet_bytes)
        elif file_type == "json":
            return pd.read_json(uploaded_file)
        elif file_type == "txt":
            return pd.read_csv(uploaded_file, sep='\t')
        elif file_type == "feather":
            return pd.read_feather(uploaded_file)
        elif file_type == "pickle":
            return pd.read_pickle(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def create_sql_table(df, table_name):
    """Create SQLite table from DataFrame"""
    conn = sqlite3.connect(':memory:')
    df.to_sql(table_name, conn, index=False)
    return conn

def run_query(conn, query):
    """Execute SQL query and return results as DataFrame"""
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

def prepare_download_data(df, format_type):
    """Prepare data for download with proper type handling"""
    try:
        if format_type == "csv":
            return df.to_csv(index=False).encode('utf-8')
        elif format_type == "excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()
        else:  # json
            return df.to_json(orient='records').encode('utf-8')
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")
        return None

def main():
    st.title("🔍 Multi-format SQL Query Tool For Analogue Shifts")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📁 File Upload")
        supported_formats = {
            "csv": "CSV (Comma Separated Values)",
            "excel": "Excel (XLSX, XLS)",
            "parquet": "Parquet",
            "json": "JSON",
            "txt": "TXT (Tab Separated)",
            "feather": "Feather",
            "pickle": "Pickle (PKL)"
        }
        
        file_type = st.selectbox(
            "Select file type",
            list(supported_formats.keys()),
            format_func=lambda x: supported_formats[x]
        )
        
        allowed_types = ["xlsx", "xls"] if file_type == "excel" else [file_type]
        uploaded_file = st.file_uploader(
            f"Choose a {supported_formats[file_type]} file",
            type=allowed_types
        )
        
        if uploaded_file:
            st.subheader("🧹 Data Cleaning")
            clean_data = st.checkbox("Enable automatic data cleaning", value=True,
                help="Clean column names, standardize NULL values, handle special characters, etc.")
    
    with col2:
        if uploaded_file is not None:
            df = read_file(uploaded_file, file_type)
            
            if df is not None:
                # Data cleaning section
                if clean_data:
                    with st.spinner("Cleaning data..."):
                        df = clean_for_sql(df)
                        issues = validate_data(df)
                        
                        if issues:
                            with st.expander("⚠️ Data Quality Issues Found"):
                                if 'duplicate_columns' in issues:
                                    st.warning("Duplicate column names found: " + 
                                             ", ".join(issues['duplicate_columns']))
                                
                                if 'mixed_types' in issues:
                                    st.warning("Columns with mixed data types:")
                                    for col in issues['mixed_types']:
                                        st.write(f"- {col['column']}: {', '.join(col['types'])}")
                
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                with st.expander("📋 Dataset Information"):
                    col3, col4, col5 = st.columns(3)
                    with col3:
                        st.metric("Rows", df.shape[0])
                    with col4:
                        st.metric("Columns", df.shape[1])
                    with col5:
                        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.2f} KB")
                    
                    st.markdown("#### Column Details:")
                    col_info = pd.DataFrame({
                        'Data Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum()
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                conn = create_sql_table(df, "data")
                
                st.subheader("🔍 SQL Query")
                
                query_templates = {
                    "Select all": "SELECT * FROM data LIMIT 5",
                    "Count rows": "SELECT COUNT(*) FROM data",
                    "Group by": "SELECT column_name, COUNT(*) FROM data GROUP BY column_name",
                    "Filter data": "SELECT * FROM data WHERE column_name = value",
                    "Sort data": "SELECT * FROM data ORDER BY column_name DESC LIMIT 5",
                    "Null values": "SELECT column_name, COUNT(*) FROM data WHERE column_name IS NULL",
                    "Distinct values": "SELECT DISTINCT column_name FROM data"
                }
                
                template = st.selectbox("Query templates", ["Custom"] + list(query_templates.keys()))
                query = st.text_area(
                    "SQL Query",
                    query_templates.get(template, "SELECT * FROM data LIMIT 5"),
                    height=100
                )
                
                if st.button("🚀 Run Query"):
                    if query.strip():
                        result = run_query(conn, query)
                        if result is not None:
                            st.subheader("📈 Query Result")
                            st.dataframe(result, use_container_width=True)
                            
                            col6, col7 = st.columns([2, 1])
                            with col6:
                                download_format = st.selectbox(
                                    "Select download format",
                                    ["csv", "excel", "json"]
                                )
                            
                            with col7:
                                download_data = prepare_download_data(result, download_format)
                                if download_data is not None:
                                    st.download_button(
                                        label=f"📥 Download as {download_format.upper()}",
                                        data=download_data,
                                        file_name=f"query_results.{download_format}",
                                        mime={
                                            "csv": "text/csv",
                                            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            "json": "application/json"
                                        }[download_format]
                                    )
                    else:
                        st.warning("⚠️  Please enter a SQL query")

if __name__ == "__main__":
    main()