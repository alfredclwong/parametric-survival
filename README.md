process.py
- Load dummy data (dummy.csv)
- Filter out T & C both nan, T < 0, X all nan
- Add Y = min(T, C)
- Add D = T < C
- Standardise X features (saving the scalers to disk)
- Fill missing X data with zeros
- Save parquet (dummy_processed.parquet)
- Visualise data

