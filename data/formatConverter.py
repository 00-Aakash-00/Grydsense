import polars as pl
import pandas as pd

df = pd.read_excel("Primary_day_wise.xlsx")  
df.to_parquet("Primary_day_wise.parquet")