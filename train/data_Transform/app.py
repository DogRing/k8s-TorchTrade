from local_values import tickers,raw_folder,data_folder
from data_loader import load_data
from data_transform import data_transform,data_scale
import pandas as pd
import json
import os

indicator_config_file=os.environ.get('INDICATOR_CONFIG',f'{data_folder}indicator.json')
scale_config_file=os.environ.get('SCALER_CONFIG',f'{data_folder}scale.json')
data_length = int(os.environ.get('DATA_LEN','2000000'))

print(f'INDICATOR_CONFIG: {indicator_config_file}')
print(f'DATA_LENGTH: {data_length}')

with open(indicator_config_file,'r') as f:
    tf_config=json.load(f)
if os.path.exists(scale_config_file):
    try:
        with open(scale_config_file,'r') as f:
            scale_config=json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {scale_config_file}")
else:
    scale_config=None
    print(f"Config file {scale_config_file} not found")

print("")
for i,tick in enumerate(tickers):
    folder_path=raw_folder+tick+'/'
    print(f'Data Load from {folder_path} with {data_length}')
    df=load_data(folder_path,data_length)
    print("\tresample, fillna")
    df=df.resample(rule='min').first()
    df[['volume','value']]=df[['volume','value']].fillna(0)
    df=df.ffill()
    print("\t Transform data as INDICATOR_CONFIG")
    dfs=data_transform(df,tf_config)
    print(f"\ttransformed file {tick} length : {len(df)}")
    if scale_config:
        os.makedirs(sacler_path, exist_ok=True)
        dfs=data_scale(dfs,scale_config,True,sacler_path)
        print(f"\t scaling file")
    df=pd.concat(dfs,axis=1)
    non_nan_index = df.index.get_loc(df.index[~df.isna().any(axis=1)][0])
    target_file = data_folder+tick+'.csv'
    print(f'Save as {target_file}\t{i+1} / {len(tickers)}\t {len(df)} - {non_nan_index}')
    df[non_nan_index:].to_csv(target_file)
print(df.columns)