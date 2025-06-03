from local_values import data_folder
import requests
import numpy as np
import pyupbit
import json
import os

url=os.environ['MODEL_URL']
access_key=os.environ['ACCESS_KEY']
secret_key=os.environ['SECRET_KEY']
upbit=pyupbit.Upbit(access_key,secret_key)
MODEL_CONFIG = os.environ.get('MODEL_CONFIG',f'{data_folder}model.json')

with open(MODEL_CONFIG,'r') as f:
    md_config=json.load(f)

timeframes = md_config.get("timeframes")
valid_cols = md_config.get("cols")
in_dims = md_config.get("inputs")
position = 0

def ttrade(dfs,time_df):
    global position
    x_parts = [dfs[tf][valid_cols[tf]].iloc[-in_dims[tf]:]
            .to_numpy(dtype=np.float32, copy=False)
            for tf in timeframes]
    t_parts = [time_df[tf].iloc[-in_dims[tf]:]
            .to_numpy(dtype=np.float32, copy=False)
            for tf in timeframes]
    payload = b''.join(
        x.tobytes() + t.tobytes()
        for x, t in zip(x_parts,t_parts)
    )
    pos = np.array([position], dtype=np.float32)
    payload += pos.tobytes()

    headers={"Content-Type":"application/octet-stream"}
    response=requests.post(url,data=payload,headers=headers)
    result=np.array(response.json())

    print(f"{result}")

    if result == 0: 
        print(upbit.buy_market_order(ticker, upbit.get_balance("KRW")*0.985))
        postion = 1
    elif result == 2:
        print(upbit.sell_market_order(ticker, upbit.get_balance(ticker)*0.985))
        position = 0