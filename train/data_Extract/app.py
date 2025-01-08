from local_values import raw_datas,tickers,raw_folder
from data_loader import request_data,append_data,get_last_time
from datetime import datetime, timedelta
import pandas as pd
import time
import os

if os.path.exists(raw_folder)==False:
    print(f'Need Linked Folder at {raw_folder}')
    exit()

for tick in tickers:
    folder_path=raw_folder+tick+'/'
    print(f'Folder Path: {folder_path}')
    if os.path.exists(folder_path)==False:
        print(f'No folder {folder_path}')
        os.mkdir(folder_path)
    end_time=get_last_time(folder_path)+timedelta(minutes=1)
    print(f'\tSet End Time to {end_time}')
    print(f'Start collect data {tick}')
    contents=pd.DataFrame()
    response=request_data(tick)
    while True:
        response=response[:end_time]
        if (response.shape[0] == 0):
            break
        contents=pd.concat([contents,response])
        to_time=(response.index[-1]-timedelta(hours=9)).strftime('%Y-%m-%dT%H:%M:%S')
        time.sleep(0.65)
        response=request_data(tick,'&to='+to_time)
    append_data(folder_path,contents.sort_index())
    print(f"update {tick} from {contents.index[0]} to {contents.index[-1]}\n")
