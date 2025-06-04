from local_values import data_folder
from data_transform import data_transform,data_scale
from ttrade import ttrade

from contextlib import redirect_stdout,redirect_stderr
from confluent_kafka import Consumer
import numpy as np
import pandas as pd
import json
import copy
import time
import ast
import os
import io 

buf = io.StringIO()

TICK = os.environ.get('TICK','KRW-BTC')
TOPIC = os.environ.get('TOPIC','krw-btc')
INTERVAL = int(os.environ.get('INTERVAL','30'))
DATA = int(os.environ.get('DATA_LENGTH','10000'))
LARGE_N = ast.literal_eval(os.environ.get('LARGE','[]'))
DATA_N = ast.literal_eval(os.environ.get('DATA_N_LENGTH','{}'))
DEBUG = os.environ.get('DEBUG',False)

DATA_CONFIG = os.environ.get('DATA_CONFIG',f'{data_folder}indicator.json')
SCALER_CONFIG = os.environ.get('SCALER_CONFIG',f'{data_folder}scale.json')

print(f"TICK: {TICK}")
print(f"TOPIC: {TOPIC}")
print(f"INTERVAL: {INTERVAL}")
print(f"DATA: {DATA}")
print(f"LARGE_N: {LARGE_N}")
print(f"DATA_N: {DATA_N}")

cols = ['open','high','low','close','value']
SEP = 60 // INTERVAL
ROWS = DATA * SEP
COLS = len(cols)

with open(DATA_CONFIG,'r') as f:
    tf_config=json.load(f)
with open(SCALER_CONFIG,'r') as f:
    sc_config=json.load(f)

n_config = copy.deepcopy(tf_config)
n_config["data"] = [i for i in tf_config["data"] if "RANGE" in i and i["RANGE"] in DATA_N]
tf_config["data"] = [i for i in tf_config["data"] if "RANGE" not in i or i["RANGE"] not in DATA_N]

consumer = Consumer({
    'bootstrap.servers' : 'my-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092',
    'group.id' : f'handle-{TOPIC}',
    'auto.offset.reset' : 'earliest',
    'enable.auto.commit' : False,
})
consumer.subscribe([TOPIC]+[f'{TOPIC}-{n}' for n in DATA_N])


x_buf = np.zeros((ROWS, COLS), dtype=np.float32)
t_buf = np.zeros(ROWS, dtype=np.float32)
index_ns = np.zeros(ROWS, dtype=np.int64)

base_df = pd.DataFrame(x_buf, columns=cols, index=pd.DatetimeIndex(index_ns))
time_df = pd.DataFrame(t_buf, columns=['timestamp'], index=pd.DatetimeIndex(index_ns))
order_tem = np.arange(ROWS, dtype=np.int64)
x_idx=np.where(order_tem % SEP == (SEP-1))[0]

n_buf = { k: np.zeros((v,COLS), dtype=np.float32) for k,v in DATA_N.items() }
n_t_buf = { k: np.zeros(v, dtype=np.float32) for k,v in DATA_N.items() }
n_index_ns = {k: np.zeros(v, dtype=np.int64) for k, v in DATA_N.items()}

n_base_df = {k: pd.DataFrame(n_buf[k],  columns=cols, index=pd.DatetimeIndex(n_index_ns[k])) for k in DATA_N }
n_time_df = {k: pd.DataFrame(n_t_buf[k], columns=['timestamp'], index=pd.DatetimeIndex(n_index_ns[k])) for k in DATA_N }
_n_order = {k: np.arange(v) for k,v in DATA_N.items() } 
n_order = _n_order.copy()
n_head = { k: 0 for k in DATA_N }
head = 0
_index_values = base_df.index.view("int64")

def angle_encoding(timestamp):
    weights = { 'hour':0.1,'day':0.4,'week':0.3,'year':0.2 }
    
    hour_angle = (timestamp%3600) / 3600 * 2 * np.pi
    day_angle = (timestamp%86400) / 86400 * 2 * np.pi
    week_angle = (timestamp%604800) / 604800 * 2 * np.pi
    year_seconds = 365.25 * 86400
    year_angle = (timestamp % year_seconds)/year_seconds*2*np.pi
    
    sin_sum = (
        weights['hour']*np.sin(hour_angle)+
        weights['day']*np.sin(day_angle)+
        weights['week']*np.sin(week_angle)+
        weights['year']*np.sin(year_angle)
    )
    cos_sum = (
        weights['hour']*np.cos(hour_angle)+
        weights['day']*np.cos(day_angle)+
        weights['week']*np.cos(week_angle)+
        weights['year']*np.cos(year_angle)
    )
    final_angle = (np.arctan2(sin_sum,cos_sum)+np.pi)/(2*np.pi)
    return final_angle

def feed_one(m):
    global head
    val = json.loads(m.value().decode('utf-8'))
    timestamp = int(val['timestamp'] * 1e9)

    parts = m.topic().split('-')
    if len(parts)== 2:
        x_buf[head] = [val[k] for k in cols]
        t_buf[head] = angle_encoding(timestamp)
        _index_values[head] = timestamp

        head = (head + 1) % ROWS
        return False

    elif len(parts)== 3:
        l_n = parts[2]
        idx = n_head[l_n]

        n_buf[l_n][idx] = [val[k] for k in cols]
        n_t_buf[l_n][idx] = angle_encoding(timestamp)
        n_index_ns[l_n][index] = timestamp

        n_head[l_n] = (idx + 1) % DATA_N[l_n]
        return l_n
    else:
        print("error in topic value")

try:
    batch = consumer.consume(num_messages=ROWS+sum(DATA_N.values()), timeout=-1)
    for msg in batch:
        feed_one(msg)
    
    msg = consumer.poll(timeout=10)
    while msg is not None:
        feed_one(msg)
        msg = consumer.poll(timeout=1)

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(int(index_ns[(head-1) % ROWS]/1000000000)))}\tFirst batch: {len(batch)}")

    while True:
        msg = consumer.poll(timeout=INTERVAL+5)
        if msg is None:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t No Message in {INTERVAL+5}")
        elif msg.error():
            print("Error: ",msg.error())
        else:
            n_res = feed_one(msg)
            if not n_res:
                order = (order_tem + head) % ROWS
                x_view = base_df.take(order)
                with redirect_stdout(buf), redirect_stderr(buf):
                    dfs = data_transform(x_view,tf_config,0)
                if DEBUG:
                    for tf, df in dfs.items():
                        if df.isnull().values.any():
                            print(f"[DEBUG] NaN 발견: dfs['{tf}'] 전체에 NaN이 {df.isnull().values.sum()}개 있음")
                
                t_view = { k:time_df.loc[dfs[k].index] for k in dfs }
                if DEBUG:
                    for tf, df in t_view.items():
                        if df.isnull().values.any():
                            print(f"[DEBUG] NaN 발견: t_view['{tf}'] 전체에 NaN이 {df.isnull().values.sum()}개 있음")


                n_x_view = {k: n_base_df[k].take(n_order[k]) for k in DATA_N }
                n_t_view = {k: n_time_df[k].take(n_order[k]) for k in DATA_N }
                with redirect_stdout(buf), redirect_stderr(buf):
                    n_dfs = data_transform(n_x_view,n_config,2)
                if DEBUG:
                    for tf, df in n_dfs.items():
                        if df.isnull().values.any():
                            print(f"[DEBUG] NaN 발견: n_dfs['{tf}'] 전체에 NaN이 {df.isnull().values.sum()}개 있음")

                with redirect_stdout(buf), redirect_stderr(buf):
                    dfs = data_scale(dfs|n_dfs,sc_config,save=False,path=f'{data_folder}{sc_config.get("path")}/{TICK}/')
                if DEBUG:
                    for tf, df in dfs.items():
                        if df.isnull().values.any():
                            print(f"[DEBUG] NaN 발견: scaled dfs['{tf}'] 전체에 NaN이 {df.isnull().values.sum()}개 있음")
                    print(dfs,x_view,n_x_view)
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t apply ttrade")
                ttrade(dfs,t_view|n_t_view)
                
                buf.truncate(0)
                buf.seek(0)
            else:
                n_order[n_res] = (_n_order[n_res] + n_head[n_res]) % DATA_N[n_res]
finally:
    consumer.close()
