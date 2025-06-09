from confluent_kafka import Producer, Consumer
import numpy as np
import requests
import time
import json
import os

kafka_host=os.environ.get('KAFKA_SERVICE','my-cluster-kafka-bootstrap.kafka.svc:9092')
TICK = os.environ['TICK']
TOPIC = os.environ['TOPIC']
RANGE = int(os.environ['RANGE'])
DATA = int(os.environ.get('DATA',100))

kf=Producer({
    'bootstrap.servers': kafka_host,
    'compression.type': 'gzip',
    'acks': '0',
    'client.id': f'{TOPIC}-{RANGE}',
    "batch.size": 0,
    "linger.ms": 0
})

def request_data(tick,to_time='',count=200):
    headers = {"accept":"application/json"}
    url=f'https://api.upbit.com/v1/candles/minutes/1?market={tick}&count={count}{to_time}'
    response=requests.get(url, headers=headers)
    return json.loads(response.text)

def kf_message(topic,message):
    key = str(message['timestamp'])
    try:
        message = json.dumps(message).encode('utf-8')
        kf.produce(topic,key=key.encode('utf-8'), value=message)
        kf.poll(0)
        kf.flush()
    except Exception as e:
        print(f"Failed to send message: {str(e)}")

def range_minute():
    global DATA
    cols = ['opening_price', 'high_price', 'low_price']
    _ohl = np.zeros((RANGE,3), dtype=np.int64)
    _c = 0
    _v = np.zeros(RANGE,dtype=np.float64)
    head = 0

    now = time.time() // 60 * 60
    end_time = int(now) - ((DATA+RANGE+10) * 60)
    end_string = time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(end_time))
    batch = request_data(TICK,'&to='+end_string)
    ts_head = time.mktime(time.strptime(batch[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S'))
    data_sum = 0 

    print(f'{TOPIC}-{RANGE}')
    
    while DATA > 0:
        for tick in reversed(batch):
            date_time_utc = time.strptime(tick['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
            ts_tick = time.mktime(date_time_utc)

            if ts_tick < ts_head:
                continue
            elif ts_tick == ts_head:
                _ohl[head] = [tick[k] for k in cols]
                _v[head] = tick['candle_acc_trade_volume']
                _c = tick['trade_price']
            else:
                _ohl[head] = [_c,_c,_c]
                _v[head] = 0
            
            ts_head += 60
            head = (head + 1) % RANGE
            data_sum += 1
        
            if data_sum > RANGE:
                open = int(_ohl[(head+1) % RANGE][0])
                high = int(_ohl[:,1].max())
                low = int(_ohl[:,2].min())
                close = int(_c)
                volume = float(_v.sum())
                kf_message(f'{TOPIC}-{RANGE}',message={'tick':TICK,'timestamp':ts_head,'open':open,'low':low,'high':high,'close':close,'value':volume})
                DATA -= 1
            
        time.sleep(3)
        end_string = time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(ts_tick+200*60))
        batch = request_data(TICK,'&to='+end_string)
        print(f"{end_string} get new batch")

    consumer = Consumer({
        'bootstrap.servers' : kafka_host,
        'group.id' : f'{TOPIC}-{RANGE}',
        'auto.offset.reset' : 'earliest',
        'enable.auto.commit' : False,
    })
    consumer.subscribe([TOPIC])

    print()
    try:
        while True:
            msg = consumer.poll(timeout=60)
            if msg is None:
                print("No message received, continuing...")
                continue
            val = json.loads(msg.value().decode('utf-8'))
            if int(val['timestamp']) >= ts_head:
                _ohl[head] = [val['open'],val['high'],val['low']]
                _c = val['close']
                _v[head] = val['value']

                print(val)
                ts_head += 60
                head = (head + 1) % RANGE

                open = int(_ohl[(head+1) % RANGE][0])
                high = int(_ohl[:,1].max())
                low = int(_ohl[:,2].min())
                close = int(_c)
                volume = float(_v.sum())
                kf_message(f'{TOPIC}-{RANGE}',message={'tick':TICK,'timestamp':ts_head,'open':open,'low':low,'high':high,'close':close,'value':volume})
    finally:
        consumer.close()
