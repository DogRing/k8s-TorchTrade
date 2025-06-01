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
    'client.id': f'{TOPIC}-{RANGE}'
})

def request_data(tick,to_time='',count=200):
    headers = {"accept":"application/json"}
    url=f'https://api.upbit.com/v1/candles/minutes/1?market={tick}&count={count}{to_time}'
    response=requests.get(url, headers=headers)
    return json.loads(response.text)

def kf_message(topic,message):
    key = str(message['timestamp'])
    try:
        message = json.dump(message).encode('utf-8')
        kf.produce(topic,key=key.encode('utf-8'), value=serialized_message)
        kf.poll(0)
    except Exception as e:
        print(f"Failed to send message: {str(e)}")

def range_minute():
    cols = ['opening_price', 'high_price', 'low_price']
    _ohl = np.zeros((RANGE,3), dtype=np.int64)
    _c = 0
    _v = np.zeros(RANGE,dtype=np.float64)
    head = 0

    now = time.time() // 60 * 60
    end_time = int(now) - ((DATA+RANGE) * 60)
    end_string = time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(end_time))
    batch = request_data(TICK,'&to='+end_string)
    ts_head = time.mktime(time.strptime(batch[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S'))
    data_sum = 0 

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
                open = _ohl[(head-1) % RANGE][0]
                high = _ohl[:,1].max()
                low = _ohl[:,2].min()
                close = _c
                volume = _v.sum()
                kf_message(f'{TOPIC}-{RANGE}',message={'tick':TICK,'timestamp':ts_head,'open':open,'low':low,'high':high,'close':close,'value':volume})
                DATA -= 1
            
        kf.flush()
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

    try:
        msg = consumer.poll(timeout=60)
        while True:
            val = json.loads(msg.value().decode('utf-8'))
            if int(val['timestamp']) >= ts_head:
                print(val['timestamp'], ts_head)
                _ohl[head] = [val['open'],val['high'],val['low']]
                _v[head] = val['value']

                ts_head += 60
                head = (head + 1) % RANGE

                open = _ohl[(head-1) % RANGE][0]
                high = _ohl[:,1].max()
                low = _ohl[:,2].min()
                close = _c
                volume = _v.sum()
                kf_message(f'{TOPIC}-{RANGE}',message={'tick':TICK,'timestamp':ts_head,'open':open,'low':low,'high':high,'close':close,'value':volume})
            msg = consumer.poll(timeout=60)
    finally:
        consumer.close()
