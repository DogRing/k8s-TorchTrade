from kafka import KafkaProducer, KafkaConsumer, TopicPartition
import multiprocessing as mp
from bitsocket import update_data
import json
import time
import os

tick=os.environ['TICK']
topic=os.environ['TOPIC']
interval=int(os.environ.get('INTERVAL','15'))
kafka_host=os.environ.get('KAFKA_SERVICE','my-cluster-kafka-bootstrap.kafka.svc:9092')

kf=KafkaProducer(
    bootstrap_servers=[kafka_host],
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    acks=1,
    linger_ms=0,
    compression_type='gzip'
)

def get_last_kf(topic):
    cons = KafkaConsumer(
        bootstrap_servers=[kafka_host],
        value_deserializer=lambda x: json.loads(x.decode()),
        auto_offset_reset="latest",
        enable_auto_commit=False
    )
    tp = TopicPartition(topic,0)
    cons.assign([tp])
    cons.seek_to_end(tp)
    end = cons.end_offsets([tp])[tp]
    if end == 0:
        print(f"No message in topic {topic}")
        return None
    cons.seek(tp,end - 1)
    msg = next(cons)
    cons.close()
    return msg.value

def kf_message(topic,message):
    future=kf.send(topic,value=message)
    record_md = future.get(timeout=5)
    print(f"[SEND] {time.time()} | {message['timestamp']} partition={record_md.partition} offset={record_md.offset}")

def candle_interval():
    print('Working pod')
    q=mp.Queue()
    p=mp.Process(name="Producer",target=update_data,args=(q,tick,),daemon=True)
    p.start()
    interval_len=60//interval
    while q.empty(): continue 
    price,_,_=q.get()
    interval_data=[[price]*4+[0]]*interval_len
    now=time.time()
    start=time.localtime(now)
    print(f'Start at {start.tm_year}-{start.tm_mon}-{start.tm_mday} {start.tm_hour+9}:{start.tm_min}:{start.tm_sec} KST')
    now_interval=now-now%interval+interval
    try:
        last = get_last_kf(topic)
        if last:
            ts = last['timestamp'] + interval
            price = last['close']
            while ts < now_interval:
                kf_message(topic,message={'tick':tick,'timestamp':ts,'open':price,'low':price,'high':price,'close':price,'value':0})
                ts += interval
        time.sleep(now_interval-time.time())
        while True:
            index=int(now_interval%60/interval)
            if not q.empty():
                price,ttms,volume=q.get()
                interval_data[index]=[price]*4+[volume]
                while not q.empty():
                    price,ttms,volume=q.get()
                    if interval_data[index][1]>price: interval_data[index][1]=price
                    if interval_data[index][2]<price: interval_data[index][2]=price
                    interval_data[index][3]=price
            else:
                interval_data[index]=[interval_data[index-1][3]]*4+[0]
            
            open=interval_data[(index+1)%interval_len][3]
            close=interval_data[index][3]
            low=min(row[1] for row in interval_data)
            high=max(row[2] for row in interval_data)
            volume=sum(row[4] for row in interval_data)
            kf_message(topic,message={'tick':tick,'timestamp':now_interval,'open':open,'low':low,'high':high,'close':close,'value':volume})
            
            now_interval+=interval
            left_time=now_interval-time.time()
            if left_time<0: print("interval is too short")
            else: time.sleep(left_time)
    except Exception as e:
        print(f"Error message: {str(e)}")
    finally:
        kf.flush()
        kf.close()
