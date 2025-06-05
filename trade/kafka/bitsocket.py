import websockets
import asyncio
import json
import signal
import os

TIMEOUT = int(os.getenv("WS_TIMEOUT","240"))
MAX_RETRIES = int(os.getenv("WS_MAX_RETRIES","5"))
RETRY_SLEEP = int(os.getenv("WS_RETRY_SLEEP","5"))

async def upbit_ws_client(q, ticker):
    uri="wss://api.upbit.com/websocket/v1"
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with websockets.connect(uri,ping_interval=TIMEOUT // 2) as websocket:
                print(f"WebSocket connected → {ticker}")
                subscribe_fmt=[
                    {"ticket": "test"},
                    {
                        "type": "ticker",
                        "codes": [ticker],
                        "is_only_realtime":True
                    },
                    {"format": "SIMPLE"}
                ]
                subscribe_data=json.dumps(subscribe_fmt)
                await websocket.send(subscribe_data)
                while True:
                    try:
                        data=await asyncio.wait_for(websocket.recv(),timeout=TIMEOUT)
                        data=json.loads(data)
                        data=(data['tp'],data['ttms']/1000.0,data['tv'])
                        q.put(data)
                    except asyncio.TimeoutError:
                        print(f"No data for {TIMEOUT} - closing connection")
                        raise RuntimeError("data timeout ")
        except (websockets.exceptions.ConnectionClosedError, RuntimeError, Exception) as e:
            retries += 1
            print(f"WebSocket error ({retries}/{MAX_RETRIES}): {e}. Retrying in {RETRY_SLEEP}")
            await asyncio.sleep(RETRY_SLEEP)
    print("Max retries exceeded – terminating container")
    os.kill(1, signal.SIGTERM)
    os.kill(1, signal.SIGKILL)
async def main(q,ticker):
    await upbit_ws_client(q,ticker=ticker)

def update_data(q,ticker):
    asyncio.run(main(q,ticker))