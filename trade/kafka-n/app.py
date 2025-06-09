import signal
import time
import threading
import os

from kazoo.client import KazooClient
from kazoo.retry import KazooRetry
from kazoo.recipe.election import Election
from candle import range_minute

running = True

def handle_signal(signum,frame):
    global running
    print("Received shutdown signal")
    running = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

election_path="/ttrade/"+os.environ['TICK']+os.environ['RANGE']
zookeeper_host=os.environ.get('ZK_SERVICE','zk-cs.zookeeper.svc:2181')

retry = KazooRetry(max_tries=-1, delay=0.5, max_delay=5.0)
zk=KazooClient(
    hosts=zookeeper_host,
    timeout=10.0,
    connection_retry=retry,
    command_retry=retry
)

def run_as_leader():
    print("I am the leader, starting candle task...")
    try:
        range_minute()
    except Exception as e:
        print(f"Leader task error: {e}")

def election_loop():
    election = Election(zk, election_path)
    print("Participating in election...")
    while running:
        election.run(run_as_leader)
        if running:
            print("Election finished or lost, retrying in 5 seconds...")
            time.sleep(5)

if __name__=="__main__":
    try:
        zk.start()
        t = threading.Thread(target=election_loop)
        t.start()
        while running:
            time.sleep(3)
    finally:
        print("Shutting down...")
        zk.stop()
        zk.close()
