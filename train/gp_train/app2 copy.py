import random
import numpy as np
import pandas as pd
from deap import base,creator,tools,algorithms

def MACD(df, short=12, long=26, signal=9):
    macd = df['close'].ewm(span=short).mean() - df['close'].ewm(span=long).mean()
    macd_signal = macd.ewm(span=signal).mean()
    macd_oscillator = macd - macd_signal
    return  macd_oscillator, (macd, macd_signal)

def RSI(df, window=14):
    import numpy as np
    import pandas as pdㄷ
    diff = df['close'] - df['close'].shift(1)
    rise = pd.Series(np.where(diff >=0, diff, 0),index=df.index)
    fall = pd.Series(np.where(diff < 0, diff.abs(), 0),index=df.index)
    AU = rise.ewm(alpha=1/window, min_periods=window).mean()
    AD = fall.ewm(alpha=1/window, min_periods=window).mean()
    return AU / (AU+AD) * 100

def macd_rsi(df,macd_short=12,macd_long=26,signal=9,rsi_period=14,buy_point=30,sell_point=70):
    macd,_ = MACD(df,macd_short,macd_long,signal)
    rsi = RSI(df,rsi_period)
    signal = np.where((macd > 0)&(rsi > buy_point)&(rsi.shift(1) <=buy_point),1,0)
    signal = pd.Series(np.where((macd < 0)&(rsi <sell_point)&(rsi.shift(1) >= sell_point),-1,signal),index=df.index)
    return signal

def MFI(df,period=14):
    typical_price = (df['high']+df['low']+df['close']) / 3
    money_flow = typical_price * df['volume']
    price_diff = typical_price.diff()
    positive_flow = pd.Series(np.where(price_diff > 0,money_flow,0),index=df.index)
    negative_flow = pd.Series(np.where(price_diff < 0,money_flow,0),index=df.index)
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    mfr = positive_mf / negative_mf
    return 100 - (100 / (1 + mfr))

def macd_mfi(df,macd_short=12,macd_long=26,signal=9,mfi_period=14,buy_point=20,sell_point=80):
    macd,_ = MACD(df,macd_short,macd_long,signal)
    mfi = MFI(df,mfi_period)
    signal = np.where((macd > 0)&(mfi > buy_point)&(mfi.shift(1) <= buy_point),1,0)
    signal = pd.Series(np.where((macd < 0)&(mfi < sell_point)&(mfi.shift(1) >= sell_point),-1,signal),index=df.index)
    return signal

def fitness_function(individual,dfs):
    sum = 1
    for df in dfs:
        df = df.copy()
        df['signal'] = macd_rsi(df,
            individual.macd_short,
            individual.macd_long,
            individual.signal,
            individual.rsi_period,
            individual.buy_point,
            individual.sell_point
        )
        df = df[df['signal'] != 0].copy()
        mask = df['signal'] != df['signal'].shift()
        df['signal'] = df['signal'].where(mask,0)
        df = df[df['signal'] != 0].copy()
        if len(df) < 5:
            return (-1,)
        if df['signal'].iloc[-1] == 1:
            buy = df['close'][df['signal'] == 1].iloc[:-1].reset_index(drop=True)
        else:
            buy = df['close'][df['signal'] == 1].reset_index(drop=True)
        if df['signal'].iloc[0] == -1:
            sell = df['close'][df['signal'] == -1].iloc[1:].reset_index(drop=True)
        else:
            sell = df['close'][df['signal'] == -1].reset_index(drop=True)
        gain = ((sell - buy) / buy)
        gain[gain > 0.75] = 0.75
        performance = (1 + gain - 0.00175).cumprod()
        sum *= performance.iloc[-1]
    return (sum,) 

class Individual:
    def __init__(self, macd_short, macd_long, signal, rsi_period, buy_point, sell_point):
        self.macd_short = macd_short
        self.macd_long = macd_long
        self.signal = signal
        self.rsi_period = rsi_period
        self.buy_point = buy_point
        self.sell_point = sell_point
    def __str__(self):
        return f"MACD({self.macd_short},{self.macd_long},{self.signal}), RSI({self.rsi_period}), Buy/Sell {self.buy_point}/{self.sell_point}"
    def __len__(self):
        return 6
    def __getitem__(self, key):
        return [self.macd_short, self.macd_long, self.signal, self.rsi_period, self.buy_point, self.sell_point][key]
    def __setitem__(self, key, value):
        if key == 0:
            self.macd_short = value
        elif key == 1:
            self.macd_long = value
        elif key == 2:
            self.signal = value
        elif key == 3:
            self.rsi_period = value
        elif key == 4:
            self.buy_point = value
        elif key == 5:
            self.sell_point = value
        else:
            raise IndexError("Index out of range")

def custom_crossover(ind1, ind2):
    child1, child2 = creator.Individual(ind1.macd_short, ind1.macd_long, ind1.signal, ind1.rsi_period, ind1.buy_point, ind1.sell_point), creator.Individual(ind2.macd_short, ind2.macd_long, ind2.signal, ind2.rsi_period, ind2.buy_point, ind2.sell_point)
    if random.random() < 0.5:
        child1.macd_short, child2.macd_short = child2.macd_short, child1.macd_short
    if random.random() < 0.5:
        child1.macd_long, child2.macd_long = child2.macd_long, child1.macd_long
    if random.random() < 0.5:
        child1.signal, child2.signal = child2.signal, child1.signal
    if random.random() < 0.5:
        child1.rsi_period, child2.rsi_period = child2.rsi_period, child1.rsi_period
    if random.random() < 0.5:
        child1.buy_point, child2.buy_point = child2.buy_point, child1.buy_point
    if random.random() < 0.5:
        child1.sell_point, child2.sell_point = child2.sell_point, child1.sell_point
    return child1, child2

def custom_mutation(individual, indpb, low, up):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(low[i], up[i])
    return individual,

creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",Individual,fitness=creator.FitnessMax)

def create_individual():
    return creator.Individual(
        np.random.randint(1,30),    # short
        np.random.randint(10,70),   # long
        np.random.randint(1,25),    # signal
        np.random.randint(3,60),    # rsi_period
        np.random.randint(1,70),
        np.random.randint(10,100)
    )

def run_gp(dfs,population_size=50,generations=50):
    toolbox = base.Toolbox()
    toolbox.register("individual",create_individual)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)
    toolbox.register("evaluate",fitness_function,dfs=dfs)
    # toolbox.register("mate",tools.cxTwoPoint)
    toolbox.register("mate",custom_crossover)
    # toolbox.register("mutate",tools.mutUniformInt,low=[5,20,5,7],up=[30,50,20,30],indpb=0.2)
    toolbox.register("mutate",custom_mutation,low=[1,10,1,3,1,10],up=[30,70,25,60,70,100],indpb=0.2)
    toolbox.register("select",tools.selTournament,tournsize=3)
    population = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg",np.mean)
    stats.register("max",np.max)
    hof = tools.HallOfFame(1)
    population,logbook = algorithms.eaSimple(
        population,toolbox,
        cxpb=0.7,mutpb=0.2,
        ngen=generations,
        stats=stats,halloffame=hof,verbose=True
    )
    best_individual = tools.selBest(population,k=1)[0]
    return best_individual,logbook


raw_data = pd.read_csv('/data/raw/KRW-XRP.csv',parse_dates=[0],index_col=[0])
data = raw_data[-1500000:]
btc_raw_data = pd.read_csv('/data/raw/KRW-BTC.csv',parse_dates=[0],index_col=[0])
btc_data = btc_raw_data[-1500000:]
doge_raw_data = pd.read_csv('/data/raw/KRW-DOGE.csv',parse_dates=[0],index_col=[0])
doge_data = doge_raw_data[-1500000:]
data_array = [data,btc_data,doge_data]

best_individual,logbook = run_gp([data],1000,70)

print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_individual.fitness.values[0]}")
print(f"Final performance: {fitness_function(best_individual,[raw_data])}")

dd_individual = Individual(6,65,18,45,33,69)
fitness_function(dd_individual,[test_data])

fitness_function(best_individual,[data])
fitness_function(best_individual,[raw_data])
fitness_function(best_individual,[btc_data])
fitness_function(best_individual,[btc_raw_data])
fitness_function(best_individual,[doge_data])
fitness_function(best_individual,[doge_raw_data])

for gen in range(len(logbook)):
    print(f"Generation {gen}: {hof.items[0]}, Fitness: {hof.items[0].fitness.values}")