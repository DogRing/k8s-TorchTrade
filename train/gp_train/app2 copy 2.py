import random
import numpy as np
import pandas as pd
from deap import base,creator,tools,algorithms

# 코스캐스틱
def stochastic(df, k_window=14, d_window=3):
    low_min = df['low'].rolling(window=k_window).min()
    high_max = df['high'].rolling(window=k_window).max()
    df_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    df_d = df_k.rolling(window=d_window).mean()
    return df_k,df_d

def stochastic_signal(df,k_window=14,d_window=3,buy=20,sell=80):
    k,d = stochastic(df,k_window,d_window)
    signal = np.where((k > d)&(k > buy),1,0)
    signal = pd.Series(np.where((k < d)&(k > sell),-1,signal),index=df.index)
    return signal

def fitness_function(individual,dfs):
    sum = 1
    for df in dfs:
        df = df.copy()
        df['signal'] = stochastic_signal(df,
            individual.BB,
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
        performance = (1 + gain - 0.0025).cumprod()
        sum *= performance.iloc[-1]
    return (sum,) 

class Individual:
    def __init__(self, BB, rsi_period, buy_point, sell_point):
        self.BB = BB
        self.rsi_period = rsi_period
        self.buy_point = buy_point
        self.sell_point = sell_point
    def __str__(self):
        return f"BB({self.BB}), RSI({self.rsi_period}), Buy/Sell {self.buy_point}/{self.sell_point}"
    def __len__(self):
        return 4
    def __getitem__(self, key):
        return [self.BB, self.rsi_period, self.buy_point, self.sell_point][key]
    def __setitem__(self, key, value):
        if key == 0:
            self.BB = value
        elif key == 1:
            self.rsi_period = value
        elif key == 2:
            self.buy_point = value
        elif key == 3:
            self.sell_point = value
        else:
            raise IndexError("Index out of range")

def custom_crossover(ind1, ind2):
    child1, child2 = creator.Individual(ind1.BB, ind1.rsi_period, ind1.buy_point, ind1.sell_point), creator.Individual(ind2.BB, ind2.rsi_period, ind2.buy_point, ind2.sell_point)
    if random.random() < 0.5:
        child1.BB, child2.BB = child2.BB, child1.BB
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
        np.random.randint(4,24),    # BB
        np.random.randint(1,10),    # rsi_period
        np.random.randint(5,35),
        np.random.randint(60,100)
    )

def run_gp(dfs,population_size=50,generations=50):
    toolbox = base.Toolbox()
    toolbox.register("individual",create_individual)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)
    toolbox.register("evaluate",fitness_function,dfs=dfs)
    # toolbox.register("mate",tools.cxTwoPoint)
    toolbox.register("mate",custom_crossover)
    # toolbox.register("mutate",tools.mutUniformInt,low=[5,20,5,7],up=[30,50,20,30],indpb=0.2)
    toolbox.register("mutate",custom_mutation,low=[4,1,5,60],up=[24,10,35,100],indpb=0.2)
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


raw_data = pd.read_csv('/data/raw/KRW-ETC.csv',parse_dates=[0],index_col=[0])
data = raw_data[-1500000:]
btc_raw_data = pd.read_csv('/data/raw/KRW-BTC.csv',parse_dates=[0],index_col=[0])
btc_data = btc_raw_data[-1500000:]
doge_raw_data = pd.read_csv('/data/raw/KRW-DOGE.csv',parse_dates=[0],index_col=[0])
doge_data = doge_raw_data[-1500000:]
data_array = [data,btc_data,doge_data]

train_data =btc_raw_data[-2500000:-1000000]
test_data = btc_raw_data[-1000000:]

best_individual,logbook = run_gp(data_array,700,100)

print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_individual.fitness.values[0]}")
print(f"Final performance: {fitness_function(best_individual,[data])}")

fitness_function(best_individual,[data])
fitness_function(best_individual,[raw_data])
fitness_function(best_individual,[btc_data])
fitness_function(best_individual,[btc_raw_data])
fitness_function(best_individual,[doge_data])
fitness_function(best_individual,[doge_raw_data])

for gen in range(len(logbook)):
    print(f"Generation {gen}: {hof.items[0]}, Fitness: {hof.items[0].fitness.values}")