import random
import operator
import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms

def calculate_indicators(data,short_window,long_window):
    sma_short=data.rolling(window=short_window).mean()
    sma_long=data.rolling(window=long_window).mean()
    return sma_short,sma_long

def evaluate_strategy(individual,data):
    buy_threshold = individual[0]
    sell_threshold = individual[1]
    short_window = int(individual[2])
    long_window = int(individual[3])
    sma_short,sma_long = calculate_indicators(data,short_window,long_window)
    position = 0
    balance = 1000
    for i in range(long_window,len(data)):
        if sma_short.iloc[i] > sma_long.iloc[i] * (1 + buy_threshold) and position == 0:
            position = balance / data.iloc[i]
            balance = 0
        elif sma_short.iloc[i] < sma_long.iloc[i] * (1 - sell_threshold) and position > 0:
            balance = position * data.iloc[i]*0.99
            position = 0
        if position > 0:
            balance = position * data.iloc[-1]*0.99
        return balance,

creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float",random.uniform,-1.0,0.5)
toolbox.register("attr_int",random.randint,5,100)
toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.attr_float, toolbox.attr_float, toolbox.attr_int, toolbox.attr_int), n=1)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

toolbox.register("mate",tools.cxTwoPoint)
toolbox.register("mutate",tools.mutFlipBit,indpb=0.25)
toolbox.register("select",tools.selTournament,tournsize=2)

# raw_data = df=pd.read_csv('/data/raw/KRW-BTC.csv',parse_dates=[0],index_col=[0])['close']
# data = raw_data[-1500000:]

def main():
    toolbox.register("evaluate",evaluate_strategy,data=data)
    population = toolbox.population(n=700)
    ngen = 400
    algorithms.eaSimple(population,toolbox,cxpb=0.7,mutpb=0.2,ngen=ngen,verbose=True)
    best_individual = tools.selBest(population,k=1)[0]
    print("parameters:",best_individual)
    print("profit:",evaluate_strategy(best_individual,data)[0])

main()