import random
import operator
import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms

def calculate_indicators(data):
    sma_short=data.rolling(window=7).mean()
    sma_long=data.rolling(window=15).mean()
    return sma_short,sma_long

def evaluate_strategy(individual,data,sma_short,sma_long):
    buy_threshold = individual[0]
    sell_threshold = individual[1]
    position = 0
    balance = 1000
    for i in range(30,len(data)):
        if sma_short.iloc[i] > sma_long.iloc[i] * (1 + buy_threshold) and position == 0:
            position = balance / data.iloc[i]
            balance = 0
        elif sma_short.iloc[i] < sma_long.iloc[i] * (1 - sell_threshold) and position > 0:
            balance = position * data.iloc[i]
            position = 0
        if position > 0:
            balance = position * data.iloc[-1]
        return balance,

creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float",random.uniform,0,0.1)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_float,n=2)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

toolbox.register("mate",tools.cxBlend,alpha=0.5)
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.01,indpb=0.2)
toolbox.register("select",tools.selTournament,tournsize=3)

# data = df=pd.read_csv('/data/raw/KRW-BTC.csv',parse_dates=[0],index_col=[0])['close']
sma_short,sma_long = calculate_indicators(data)

def main():
    toolbox.register("evaluate",evaluate_strategy,data=data,sma_short=sma_short,sma_long=sma_long)
    population = toolbox.population(n=50)
    ngen = 100
    algorithms.eaSimple(population,toolbox,cxpb=0.7,mutpb=0.2,ngen=ngen,verbose=True)
    best_individual = tools.selBest(population,k=1)[0]
    print("최적의 매개변수:",best_individual)
    print("최종 수익:",evaluate_strategy(best_individual,data,sma_short,sma_long)[0])

main()