import random
import operator
import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms


# 상대 강도지수 (기간동안 상승, 하강 변화량)
def RSI(df, window):
    diff = df['close'] - df['close'].shift(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def MACD(df, short=12, long=26, signal=9):
    exp1 = df['close'].ewm(span=short, adjust=False).mean()
    exp2 = df['close'].ewm(span=long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# 볼린저 밴드
# bw: 밴드폭, preb: 하한선 0, 상한선 1
def BB(df, window):
    middle = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window).std(ddof=0)
    upper = middle + 2 * std
    lower = middle - 2 * std
    bw = 4*std / middle
    perb = (df['close'] - lower) / 4*std
    return (bw, perb), (upper, middle, lower)

# 모멘텀
def Mmt(df, window):
    return (df['close'] / df['close'].shift(window)) * 100

# 추세지표
# def MACD(df, short=12, long=26, signal=9):
#     macd = df.ewm(span=short).mean() - df.ewm(span=long).mean()
#     macd_signal = macd.ewm(span=signal).mean()
#     return macd, macd_signal

def evaluate_strategy(individual,data,macd,macd_signal,rsi):
    macd_threshold = individual[0]
    rsi_buy = individual[1]
    rsi_sell = individual[2]
    position = 0
    balance = 1000
    for i in range(100,len(data)):
        macd_cross = (macd.iloc[i] > macd_signal.iloc[i] + macd_threshold) and (macd.iloc[i-1] <= macd_signal.iloc[i-1] + macd_threshold)
        rsi_buy_signal = (rsi.iloc[i] > rsi_buy)[0] and (rsi.iloc[i-1] <= rsi_buy)[0]
        
        macd_cross_down = (macd.iloc[i] < macd_signal.iloc[i] - macd_threshold) and (macd.iloc[i-1] >= macd_signal.iloc[i-1] - macd_threshold)
        rsi_sell_signal = (rsi.iloc[i] < rsi_sell)[0] and (rsi.iloc[i-1] >= rsi_sell)[0]

        if macd_cross and rsi_buy_signal and position == 0:
            position = balance / data.iloc[i]
            balance = 0
        elif macd_cross_down and rsi_sell_signal and position > 0:
            balance = position * data.iloc[i]*0.99
            position = 0
        if position > 0:
            balance = position * data.iloc[-1]*0.99
        return balance,

creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_macd",random.uniform,-0.5,0.5)
toolbox.register("attr_rsi",random.randint,20,80)
toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.attr_macd, toolbox.attr_rsi, toolbox.attr_rsi), n=1)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

toolbox.register("mate",tools.cxTwoPoint)
toolbox.register("mutate",tools.mutFlipBit,indpb=0.2)
toolbox.register("select",tools.selTournament,tournsize=3)

# data = df=pd.read_csv('/data/raw/KRW-BTC.csv',parse_dates=[0],index_col=[0])['close']
# sma_short,sma_long = calculate_indicators(data)
macd,macd_signal = MACD(data)
rsi = RSI(data, 14)

def main():
    toolbox.register("evaluate",evaluate_strategy,data=data,macd=macd,macd_signal=macd_signal,rsi=rsi)
    population = toolbox.population(n=50)
    ngen = 50
    algorithms.eaSimple(population,toolbox,cxpb=0.7,mutpb=0.2,ngen=ngen,verbose=True)
    best_individual = tools.selBest(population,k=1)[0]
    print("최적의 매개변수:",best_individual)
    print("최종 수익:",evaluate_strategy(best_individual,data,macd,macd_signal,rsi)[0])

main()