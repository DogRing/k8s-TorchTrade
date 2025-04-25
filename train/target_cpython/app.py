from local_values import tickers,raw_folder,target_folder
import ctypes
import pandas as pd
import numpy as np
import json
import os

feature = os.environ.get('TARGET','close')
c_file = os.environ.get('C_FILE','./libtarget.so')
function_name = os.environ.get('FUNCTION_NAME','price_barrier_volat')
function_args = os.environ.get('FUNCTION_ARGS','[600]')
function_arg_types = os.environ.get('ARG_TYPES','["int"]')
return_type = os.environ.get('RETURN_TYPE','int')
set_volatility = os.environ.get('VOLATILITY','False') == 'True'
volatility_per = os.environ.get('VOLATILITY_PER','[0.01,0.02,14]')

args = json.loads(function_args)
dynamic_arg_types = json.loads(function_arg_types)
if set_volatility:
    volat_min,volat_max,volat_window = json.loads(volatility_per)
_dll = ctypes.cdll.LoadLibrary(c_file)
func = getattr(_dll,function_name)

arg_types = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32)
]

if return_type == 'int':
    arg_types.append(ctypes.POINTER(ctypes.c_int32))
else:
    arg_types.append(ctypes.POINTER(ctypes.c_float))

if set_volatility: 
    arg_types.append(ctypes.POINTER(ctypes.c_float))

for type_name in dynamic_arg_types:
    if type_name == 'int':
        arg_types.append(ctypes.c_int)
    elif type_name == 'float':
        arg_types.append(ctypes.c_float)
func.arg_types = arg_types

converted_args = []
for i,arg in enumerate(args):
    if i < len(dynamic_arg_types):
        if dynamic_arg_types[i] == 'int':
            converted_args.append(ctypes.c_int(arg))
        elif dynamic_arg_types[i] == 'float':
            converted_args.append(ctypes.c_float(arg))

print(f"function: {function_name}")
print(f"args: {converted_args}")
print(f"arg tpye: {function_arg_types}")
print(f"return array type: {return_type}\n")

for tick in tickers:
    df = pd.read_csv(raw_folder+tick+'.csv',parse_dates=[0],index_col=[0])

    x = df[feature].to_numpy(dtype=np.int32,copy=True).flatten()
    x_len = len(x)

    print(f"  Read df {tick} {feature} {x_len}")

    if return_type == 'int':
        y = np.zeros(x_len,dtype=np.int32)
        c_y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    else:
        y = np.zeros(x_len,dtype=np.float32)
        c_y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    c_len = ctypes.c_int(x_len)
    c_x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    print(f"  start cpython")
    if set_volatility:
        print(f"  set volatility")
        returns = np.diff(np.log(x))
        volatility = np.zeros_like(x)
        for i in range(volat_window, x_len):
            weights = np.arange(1, volat_window+1)
            weighted_returns = returns[i-volat_window:i] * weights
            volatility[i] = np.std(weighted_returns)
        scaled_vol = (volatility - np.min(volatility)) / (np.max(volatility) - np.min(volatility) + 1e-8)
        volat = volat_min + scaled_vol * (volat_max - volat_min)
        print(f"first: {volat[0]} last: {volat[-1]}")
        c_vol = volat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        func(c_len,c_x,c_y,c_vol,*converted_args)
    else:
        func(c_len,c_x,c_y,*converted_args)
    
    try:
        y_df = pd.DataFrame({'period':np.ctypeslib.as_array(c_y,shape=(x_len,))})
        print("  Numpy-DataFrame complete")
        y_df.index = df.index
        y_df.to_csv(target_folder+tick+'.csv')
        print(f"save CSV: {target_folder+tick+'.csv'}")
    except Exception as e:
        print(f"error: {str(e)}")
        try:
            print("save with chunk...")
            CHUNK_SIZE = 100000
            
            with open(target_folder+tick+'.csv','w') as f:
                f.write(',period\n')
            for i in range(0,x_len, CHUNK_SIZE):
                end_idx = min(i + CHUNK_SIZE,x_len)
                chunk_size = end_idx - 1

                chunk_data = np.ctypeslib.as_array(
                    ctypes.cast(c_y + i, ctypes.POINTER(ctypes.c_int)),
                    shape=(chunk_size,)
                )

                chunk_df = pd.DataFrame({'period': chunk_data})
                chunk_df.index = df.index[i:end_idx]

                chunk_df.to_csv(target_folder+tick+'.csv',mode='a',header=False)

                print(f"chunk {i//CHUNK_SIZE+1} index ({i}~{end_idx})")

            print("Chunk-DataFrame complete")
        except Exception as chunk_error:
            print(f"chunk error: {str(chunk_error)}")
            raise
