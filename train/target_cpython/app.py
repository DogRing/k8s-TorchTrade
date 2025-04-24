from local_values import tickers,raw_folder,target_folder
import ctypes
import pandas as pd
import numpy as np
import json
import os

feature = os.environ.get('TARGET','close')
c_file = os.environ.get('C_FILE','./libtarget.so')
function_name = os.environ.get('FUNCTION_NAME','pred_period')
function_args = os.environ.get('FUNCTION_ARGS','[0.0075]')
function_arg_types = os.environ.get('ARG_TYPES','["float"]')
return_type = os.environ.get('RETURN_TYPE','int')

args = json.loads(function_args)
dynamic_arg_types = json.loads(function_arg_types)
_dll = ctypes.cdll.LoadLibrary(c_file)
func = getattr(_dll,function_name)

arg_types = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32)
]

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
    df = df.resample(rule='min').first()
    df=df.ffill()

    x = df[feature].to_numpy(dtype=np.int32,copy=True).flatten()
    x_len = len(x)

    print(f"  Read df {tick} {feature} {x_len}")

    if return_type == 'int':
        y = np.zeros(x_len,dtype=np.int32)
        c_y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    if return_type == 'float':
        y = np.zeros(x_len,dtype=np.float32)
        c_y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float32))

    c_len = ctypes.c_int(x_len)
    c_x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    print(f"  start cpython")
    func(c_len,c_x,c_y,*converted_args)

    print(f"  end cpython")
    y = pd.DataFrame(list(c_y),columns=['period'])
    y.index = df.index

    y.to_csv(target_folder+tick+'.csv')