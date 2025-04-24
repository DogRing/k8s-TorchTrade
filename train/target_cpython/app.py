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
    print(f"len: {len(y)}")
    print(f"target {target_folder}")
    
    try:
        y_df = pd.DataFrame({'period':np.ctypeslib.as_array(c_y,shape=(x_len,))})
        print("DataFrame - numpy 직접 변환")
        y_df.index = df.index
        y_df.to_csv(target_folder+tick+'.csv')
        print(f"CSV 저장: {target_folder+tick+'.csv'}")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        try:
            print("청크 단위 처리 시도...")
            CHUNK_SIZE = 100000
            
            with open(target_folder+tick+'.csv','w') as f:
                f.write('timestamp,period\n')
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

                print(f"chunk {i//CHUNK_SIZE+1} 처리 ({i}~{end_idx})")

            print("chunk 완")
        except Exception as chunk_error:
            print(f"chunk error: {str(chunk_error)}")
            raise



    # try:
    #     y = pd.DataFrame(list(c_y),columns=['period'])
    #     print("DataFrame 성공")
    # except Exception as e:
    #     print(f"DataFrame 생성 실패: {str(e)}")
    #     raise
    # try:
    #     y.index = df.index
    #     print("index 할당")
    # except ValueError as ve:
    #     print(f"인덱스 길이 불일치: y({len(y)}) vs df({len(df)})")
    #     raise
    # print(f"target {target_folder}")
    # try:
    #     y.to_csv(target_folder+tick+'.csv')
    #     print(f"{tick} CSV 저장 완료")
    # except PermissionError:
    #     print(f"권한 오류: {target_folder}")
    #     raise
    # except FileNotFoundError:
    #     print(f"경로 없음: {target_folder}")
    #     raise