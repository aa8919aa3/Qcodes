
import os
import sys
import time
import pint
import numpy as np
import pyppt as ppt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pprint import pprint
from datetime import datetime
from functools import lru_cache
from pdf2image import convert_from_path
from matplotlib import colormaps
from tqdm.notebook import tqdm
from IPython.display import clear_output

from scipy import stats, optimize
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


import qcodes as qc
from qcodes import Parameter, initialise_or_create_database_at, load_or_create_experiment, Measurement 
from qcodes.dataset.plotting import plot_dataset, plot_by_id
from qcodes.utils.metadata import diff_param_values
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter

print('Imported all modules, QCoDeS version:', qc.__version__, 'initialized')

def search_and_initialise_db(directory=None):
    """
    自動搜索目錄中的 .db 文件並初始化選定的數據庫
    
    Parameters:
    -----------
    directory : str, optional
        要搜索的目錄路徑，如果不提供則使用當前目錄的父目錄
        
    Returns:
    --------
    str
        初始化的數據庫路徑
    """
    # 如果沒有提供目錄，使用當前目錄的父目錄
    if directory is None:
        directory = os.path.dirname(os.getcwd())
    
    # 搜索 .db 文件
    db_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.db'):
                db_files.append(os.path.join(root, file))
    
    # 如果沒有找到 .db 文件
    if not db_files:
        print(f"在 {directory} 中找不到 .db 文件")
        return None
    
    # 顯示找到的 .db 文件
    print(f"找到 {len(db_files)} 個 .db 文件:")
    for i, db_file in enumerate(db_files):
        print(f"{i+1}. {db_file}")
    
    # 選擇要初始化的文件
    if len(db_files) == 1:
        selected_db = db_files[0]
        print(f"自動選擇唯一的 .db 文件: {selected_db}")
    else:
        while True:
            try:
                choice = int(input("輸入要初始化的 .db 文件編號: "))
                if 1 <= choice <= len(db_files):
                    selected_db = db_files[choice-1]
                    break
                else:
                    print(f"請輸入 1 到 {len(db_files)} 之間的數字")
            except ValueError:
                print("請輸入有效的數字")
    
    # 初始化選定的數據庫
    print(f"初始化數據庫: {selected_db}")
    initialise_or_create_database_at(selected_db)
    
    return selected_db

# Initialize pint unit registry
ureg = pint.UnitRegistry()
ureg.formatter.default_format = '~P'  # Use SI prefix formatting

class SI:
    """Class containing SI units for easy access and formatting."""
    A = ureg.ampere
    V = ureg.volt
    Ω = ureg.ohm
    F = ureg.farad
    H = ureg.henry
    W = ureg.watt
    J = ureg.joule
    s = ureg.second
    m = ureg.meter
    g = ureg.gram
    C = ureg.coulomb
    K = ureg.kelvin
    dB = ureg.decibel

    @staticmethod
    def f(value, unit):
        """Format a value or array with its unit using SI prefixes."""
        # Handle array formatting: convert each element separately
        if isinstance(value, (np.ndarray, np.ndarray)):
            if value.ndim == 0:
                quantity = float(value) * unit
                return f"{quantity.to_compact():.2f~P}"
            else:
                formatted_values = [f"{(float(v) * unit).to_compact():.2f~P}" for v in value]
                return ", ".join(formatted_values)
        else:  
            quantity = float(value) * unit
            return f"{quantity.to_compact():.2f~P}"
       

class FitResult:
    def __init__(self, fit_name, slope, intercept, r_value, p_value, stderr, mean_value, step_size, noise_std, SNR):
        self.fit_name = fit_name
        self.slope = slope
        self.intercept = intercept
        self.r_value = r_value
        self.p_value = p_value
        self.stderr = stderr
        self.mean_value = mean_value
        self.step_size = step_size
        self.noise_std = noise_std
        self.SNR = SNR
        
@lru_cache(maxsize=32)
def polyfit(fit_name, x, y):
    step_size = np.diff(x)[0].round(12)
    mean_value = np.mean(y)
    fit_result = stats.linregress(x, y)
    estimated_signal = fit_result.slope * x + fit_result.intercept
    noise = y - estimated_signal
    signal_power = np.mean(estimated_signal**2)
    noise_std = float(np.std(noise))
    SNR = 10 * np.log10(signal_power / noise_std**2)
    
    return FitResult(
        fit_name,
        fit_result.slope,
        fit_result.intercept,
        fit_result.rvalue,
        fit_result.pvalue,
        fit_result.stderr,
        mean_value,
        step_size,
        noise_std,
        SNR
    )
    
@lru_cache(maxsize=32)
def constfit(fit_name, x, y):
    step_size = np.diff(x)[0].round(12)
    mean_value = np.mean(y)
    fit_result = np.polyfit(x, y, 0)
    slope = 0
    intercept = fit_result[0]
    estimated_signal = np.polyval(fit_result, x)
    noise = y - estimated_signal
    stderr = np.std(y, ddof=1) 
    signal_power = np.mean(estimated_signal**2)
    noise_std = float(np.std(noise))
    SNR = 10 * np.log10(signal_power / noise_std**2)

    return FitResult(
        fit_name,
        slope,
        intercept,
        0,
        1,
        stderr,
        mean_value,
        step_size,
        noise_std,
        SNR
    )

def print_fit_result(fit_result):
    print(f"{fit_result.fit_name}:")
    print(f"{'Step size:':>20} {SI.f(fit_result.step_size, SI.A)}")
    print(f"{'Mean:':>20} {fit_result.mean_value:.6f}")
    print(f"{'Slope:':>20} {fit_result.slope:.6f}")
    print(f"{'Intercept:':>20} {fit_result.intercept:.6f}")
    print(f"{'R²:':>20} {fit_result.r_value**2*100:.4f}%")
    print(f"{'P-value:':>20} {fit_result.p_value:.2e}")
    print(f"{'σ(SD):':>20} {fit_result.stderr:.2e}")
    print(f"{'Noise SD:':>20} {fit_result.noise_std:.2e}")
    print(f"{'SNR:':>20} {fit_result.SNR:.2f} dB\n")
    
    
def plot_save(fig, id):
    # Save the plot as pdf file at current directory
    fig.write_image(f"{id}.pdf",scale=2)
    # Convert the pdf file to png file
    images = convert_from_path(f"{id}.pdf")
    for i, image in enumerate(images):
        image.save(f"{id}_{i}.png", "PNG", dpi=(1000, 1000))
    # Delete the pdf file
    os.remove(f"{id}.pdf")

@lru_cache(maxsize=32)
def cached_load_dataset(runid):
    """緩存數據集載入結果"""
    return qc.load_by_id(runid)

def info_df(runid):
    """
    輸入 runid，輸出數據集中的參數信息
    
    Parameters:
    runid : int
        要查看的數據集的 runid

    Returns:
    setpoint_params : list
        所有獨立參數的列表
        
    dependent_param : str
        依賴參數的名稱
        
    param_info : dict
        包含每個參數的起始值、結束值和步長的字典
        
    df : pd.DataFrame
        數據集的 pandas DataFrame
    """
    dataset = cached_load_dataset(runid)

    # Access the parameter data from the dataset
    param_data = dataset.get_parameter_data()

    # Assume the first key in param_data is the dependent parameter
    dependent_param = list(param_data.keys())[0]

    # Get all parameters associated with this dependent parameter
    all_params = list(param_data[dependent_param].keys())

    # Identify setpoint parameters (all parameters except the dependent one)
    setpoint_params = [param for param in all_params if param != dependent_param]
    
    # 創建參數信息字典
    param_info = {}

    # Print the number of points for each parameter
    print(f"Number of points for each parameter in dataset {runid}:")
    print(f"- {dependent_param} (dependent): {len(param_data[dependent_param][dependent_param])} points")
    
    # 記錄依賴參數的點數
    param_info[dependent_param] = {
        'points': len(param_data[dependent_param][dependent_param]),
        'type': 'dependent'
    }

    # Print details for each setpoint parameter
    for setpoint in setpoint_params:
        unique_values = np.unique(param_data[dependent_param][setpoint])
        num_unique_points = len(unique_values)
        
        start = unique_values[0]  
        end = unique_values[-1]   
        
        if num_unique_points > 1:
            step_size = np.diff(unique_values).mean()
        else:
            step_size = None
            
        # 添加到參數信息字典
        param_info[setpoint] = {
            'unique_points': num_unique_points,
            'start': start,
            'end': end,
            'step_size': step_size,
            'type': 'setpoint'
        }
        
        # 打印信息
        step_size_str = f"{step_size:.2e}" if step_size is not None else "N/A (only one unique point)"
        print(f"- {setpoint:<16} (setpoint): {num_unique_points} unique points, from {start:.2e} to {end:.2e}, step size: {step_size_str}")
    
    # 獲取pandas數據框
    df = dataset.to_pandas_dataframe().reset_index()
    
    # 尋找電壓和電流參數
    voltage_param = dependent_param  # 假設依賴參數是電壓
    current_param = next((param for param in setpoint_params if "curr" in param.lower()), None)
    
    if current_param is not None and voltage_param is not None:
        print(f"計算微分電阻 dV/dI，使用電壓: {voltage_param} 和電流: {current_param}")
        
        # 為每個唯一的其他參數組合計算dV/dI
        other_params = [p for p in setpoint_params if p != current_param]
        
        if other_params:
            # 需要按其他參數分組計算dV/dI
            df['dV_dI'] = np.nan  # 初始化dV/dI列
            
            # 獲取每個其他參數的唯一值
            groupby_columns = other_params
            for group_name, group_df in df.groupby(groupby_columns):
                # 按電流排序
                sorted_group = group_df.sort_values(by=current_param)
                
                # 計算dV/dI
                voltage_values = sorted_group[voltage_param].values
                current_values = sorted_group[current_param].values
                
                # 使用np.gradient計算微分電阻
                dv_di = np.gradient(voltage_values, current_values)
                
                # 將計算結果填入原始DataFrame中對應的行
                df.loc[sorted_group.index, 'dV_dI'] = dv_di
                
            print("已添加dV_dI列到數據框")
        else:
            # 如果沒有其他參數，直接計算
            df = df.sort_values(by=current_param)
            voltage_values = df[voltage_param].values
            current_values = df[current_param].values
            df['dV_dI'] = np.gradient(voltage_values, current_values)
            print("已添加dV_dI列到數據框")
    else:
        print("無法找到電流參數，跳過dV/dI計算")
    # 檢查數據框的結構
    print(df.info())

    # 顯示數據框的基本統計信息
    print(df.describe())

    # 檢查數據框中的缺失值
    print(df.isnull().sum())
    
    # 返回原始的參數、數據框和參數信息
    return setpoint_params, dependent_param, param_info, df

def _display_time(run_time):
    total_seconds = int(run_time.total_seconds())

    if total_seconds < 60:
        display_time = f"{total_seconds} sec"
    elif total_seconds < 3600:
        minutes, seconds = divmod(total_seconds, 60)
        display_time = f"{minutes} min {seconds} sec" if seconds else f"{minutes} min"
    else:
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        display_time = f"{hours} hr {minutes} min" if minutes else f"{hours} hr"
    return display_time 

def get_dataset_info(runid):
    """
    獲取數據集信息，包括每個參數的單位
    
    參數:
        runid: 運行ID
        
    返回:
        exp_name: 實驗名稱
        sample_name: 樣品名稱
        param_units: 包含參數名稱和對應單位的字典
        start_time: 開始時間
        completed_time: 完成時間
        display_time: 格式化的運行時間
    """
    dataset = cached_load_dataset(runid)
    exp_name = dataset.exp_name
    sample_name = dataset.sample_name
    description = dataset.description
    
    # 提取參數單位
    param_units = {}
    
    # 從 RunDescriber 中提取參數單位
    try:
        # 獲取依賴參數的單位
        for param in description.interdeps.dependencies:
            param_units[param.name] = param.unit
        
        # 獲取獨立參數的單位
        for param_tuple in description.interdeps.dependencies.items():
            dependent_param = param_tuple[0]
            independent_params = param_tuple[1]
            for param in independent_params:
                param_units[param.name] = param.unit
    except Exception as e:
        print(f"提取參數單位時發生錯誤: {e}")
    
    # 處理時間信息
    from datetime import datetime
    start_time = datetime.strptime(dataset.run_timestamp(), '%Y-%m-%d %H:%M:%S')
    completed_time = datetime.strptime(dataset.completed_timestamp(), '%Y-%m-%d %H:%M:%S')
    run_time = completed_time - start_time
    display_time = _display_time(run_time)
    
    label_width = 19  # Adjust this to control alignment

    output = f"ID{' ' * (label_width - 4)}: {runid}\n"
    output += f"Type{' ' * (label_width - 6)}: {exp_name}\n"
    output += f"Sample{' ' * (label_width - 8)}: {sample_name}\n"
    output += f"Run time{' ' * (label_width - 10)}: {display_time}\n"
    output += f"Parameter unit{' ' * (label_width - 20)}: \n"

    for param, unit in param_units.items():
        output += f"- {param:<{label_width - 4}}: {unit}\n"
    print(output)
        
    return exp_name, sample_name, param_units, start_time, completed_time, run_time, display_time


def plot_heatmaps(runid):
    """
    自動繪製電壓和微分電阻的熱圖，使用從數據集獲取的參數單位。
    
    參數:
        runid: 運行ID
    
    返回:
        fig1, fig2: 電壓熱圖和微分電阻熱圖
    """
    # 獲取數據集信息和參數單位
    exp_name, sample_name, param_units, start_time, completed_time, run_time, display_time = get_dataset_info(runid)
    
    # 獲取數據框
    setpoint_params, dependent_param, param_info, df = info_df(runid)
    
    # 獲取參數名稱
    params = list(df.keys())
    
    # 檢查參數
    if len(params) < 3:
        raise ValueError("DataFrame 必須至少包含 3 個列 (x, y, z)")
    
    # 使用原始參數名稱
    x_param = params[0]  # y_field
    y_param = params[1]  # appl_current
    v_param = params[2]  # meas_voltage_K2
    
    # 如果有第四個參數，假設為 dV/dI
    dvdi_param = params[3] if len(params) >= 4 else None
    
    

    # 使用從數據集獲取的單位，如果未找到則使用空字符串
    if dvdi_param and dvdi_param not in param_units:
        param_units[dvdi_param] = 'Ω'
    
    # 為軸標題添加點數和步長信息
    x_info = ""
    y_info = ""
    
    # 為 X 軸添加信息
    if x_param in param_info and param_info[x_param]['type'] == 'setpoint':
        points = param_info[x_param]['unique_points']
        step = param_info[x_param]['step_size']
        step_str = f"{step:.2e}" if step is not None else "N/A"
        x_info = f"{points} pts           <br>step={step_str}"
    
    # 為 Y 軸添加信息
    if y_param in param_info and param_info[y_param]['type'] == 'setpoint':
        points = param_info[y_param]['unique_points']
        step = param_info[y_param]['step_size']
        step_str = f"{step:.2e}" if step is not None else "N/A"
        y_info = f"{points} pts           <br>step={step_str}"
    
    # 創建電壓熱圖
    fig1 = go.Figure(data=go.Heatmap(
        x=df[x_param],
        y=df[y_param],
        z=df[v_param],
        colorscale='RdBu',
        colorbar=dict(
        title=dict(
            text=f"{v_param} ({param_units.get(v_param, '')})",
            side='right'))
    ))
    
    # 添加標題和軸標籤
    fig1.update_layout(
        title=f"<b>#{runid}|{exp_name}|</b> {sample_name}",
        xaxis_title=f"{x_param} ({param_units.get(x_param, '')})",
        yaxis_title=f"{y_param} ({param_units.get(y_param, '')})"
    )
    
    # 創建微分電阻熱圖（如果有）
    fig2 = None
    if dvdi_param:
        fig2 = go.Figure(data=go.Heatmap(
            x=df[x_param],
            y=df[y_param],
            z=df[dvdi_param],
            colorscale='RdBu',
            colorbar=dict(
                title=dict(
                    text=f"{dvdi_param} ({param_units.get(dvdi_param, '')})",
                    side="top" ))
        ))
        
        # 添加標題和軸標籤
        fig2.update_layout(
            title=f"<b>#{runid}|{exp_name}(dV/dI)|</b> {sample_name}",
            xaxis_title=f"{x_param} ({param_units.get(x_param, '')})",
            yaxis_title=f"{y_param} ({param_units.get(y_param, '')})"
        )
    
    # 添加樣品信息到圖下方，確保所有註釋都能顯示
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M')
    sample_info = f"{timestamp_str} | Duration: {display_time}"
    annotations = [
        # 上方左側 - 時間信息
        dict(
            text=sample_info,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,  # 略微偏離左邊緣
            y=1,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=10)
        ),
        # 底部右邊 - X軸信息
        dict(
            text=x_info,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1,
            y=-0.03,
            xanchor="right", 
            yanchor="top",
            font=dict(size=10)
        ),
        # 左側上方 - Y軸信息
        dict(
            text=y_info,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=-0.05,
            y=1,
            xanchor="right",
            yanchor="top",
            textangle=-90,
            font=dict(size=10)
        )
    ]
    
    # 更新兩個圖的佈局，增加邊距以顯示註釋
    fig1.update_layout(
        annotations=annotations,
        width=800,
        height=800,

    )
    
    if fig2:
        fig2.update_layout(
            annotations=annotations,
            width=800,
            height=800,
        )
    
    # 顯示圖形
    fig1.show()
    if fig2:
        fig2.show()
    
    return fig1, fig2

def plot_combined_iv_dvdi(runid):
    """
    在同一個圖表上繪製 I-V 曲線和微分電阻曲線，使用左右兩個 y 軸
    
    參數:
        df: 包含測量數據的 DataFrame，必須包含 appl_current, meas_voltage_K2 和 dV_dI 列
    
    返回:
        fig: 包含 I-V 和微分電阻曲線的圖
    """
    setpoint_params, dependent_param, param_info, df = info_df(runid)
    
    # 獲取數據集信息和參數單位
    exp_name, sample_name, param_units, start_time, completed_time, run_time, display_time = get_dataset_info(runid)
    # 獲取參數名稱
    params = list(df.keys())
    
    # 檢查參數
    if len(params) < 3:
        raise ValueError("DataFrame 必須至少包含 3 個列 (x, y, z)")
    
    # 使用原始參數名稱
    x_param = params[0]  # appl_current
    y_param = params[1]  # meas_voltage_K2
    
    # 如果有第四個參數，假設為 dV/dI
    dvdi_param = params[2] if len(params) >= 3 else None
    
     # 使用從數據集獲取的單位，如果未找到則使用空字符串
    # 對於 dV_dI，由於它是計算得出的，可能不在原始參數中，所以提供預設值 'Ω'
    if dvdi_param and dvdi_param not in param_units:
        param_units[dvdi_param] = 'Ω'
    
    R = np.polyfit(df[x_param],df[y_param],1)[0]
    
    # 為軸標題添加點數和步長信息
    x_info = ""
    
    # 為 X 軸添加信息
    if x_param in param_info and param_info[x_param]['type'] == 'setpoint':
        points = param_info[x_param]['unique_points']
        step = param_info[x_param]['step_size']
        step_str = f"{step:.2e}" if step is not None else "N/A"
        x_info = f"{points} pts           <br>step={step_str}"

    # 創建包含兩個 y 軸的圖
    fig = go.Figure()
    
    # 添加 I-V 曲線 (左 y 軸)
    fig.add_trace(go.Scatter(
        x=df[x_param],
        y=df[y_param],
        mode='lines+markers',
        name='I-V',
        marker=dict(size=5),
        line=dict(width=2)
    ))
    
    # 添加微分電阻曲線 (右 y 軸)
    fig.add_trace(go.Scatter(
        x=df[x_param],
        y=df[dvdi_param],
        mode='lines+markers',
        name='dV/dI',
        marker=dict(size=5),
        line=dict(width=2),
        yaxis='y2'  # 使用第二個 y 軸
    ))
    
    # 添加樣品信息到圖左下角
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M')
    sample_info = f"{timestamp_str} | Duration: {display_time}"
    annotation=[
        # 上方左側 - 時間信息
        dict(
            text=sample_info,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,  # 略微偏離左邊緣
            y=1,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=10)
        ),
            # 底部中間 - X軸信息
        dict(
            text=x_info,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1,
            y=-0.03,
            xanchor="right", 
            yanchor="top",
            font=dict(size=10)
        ),
        ]
    
    # 設置布局，包含兩個 y 軸
    fig.update_layout(
        title=f'<b>#{runid}|{exp_name}|</b>{sample_name}|R<sub>fit</sub>: {R:.4f} Ω',
        xaxis_title=f"{x_param} ({param_units.get(x_param, '')})",
        yaxis_title=f"{y_param} ({param_units.get(y_param, '')})",
        yaxis2=dict(
            title=f"{dvdi_param} ({param_units.get(dvdi_param, '')})",
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        legend=dict(orientation="h", xref="paper", yref="paper",x=0.5, y=1,xanchor="center", yanchor="bottom",),
        annotations=annotation,
        width = 800,
        height = 800
    )
    
    # 顯示圖形
    fig.show()
    return fig

def determine_scan_type(df, x_param):
    """
    Determine if the IV scan is unidirectional or bidirectional.
    
    Args:
        df (pd.DataFrame): DataFrame containing the IV data.
        x_param (str): Name of the current parameter.
    
    Returns:
        bool: True if the scan is unidirectional, False otherwise.
    """
    min_current = df[x_param].min()
    max_current = df[x_param].max()
    is_unidirectional = abs(min_current) < abs(max_current) * 0.05
    return is_unidirectional

def find_peaks_in_dvdi(df, x_param, dvdi_param):
    """
    Find peaks in the dV/dI curve for positive and negative currents.
    
    Args:
        df (pd.DataFrame): DataFrame containing the IV data.
        x_param (str): Name of the current parameter.
        dvdi_param (str): Name of the dV/dI parameter.
    
    Returns:
        positive_peak_indices (pd.Index): Indices of positive peaks.
        negative_peak_indices (pd.Index): Indices of negative peaks.
        current_0_index (int): Index of the current closest to zero.
    """
    positive_indices = df.index[df[x_param] > 0]
    negative_indices = df.index[df[x_param] < 0]
    
    positive_peaks, _ = find_peaks(df[dvdi_param].loc[positive_indices])
    positive_peak_indices = positive_indices[positive_peaks]
    
    negative_peaks, _ = find_peaks(df[dvdi_param].loc[negative_indices])
    negative_peak_indices = negative_indices[negative_peaks]
    
    current_0_index = df[x_param].abs().idxmin()
    
    return positive_peak_indices, negative_peak_indices, current_0_index

def identify_critical_currents(df, x_param, dvdi_param, positive_peak_indices, negative_peak_indices, is_unidirectional):
    """
    Identify the critical currents Ic and Ir based on peaks in dV/dI.
    
    Args:
        df (pd.DataFrame): DataFrame containing the IV data.
        x_param (str): Name of the current parameter.
        dvdi_param (str): Name of the dV/dI parameter.
        positive_peak_indices (pd.Index): Indices of positive peaks.
        negative_peak_indices (pd.Index): Indices of negative peaks.
        is_unidirectional (bool): Whether the scan is unidirectional.
    
    Returns:
        Ic (float): Critical current for positive peaks.
        Ir (float): Retrapping current for negative peaks (if applicable).
    """
    if len(positive_peak_indices) > 0:
        Ic_idx = positive_peak_indices[np.argmax(df[dvdi_param].loc[positive_peak_indices])]
        Ic = df[x_param].loc[Ic_idx]
    else:
        Ic = df[x_param].max()
    
    if not is_unidirectional and len(negative_peak_indices) > 0:
        Ir_idx = negative_peak_indices[np.argmax(df[dvdi_param].loc[negative_peak_indices])]
        Ir = df[x_param].loc[Ir_idx]
    else:
        Ir = df[x_param].min()
    
    return Ic, Ir

def get_data_regions(df, x_param, Ic, Ir, is_unidirectional):
    """
    Split the DataFrame into regions for resistance fitting.
    
    Args:
        df (pd.DataFrame): DataFrame containing the IV data.
        x_param (str): Name of the current parameter.
        Ic (float): Critical current.
        Ir (float): Retrapping current.
        is_unidirectional (bool): Whether the scan is unidirectional.
    
    Returns:
        df_0 (pd.DataFrame): Data before Ir or low current threshold.
        df_1 (pd.DataFrame): Data after Ic.
        df_between (pd.DataFrame): Data between Ir and Ic.
    """
    if is_unidirectional:
        low_current_threshold = Ic * 0.2
        df_0 = df[df[x_param] < low_current_threshold]
        df_1 = df[df[x_param] > Ic]
        df_between = df[(df[x_param] > low_current_threshold) & (df[x_param] < Ic)]
    else:
        df_0 = df[df[x_param] < Ir]
        df_1 = df[df[x_param] > Ic]
        df_between = df[(df[x_param] > Ir) & (df[x_param] < Ic)]
    return df_0, df_1, df_between

def perform_linear_fit(df, x_param, y_param):
    """
    Perform a linear fit on the specified columns of the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_param (str): Name of the x parameter.
        y_param (str): Name of the y parameter.
    
    Returns:
        slope (float): Slope of the linear fit.
        intercept (float): Intercept of the linear fit.
    """
    if df.empty or len(df) < 2:
        return 0, 0
    fit = np.polyfit(df[x_param], df[y_param], 1)
    return fit[0], fit[1]

def analyze_iv_curve(df, x_param, y_param, dvdi_param='dV_dI'):
    """
    Analyze the IV curve to detect critical currents and calculate resistances.
    
    Args:
        df (pd.DataFrame): DataFrame containing the IV data.
        x_param (str): Name of the current parameter.
        y_param (str): Name of the voltage parameter.
        dvdi_param (str, optional): Name of the dV/dI parameter. Defaults to 'dV_dI'.
    
    Returns:
        dict: Analysis results including Ic, Ir, resistance fits, and peak information.
    """
    # Compute dV/dI if not present
    if dvdi_param not in df.columns:
        df[dvdi_param] = np.gradient(df[y_param], df[x_param])
    
    # Sort the DataFrame by x_param for consistent analysis
    df = df.sort_values(by=x_param).reset_index(drop=True)
    
    # Step 1: Determine scan type
    is_unidirectional = determine_scan_type(df, x_param)
    
    # Step 2: Find peaks in dV/dI
    positive_peak_indices, negative_peak_indices, current_0_index = find_peaks_in_dvdi(df, x_param, dvdi_param)
    
    # Step 3: Identify critical currents
    Ic, Ir = identify_critical_currents(df, x_param, dvdi_param, positive_peak_indices, negative_peak_indices, is_unidirectional)
    
    # Step 4: Define data regions
    df_0, df_1, df_between = get_data_regions(df, x_param, Ic, Ir, is_unidirectional)
    
    # Step 5: Perform linear fits
    R_fit0, intercept_0 = perform_linear_fit(df_0, x_param, y_param)
    R_fit1, intercept_1 = perform_linear_fit(df_1, x_param, y_param)
    R_fit_SC, intercept_SC = perform_linear_fit(df_between, x_param, y_param)
    
    # Step 6: Compute additional metrics
    R_fit = (R_fit0 + R_fit1) / 2
    IcRn = Ic * R_fit
    
    # Return comprehensive results
    return {
        'Ic': Ic,
        'Ir': Ir,
        'current_0_index': current_0_index,
        'R_fit': R_fit,
        'R_fit0': R_fit0,
        'R_fit1': R_fit1,
        'R_fit_SC': R_fit_SC,
        'IcRn': IcRn,
        'is_unidirectional': is_unidirectional,
        'fits': {
            'fit_0': (R_fit0, intercept_0),
            'fit_1': (R_fit1, intercept_1),
            'fit_between': (R_fit_SC, intercept_SC)
        },
        'peaks': {
            'positive_peak_currents': df[x_param].loc[positive_peak_indices].values,
            'positive_peak_R': df[dvdi_param].loc[positive_peak_indices].values,
            'negative_peak_currents': df[x_param].loc[negative_peak_indices].values,
            'negative_peak_R': df[dvdi_param].loc[negative_peak_indices].values
        }
    }

def plot_heatmaps2(runid):
    """
    自動繪製電壓、微分電阻和臨界電流 (Ic) 的熱圖，使用從數據集獲取的參數單位。
    
    參數:
        runid: 運行ID
    
    返回:
        fig1, fig2, fig3: 電壓熱圖、微分電阻熱圖和 Ic 圖
    """
    # 獲取數據集信息和參數單位
    exp_name, sample_name, param_units, start_time, completed_time, run_time, display_time = get_dataset_info(runid)
    
    # 獲取數據框
    setpoint_params, dependent_param, param_info, df = info_df(runid)
    
    # 獲取參數名稱
    params = list(df.keys())
    
    # 檢查參數
    if len(params) < 3:
        raise ValueError("DataFrame 必須至少包含 3 個列 (x, y, z)")
    
    # 使用原始參數名稱
    x_param = params[0]  # e.g., y_field
    y_param = params[1]  # e.g., appl_current
    v_param = params[2]  # e.g., meas_voltage_K2
    dvdi_param = params[3] if len(params) >= 4 else None  # e.g., dV_dI
    
    # 設置默認單位
    if dvdi_param and dvdi_param not in param_units:
        param_units[dvdi_param] = 'Ω'
    if y_param not in param_units:
        param_units[y_param] = 'A'  # Assume current unit for Ic
    
    # 為軸標題添加點數和步長信息
    x_info = ""
    y_info = ""
    
    # 為 X 軸添加信息
    if x_param in param_info and param_info[x_param]['type'] == 'setpoint':
        points = param_info[x_param]['unique_points']
        step = param_info[x_param]['step_size']
        step_str = f"{step:.2e}" if step is not None else "N/A"
        x_info = f"{points} pts           <br>step={step_str}"
    
    # 為 Y 軸添加信息
    if y_param in param_info and param_info[y_param]['type'] == 'setpoint':
        points = param_info[y_param]['unique_points']
        step = param_info[y_param]['step_size']
        step_str = f"{step:.2e}" if step is not None else "N/A"
        y_info = f"{points} pts           <br>step={step_str}"
    
    # 創建電壓熱圖 (fig1)
    fig1 = go.Figure(data=go.Heatmap(
        x=df[x_param],
        y=df[y_param],
        z=df[v_param],
        colorscale='RdBu',
        colorbar=dict(title=dict(text=f"{v_param} ({param_units.get(v_param, '')})", side='right'))
    ))
    fig1.update_layout(
        title=f"#{runid} | {exp_name} | {sample_name}",
        xaxis_title=f"{x_param} ({param_units.get(x_param, '')})",
        yaxis_title=f"{y_param} ({param_units.get(y_param, '')})"
    )
    
    # 創建微分電阻熱圖 (fig2)
    fig2 = None
    if dvdi_param:
        fig2 = go.Figure(data=go.Heatmap(
            x=df[x_param],
            y=df[y_param],
            z=df[dvdi_param],
            colorscale='RdBu',
            colorbar=dict(title=dict(text=f"{dvdi_param} ({param_units.get(dvdi_param, '')})", side="top"))
        ))
        fig2.update_layout(
            title=f"#{runid} | {exp_name} (dV/dI) | {sample_name}",
            xaxis_title=f"{x_param} ({param_units.get(x_param, '')})",
            yaxis_title=f"{y_param} ({param_units.get(y_param, '')})"
        )
    
    # 分析 IV 曲線並提取 Ic
    ic_values = []
    for name, group in df.groupby(x_param):  # Group by y_field or similar parameter
        analysis = analyze_iv_curve(group, x_param=y_param, y_param=v_param, dvdi_param=dvdi_param)
        ic_values.append({'param_value': name, 'Ic': analysis['Ic']})
    
    ic_df = pd.DataFrame(ic_values)
    
    # 創建 Ic 圖 (fig3) - 假設 Ic 隨 x_param (e.g., y_field) 變化
    fig3 = go.Figure(data=go.Scatter(
        x=ic_df['param_value'],
        y=ic_df['Ic'],
        mode='lines+markers',
        name='Ic',
        marker=dict(size=5),
        line=dict(width=2)
    ))
    fig3.update_layout(
        title=f"#{runid} | {exp_name} | Ic vs. {x_param} | {sample_name}",
        xaxis_title=f"{x_param} ({param_units.get(x_param, '')})",
        yaxis_title=f"Ic ({param_units.get(y_param, '')})"
    )
    
    # 添加樣品信息和註釋
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M')
    sample_info = f"{timestamp_str} | Duration: {display_time}"
    annotations = [
        # 上方左側 - 時間信息
        dict(
            text=sample_info,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,  # 略微偏離左邊緣
            y=1,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=10)
        ),
        # 底部右邊 - X軸信息
        dict(
            text=x_info,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1,
            y=-0.03,
            xanchor="right", 
            yanchor="top",
            font=dict(size=10)
        ),
        # 左側上方 - Y軸信息
        dict(
            text=y_info,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=-0.05,
            y=1,
            xanchor="right",
            yanchor="top",
            textangle=-90,
            font=dict(size=10)
        )
    ]
    
    # 更新佈局
    for fig in [fig1, fig2, fig3]:
        if fig:
            fig.update_layout(
                annotations=annotations,
                width=800,
                height=800
            )
    
    # 顯示圖形
    fig1.show()
    if fig2:
        fig2.show()
    fig3.show()
    
    return fig1, fig2, fig3

def get_snapshot_value(dataset, path):
    """
    Extract a value from the dataset's snapshot using a dot-separated path.
    
    Args:
        dataset: QCoDeS dataset object.
        path (str): Dot-separated path to the desired parameter (e.g., 'station.parameters.temperature.value').
    
    Returns:
        The value at the specified path, or None if not found.
    """
    keys = path.split('.')
    value = dataset.snapshot
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value

def IcFinder(runids, plot_against_param, label_param_path=None):
    """
    Plot multiple Ic curves from different QCoDeS datasets on a single figure for comparison.
    
    Args:
        runids: int or list of ints
            A single run ID or list of run IDs for the datasets to analyze.
        plot_against_param (str):
            The parameter to plot Ic against (e.g., 'y_field').
        label_param_path (str, optional):
            Path in the dataset snapshot to the parameter for labeling curves (e.g., 'station.parameters.temperature.value').
            If None, uses the run ID as the label.
    
    Returns:
        fig: plotly.graph_objects.Figure
            The generated plot with multiple Ic curves.
    """
    # Ensure runids is a list
    if not isinstance(runids, (list, tuple)):
        runids = [runids]
    
    fig = go.Figure()
    
    for runid in runids:
        try:
            # Load dataset and get DataFrame with parameter info
            setpoint_params, dependent_param, param_info, df = info_df(runid)
            exp_name, sample_name, param_units, start_time, completed_time, run_time, display_time = get_dataset_info(runid)
            
            # Verify the parameter to plot against exists
            if plot_against_param not in setpoint_params:
                print(f"Warning: '{plot_against_param}' not found in setpoint parameters for runid {runid}. Skipping.")
                continue
            
            # Identify current and voltage parameters
            current_param = next((param for param in setpoint_params if 'curr' in param.lower()), None)
            if current_param is None:
                print(f"Warning: No current parameter found in runid {runid}. Skipping.")
                continue
            voltage_param = dependent_param  # Typically voltage
            
            # Group data by plot_against_param and calculate Ic for each group
            ic_values = []
            param_values = []
            for name, group in df.groupby(plot_against_param):
                analysis = analyze_iv_curve(group, x_param=current_param, y_param=voltage_param, dvdi_param='dV_dI')
                ic_values.append(analysis['Ic'])
                param_values.append(name)
            
            # Determine the label for this curve
            dataset = qc.load_by_id(runid)
            if label_param_path:
                label_value = get_snapshot_value(dataset, label_param_path)
                if label_value is not None:
                    label = f"{label_param_path.split('.')[-1]}={label_value}"
                else:
                    label = f"runid={runid}"
            else:
                label = f"runid={runid}"
            
            # Add the curve to the plot
            fig.add_trace(go.Scatter(
                x=param_values,
                y=ic_values,
                mode='lines+markers',
                name=label,
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        except Exception as e:
            print(f"Error processing runid {runid}: {e}")
            continue
    
    # Set axis units (assuming consistent units across datasets)
    if runids:
        _, _, param_units, _, _, _, _ = get_dataset_info(runids[0])
        x_unit = param_units.get(plot_against_param, '')
        y_unit = param_units.get(current_param, '')  # Ic inherits current's unit
    else:
        x_unit = ''
        y_unit = ''
    
    # Customize the plot layout
    fig.update_layout(
        title=f'Critical Current (Ic) vs. {plot_against_param}',
        xaxis_title=f"{plot_against_param} ({x_unit})",
        yaxis_title=f"Ic ({y_unit})",
        template='plotly_white',
        legend_title='Conditions',
        font=dict(size=12),
        showlegend=True,
        width = 800,
        height = 800
    )
    
    fig.show()
    return fig


db_path = search_and_initialise_db()
qc.experiments()
