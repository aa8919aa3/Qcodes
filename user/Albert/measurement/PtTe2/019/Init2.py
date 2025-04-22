
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
        # print(f"計算微分電阻 dV/dI，使用電壓: {voltage_param} 和電流: {current_param}")
        
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
                
            # print("已添加dV_dI列到數據框")
        else:
            # 如果沒有其他參數，直接計算
            df = df.sort_values(by=current_param)
            voltage_values = df[voltage_param].values
            current_values = df[current_param].values
            df['dV_dI'] = np.gradient(voltage_values, current_values)
            # print("已添加dV_dI列到數據框")
    else:
        print("無法找到電流參數，跳過dV/dI計算")
    # 檢查數據框的結構
    # print(df.info())

    # 顯示數據框的基本統計信息
    print(df.describe())

    # 檢查數據框中的缺失值
    # print(df.isnull().sum())
    
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
    start_time_str = dataset.run_timestamp()
    completed_time_str = dataset.completed_timestamp()
    
    if start_time_str:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    else:
        start_time = None
    
    if completed_time_str:
        completed_time = datetime.strptime(completed_time_str, '%Y-%m-%d %H:%M:%S')
        run_time = completed_time - start_time
        display_time = _display_time(run_time)
    else:
        completed_time = None
        run_time = None
        display_time = "N/A"
    
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
        title=f"<b>#{runid} | {exp_name} |</b> {sample_name}",
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
            title=f"<b>#{runid}| {exp_name}(dV/dI) |</b> {sample_name}",
            xaxis_title=f"{x_param} ({param_units.get(x_param, '')})",
            yaxis_title=f"{y_param} ({param_units.get(y_param, '')})"
        )
    
    # 添加樣品信息到圖下方，確保所有註釋都能顯示
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M')
    sample_info = f"{timestamp_str} | Duration: {display_time}"
    annotations = [
        # 底部左側 - 時間信息
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
        # 左側中間 - Y軸信息
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
    print(f"R_fit: {R} Ω")

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
        # 底部左側 - 時間信息
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
        title=f'<b>#{runid} | {exp_name} | {sample_name} |</b> R<sub>fit</sub>: {R:.4f} Ω',
        xaxis_title=f"{x_param} ({param_units.get(x_param, '')})",
        yaxis_title=f"{y_param} ({param_units.get(y_param, '')})",
        yaxis2=dict(
            title=f"{dvdi_param} ({param_units.get(dvdi_param, '')})",
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        legend=dict(orientation="h",x=0.5, y=1,xanchor="center", yanchor="bottom",),
        annotations=annotation,
        width = 800,
        height = 800
    )
    
    # 顯示圖形
    fig.show()
    return fig

# db_path = search_and_initialise_db()
# qc.experiments()
