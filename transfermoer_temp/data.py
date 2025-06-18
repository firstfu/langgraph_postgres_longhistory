# =============================================================================
# 天氣預報 Transformer 模型 - 詳細註解版
# =============================================================================

import warnings  # 警告控制
from datetime import datetime, timedelta  # 日期時間處理

import matplotlib.pyplot as plt  # 繪圖庫
import numpy as np  # 數值計算庫
import pandas as pd  # 數據處理庫
# 導入必要的庫
import torch  # PyTorch深度學習框架
import torch.nn as nn  # PyTorch神經網路模組
from sklearn.metrics import mean_absolute_error, mean_squared_error  # 評估指標
from sklearn.preprocessing import StandardScaler  # 數據標準化工具
from torch.utils.data import DataLoader, Dataset  # PyTorch數據載入工具
from transformers import AutoConfig  # 自動配置類
from transformers import AutoModel  # 自動模型類
from transformers import EarlyStoppingCallback  # 早停回調函數
from transformers import Trainer  # 訓練器類
from transformers import TrainingArguments  # HuggingFace Transformers庫; 訓練參數類

# warnings.filterwarnings('ignore')  # 忽略警告信息



# =============================================================================
# 1. 數據生成函數 - 創建模擬天氣數據
# =============================================================================
def generate_weather_data(days=1000):
    """
    生成模擬的天氣時間序列數據

    參數:
        days (int): 要生成的天數，默認1000天

    返回:
        pandas.DataFrame: 包含日期和天氣特徵的數據框
    """
    print(f"📅 正在生成 {days} 天的模擬天氣數據...")

    # 設置隨機種子確保結果可重現
    np.random.seed(42)

    # 創建日期範圍，從2020年1月1日開始，每日一個數據點
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    print(f"   日期範圍: {dates[0].strftime('%Y-%m-%d')} 到 {dates[-1].strftime('%Y-%m-%d')}")

    # 計算一年中的第幾天（1-365）
    day_of_year = dates.dayofyear

    # 生成季節性溫度模式
    # 基礎溫度20°C + 15°C的季節性波動
    # sin函數模擬季節變化，-80天偏移讓最冷的時候在冬天
    seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # 添加隨機噪聲，標準差為3°C
    noise = np.random.normal(0, 3, days)

    # 添加輕微的長期升溫趨勢（模擬氣候變化）
    trend = np.linspace(0, 2, days)  # 總體升溫2°C

    # 最終溫度 = 季節性 + 隨機噪聲 + 長期趨勢
    temperature = seasonal_temp + noise + trend

    # 生成濕度數據（0-100%）
    # 基礎濕度50% + 20%的季節性變化 + 隨機波動
    humidity = 50 + 20 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, days)
    humidity = np.clip(humidity, 0, 100)  # 限制在0-100%範圍內

    # 生成氣壓數據（hPa）
    # 基礎氣壓1013 hPa + 10 hPa的季節性變化 + 隨機波動
    pressure = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, days)

    # 生成風速數據（m/s）
    # 使用指數分布模擬風速（大多數時候風速較小，偶爾有大風）
    wind_speed = 5 + 3 * np.random.exponential(1, days)
    wind_speed = np.clip(wind_speed, 0, 30)  # 限制在0-30 m/s範圍內

    # 創建DataFrame存儲所有數據
    df = pd.DataFrame({
        'date': dates,           # 日期
        'temperature': temperature,  # 溫度
        'humidity': humidity,    # 濕度
        'pressure': pressure,    # 氣壓
        'wind_speed': wind_speed # 風速
    })

    print("✅ 天氣數據生成完成")
    print(f"   數據形狀: {df.shape}")
    print(f"   特徵列: {list(df.columns)}")

    return df



generate_weather_data(100)