"""数据加载模块 — 附件1-8 Excel读取 + 24场景组织"""
import pandas as pd
import numpy as np
import os

# CRITICAL: 使用相对路径，不硬编码任何盘符路径
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _find_attachment(num):
    """在 DATA_DIR 中按编号查找附件文件，返回完整路径。"""
    prefix = f'附件{num}'
    files = os.listdir(DATA_DIR)
    for f in files:
        if f.startswith(prefix):
            return os.path.join(DATA_DIR, f)
    raise FileNotFoundError(f"未找到 {prefix} 开头的文件，目录: {DATA_DIR}")


def load_attachment1():
    """加载附件1：园区典型日常规电负荷标幺功率曲线 → DataFrame[hour, p_base_pu]"""
    path = _find_attachment(1)
    df = pd.read_excel(path, header=None)
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = ['hour_str', 'p_base_pu']
    df['p_base_pu'] = df['p_base_pu'].astype(float)
    df['hour'] = range(24)
    return df[['hour', 'p_base_pu']]


def load_attachment2():
    """加载附件2：典型日风电、光伏标幺功率表 → DataFrame[hour, p_wind_pu, p_pv_pu]"""
    path = _find_attachment(2)
    df = pd.read_excel(path, header=None)
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = ['hour_str', 'p_wind_pu', 'p_pv_pu']
    df['p_wind_pu'] = df['p_wind_pu'].astype(float)
    df['p_pv_pu'] = df['p_pv_pu'].astype(float)
    df['hour'] = range(24)
    return df[['hour', 'p_wind_pu', 'p_pv_pu']]


def load_attachment3():
    """加载附件3：园区6种场景的风电标幺功率表 → DataFrame[hour, scene_1..scene_n]"""
    path = _find_attachment(3)
    df = pd.read_excel(path, header=None)
    df = df.iloc[1:].reset_index(drop=True)
    # 动态推导场景列数（总列数 - 1个 hour_str 列）
    num_scenes = df.shape[1] - 1
    df.columns = ['hour_str'] + [f'scene_{i}' for i in range(1, num_scenes + 1)]
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)
    df['hour'] = range(24)
    return df


def load_attachment4():
    """加载附件4：园区4种场景的光伏标幺功率表 → DataFrame[hour, scene_1..scene_n]"""
    path = _find_attachment(4)
    df = pd.read_excel(path, header=None)
    df = df.iloc[1:].reset_index(drop=True)
    num_scenes = df.shape[1] - 1
    df.columns = ['hour_str'] + [f'scene_{i}' for i in range(1, num_scenes + 1)]
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)
    df['hour'] = range(24)
    return df


def get_all_scenarios():
    """返回24个场景列表（6风速 × 4光伏 = 24种组合，数量从数据动态推导）。

    每个场景字典结构:
        {
            'id': int,          # 1-24
            'wind_scene': int,  # 1-6
            'pv_scene': int,    # 1-4
            'wind_pu': np.ndarray[24],
            'pv_pu': np.ndarray[24],
        }
    """
    wind_df = load_attachment3()
    pv_df = load_attachment4()

    # 从数据动态推导场景数量（scene_ 开头的列数）
    num_wind = sum(1 for c in wind_df.columns if c.startswith('scene_'))
    num_pv = sum(1 for c in pv_df.columns if c.startswith('scene_'))

    scenarios = []
    for wi in range(1, num_wind + 1):
        for pvi in range(1, num_pv + 1):
            wind_pu = wind_df[f'scene_{wi}'].values
            pv_pu = pv_df[f'scene_{pvi}'].values
            scenarios.append({
                'id': len(scenarios) + 1,
                'wind_scene': wi,
                'pv_scene': pvi,
                'wind_pu': wind_pu,
                'pv_pu': pv_pu,
            })
    return scenarios


def get_typical_day():
    """返回典型日风电/光伏标幺曲线 → (wind_pu[24], pv_pu[24])"""
    df2 = load_attachment2()
    return df2['p_wind_pu'].values, df2['p_pv_pu'].values


def get_base_load():
    """返回基荷标幺曲线 → base_pu[24]"""
    df1 = load_attachment1()
    return df1['p_base_pu'].values


def get_scenario_by_id(scenario_id):
    """根据场景ID (1-24) 返回单个场景字典。
    
    Raises:
        ValueError: 如果 scenario_id 不在有效范围内。
    """
    scenarios = get_all_scenarios()
    for s in scenarios:
        if s['id'] == scenario_id:
            return s
    raise ValueError(f"场景ID {scenario_id} 不在 1-{len(scenarios)} 范围内")


def load_price_data():
    """加载附件7：分时电价表 → 24小时电价数组 (元/kWh)。

    时段映射:
        低谷 (23:00-次日07:00):  小时 0-6, 23
        平段 (07:00-10:00, 15:00-18:00, 21:00-23:00): 小时 7-9, 15-17, 21-22
        高峰 (10:00-15:00, 18:00-21:00):  小时 10-14, 18-20
    """
    path = _find_attachment(7)
    df = pd.read_excel(path, header=None)
    df = df.iloc[1:].reset_index(drop=True)  # 跳过表头行

    # 解析时段和对应电价
    peak_price = flat_price = valley_price = None
    for _, row in df.iterrows():
        text = str(row.iloc[0])
        price = float(row.iloc[1])
        if '高峰' in text:
            peak_price = price
        elif '低谷' in text:
            valley_price = price
        elif '平' in text:
            flat_price = price

    if None in (peak_price, flat_price, valley_price):
        raise ValueError(
            f"未能从附件7解析全部电价: 高峰={peak_price}, 平段={flat_price}, 低谷={valley_price}"
        )

    price = np.zeros(24)
    for h in range(24):
        if 10 <= h <= 14 or 18 <= h <= 20:
            price[h] = peak_price
        elif 7 <= h <= 9 or 15 <= h <= 17 or 21 <= h <= 22:
            price[h] = flat_price
        else:
            price[h] = valley_price
    return price
