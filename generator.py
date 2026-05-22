import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 데이터 로드 함수 정의

def load_csv_data_public(file_name, encoding='cp949'):
    file_path = os.path.join('./dataset', file_name)
    try:    
        load_data_csv = pd.read_csv(file_path, encoding=encoding)
        print(f"Successfully loaded CSV file: {file_name}")
        return load_data_csv
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

# 데이터 로딩

print("데이터 로딩 중...")
solarPower_170101_230228 = load_csv_data_public('태양광 발전량_170101_230228.csv')
solarPower_230301_230531 = load_csv_data_public('태양광 발전량_230301_230531.csv')

windPower_ss_130101_260331 = load_csv_data_public('한국남부발전(주)_성산풍력발전실적_20260331.csv')
windPower_hk_130101_260331 = load_csv_data_public('한국남부발전(주)_한경풍력발전실적_20260331.csv')

solarWindPower_230601_230831 = load_csv_data_public('태양광 및 풍력 발전량_230601_230831.csv')
solarWindPower_230901_231130 = load_csv_data_public('태양광 및 풍력 발전량_230901_231130.csv')
solarWindPower_231201_231231 = load_csv_data_public('태양광 및 풍력 발전량_231201_231231.csv')
solarWindPower_240101_241231 = load_csv_data_public('태양광 및 풍력 발전량_240101_241231.csv')
solarWindPower_250101_251231 = load_csv_data_public('태양광 및 풍력 발전량_250101_251231.csv')

location_windPower = load_csv_data_public('한국에너지공단_풍력기 위치정보_20221231.csv')

weather_2020 = load_csv_data_public('20200101~20201231_기상.csv')
weather_2021 = load_csv_data_public('20210101~20211231_기상.csv')
weather_2022 = load_csv_data_public('20220101~20221231_기상.csv')
weather_2023 = load_csv_data_public('20230101~20231231_기상.csv')
weather_2024 = load_csv_data_public('20240101~20241231_기상.csv')
weather_2025 = load_csv_data_public('20250101~20251231_기상.csv')

fineDust_210101_260511 = load_csv_data_public('OBS_부유분진_DD_20260513125906.csv')

# 기상 데이터 통합

weather_list = [weather_2020, weather_2021, weather_2022, weather_2023, weather_2024, weather_2025]
weather_df = pd.concat([df for df in weather_list if df is not None], ignore_index=True)

weather_df.columns = [
    '지점', '지점명', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', 
    '풍향(16방위)', '습도(%)', '현지기압(hPa)', '일사(MJ/m2)', '전운량(10분위)'
]
weather_df['일시'] = pd.to_datetime(weather_df['일시'])

# 발전량 데이터 통합 및 시간 변환

# 23년 발전량 데이터 컬럼 통일 및 병합
solarWindPower_230601_230831.rename(columns={'태양광발전량(MWh)': '태양광', '풍력발전량(MWh)': '풍력'}, inplace=True)
solarWindPower_230901_231130.rename(columns={'지역명': '지역', '태양광발전량(Mwh)': '태양광', '풍력발전량(Mwh)': '풍력'}, inplace=True)

df_23 = pd.concat([solarWindPower_230601_230831, solarWindPower_230901_231130], ignore_index=True)

# 23년 12월 ~ 25년 데이터 컬럼 통일 및 병합
solarWindPower_231201_231231.rename(columns={'발전량(MWh)': '전력거래량(MWh)'}, inplace=True)
solarWindPower_250101_251231.rename(columns={'거래일': '거래일자'}, inplace=True)

df_24_25 = pd.concat([
    solarWindPower_231201_231231, 
    solarWindPower_240101_241231, 
    solarWindPower_250101_251231
], ignore_index=True)

# 피벗 테이블 활용하여 구조 변경
df_24_25_pivot = df_24_25.pivot_table(
    index=['거래일자', '거래시간', '지역'], 
    columns='연료원', 
    values='전력거래량(MWh)', 
    aggfunc='sum'
).reset_index()

# 최종 발전량 데이터 통합
power_df = pd.concat([df_23, df_24_25_pivot], ignore_index=True)
power_df['일시'] = pd.to_datetime(power_df['거래일자']) + pd.to_timedelta(power_df['거래시간']-1, unit='h')
power_df = power_df.drop(['거래일자', '거래시간'], axis=1)

# 풍력 및 미세먼지 추가 처리

def process_wind_plant(df, plant_name):
    if df is None: return None
    melted = pd.melt(
        df, 
        id_vars=['년월일'], 
        value_vars=[str(i) for i in range(1, 25)], 
        var_name='시간', 
        value_name=f'{plant_name}_발전량'
    )
    melted['일시'] = pd.to_datetime(melted['년월일']) + pd.to_timedelta(melted['시간'].astype(int)-1, unit='h')
    return melted[['일시', f'{plant_name}_발전량']]

ss_power = process_wind_plant(windPower_ss_130101_260331, '성산')
hk_power = process_wind_plant(windPower_hk_130101_260331, '한경')

if fineDust_210101_260511 is not None:
    fineDust_210101_260511.columns = ['지점', '지점명', '일시', '미세먼지농도']
    fineDust_210101_260511['일시'] = pd.to_datetime(fineDust_210101_260511['일시'])

# 시계열 전처리 및 지역별 통합 데이터셋 구축

def preprocess_time_series(df):
    processed_df = df.copy()
    
    # 숫자형 변환 처리 (FutureWarning 방지)
    cols_to_fix = processed_df.columns.difference(['일시', '지역', '지점명'])
    processed_df[cols_to_fix] = processed_df[cols_to_fix].apply(pd.to_numeric, errors='coerce')
    
    if '미세먼지농도' in processed_df.columns:
        processed_df['미세먼지농도'] = processed_df['미세먼지농도'].ffill()
    
    processed_df = processed_df.interpolate(method='linear')
    processed_df = processed_df.bfill().ffill()
    
    processed_df['시간'] = processed_df['일시'].dt.hour
    processed_df['월'] = processed_df['일시'].dt.month
    
    return processed_df

def build_integrated_dataset(target_power_df, fuel_type, mapping_dict):
    integrated_list = []
    for region, stations in mapping_dict.items():
        print(f"[{fuel_type}] {region} 데이터 병합 중...")
        region_power = target_power_df[target_power_df['지역'] == region].copy()
        if region_power.empty: continue
            
        target_weather = weather_df[weather_df['지점명'].isin(stations['weather'])].copy()
        avg_weather = target_weather.groupby('일시').mean(numeric_only=True).reset_index() if not target_weather.empty else pd.DataFrame(columns=['일시'])
        
        if fineDust_210101_260511 is not None:
            target_dust = fineDust_210101_260511[fineDust_210101_260511['지점명'].isin(stations['dust'])].copy()
            avg_dust = target_dust.groupby('일시').mean(numeric_only=True).reset_index() if not target_dust.empty else pd.DataFrame(columns=['일시', '미세먼지농도'])
        else:
            avg_dust = pd.DataFrame(columns=['일시', '미세먼지농도'])
            
        if avg_weather.empty: continue
            
        merged = pd.merge(region_power, avg_weather, on='일시', how='inner')
        if not avg_dust.empty and '미세먼지농도' in avg_dust.columns:
            merged = pd.merge(merged, avg_dust[['일시', '미세먼지농도']], on='일시', how='left')
        else:
            merged['미세먼지농도'] = np.nan
        integrated_list.append(merged)
        
    return pd.concat(integrated_list, ignore_index=True) if integrated_list else pd.DataFrame()

# 지역별 매핑 딕셔너리
solar_mapping = {
    '서울시': {'weather': ['서울'], 'dust': ['서울']}, '부산시': {'weather': ['부산'], 'dust': ['부산']},
    '대구시': {'weather': ['대구'], 'dust': ['대구']}, '인천시': {'weather': ['인천'], 'dust': ['인천']},
    '광주시': {'weather': ['광주'], 'dust': ['광주']}, '대전시': {'weather': ['대전'], 'dust': ['대전']},
    '울산시': {'weather': ['울산'], 'dust': ['울산']}, '세종시': {'weather': ['세종', '대전'], 'dust': ['세종', '대전']},
    '경기도': {'weather': ['수원', '파주', '이천'], 'dust': ['수원', '파주', '이천']},
    '강원도': {'weather': ['춘천', '원주', '강릉', '속초'], 'dust': ['춘천', '원주', '강릉', '속초']},
    '충청북도': {'weather': ['청주', '충주', '추풍령'], 'dust': ['청주', '충주', '추풍령']},
    '충청남도': {'weather': ['홍성', '천안', '보령'], 'dust': ['홍성', '천안', '보령']},
    '전라북도': {'weather': ['전주', '군산', '부안'], 'dust': ['전주', '군산', '부안']},
    '전라남도': {'weather': ['목포', '여수', '순천'], 'dust': ['목포', '여수', '순천']},
    '경상북도': {'weather': ['안동', '포항', '구미'], 'dust': ['안동', '포항', '구미']},
    '경상남도': {'weather': ['창원', '진주', '통영'], 'dust': ['창원', '진주', '통영']},
    '제주도': {'weather': ['제주', '서귀포', '성산', '고산'], 'dust': ['제주', '서귀포', '성산', '고산']},
    '제주': {'weather': ['제주', '서귀포', '성산', '고산'], 'dust': ['제주', '서귀포', '성산', '고산']}
}
wind_mapping = {
    '강원도': {'weather': ['대관령', '태백'], 'dust': ['대관령', '태백']}, 
    '제주도': {'weather': ['고산', '성산'], 'dust': ['고산', '성산']},
    '제주': {'weather': ['고산', '성산'], 'dust': ['고산', '성산']},
    '경상북도': {'weather': ['포항'], 'dust': ['포항']},
    '전라북도': {'weather': ['군산'], 'dust': ['군산']},
    '육지': {'weather': ['대관령', '포항', '군산'], 'dust': ['대관령', '포항', '군산']}
}

# 태양광/풍력 데이터 분리 및 통합
power_df_solar = power_df[['일시', '지역', '태양광']].copy().rename(columns={'태양광': '전력거래량(MWh)'})
power_df_solar = power_df_solar[power_df_solar['일시'] >= '2020-01-01'].dropna(subset=['전력거래량(MWh)'])

power_df_wind = power_df[['일시', '지역', '풍력']].copy().rename(columns={'풍력': '전력거래량(MWh)'})
power_df_wind = power_df_wind.dropna(subset=['전력거래량(MWh)'])

solar_integrated = build_integrated_dataset(power_df_solar, '태양광', solar_mapping)
if not solar_integrated.empty: solar_integrated = preprocess_time_series(solar_integrated)

wind_integrated = build_integrated_dataset(power_df_wind, '풍력', wind_mapping)
if not wind_integrated.empty: wind_integrated = preprocess_time_series(wind_integrated)

solar_integrated = solar_integrated.sort_values(by=['지역', '일시']).reset_index(drop=True)
wind_integrated = wind_integrated.sort_values(by=['지역', '일시']).reset_index(drop=True)

# CSV 저장
solar_integrated.to_csv('solar_integrated_dataset.csv', index=False, encoding='utf-8-sig')
wind_integrated.to_csv('wind_integrated_dataset.csv', index=False, encoding='utf-8-sig')

# 모델링 데이터 준비

print("\n모델링 데이터 준비 중...")
solar_df = pd.read_csv('solar_integrated_dataset.csv', encoding='utf-8-sig')
solar_df['일시'] = pd.to_datetime(solar_df['일시'])

features = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간', '월']
target = '전력거래량(MWh)'

scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
scaled_X = scaler_X.fit_transform(solar_df[features])
scaled_y = scaler_y.fit_transform(solar_df[[target]])

def create_dataset(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_data, y_data = create_dataset(scaled_X, scaled_y, 24)
split_idx = int(len(X_data) * 0.8)
X_train_np, X_test_np = X_data[:split_idx], X_data[split_idx:]
y_train_np, y_test_np = y_data[:split_idx], y_data[split_idx:]

X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

print("PyTorch 텐서 변환 및 DataLoader 생성 완료")
print(f"X_train_tensor 형태: {X_train_tensor.shape}")
print(f"y_train_tensor 형태: {y_train_tensor.shape}")