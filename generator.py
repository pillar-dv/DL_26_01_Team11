import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import joblib 
import torch
import torch.nn as nn
import torch.optim as optim
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 데이터 로드 함수 정의

DATASET_PATH = r'./dataset'

def load_csv_data_public(file_name, encoding='cp949'):
    file_path = os.path.join(DATASET_PATH, file_name)
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
power_df['일시'] = (
    pd.to_datetime(power_df['거래일자'])
    + pd.to_timedelta(power_df['거래시간'].astype(int) - 1, unit='h')
)
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
    melted['일시'] = (
        pd.to_datetime(melted['년월일'])
        + pd.to_timedelta(melted['시간'].astype(int) - 1, unit='h')
)
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
    
    if '일사(MJ/m2)' in processed_df.columns:
        processed_df['일사(MJ/m2)'] = processed_df['일사(MJ/m2)'].fillna(0)
        
    numeric_cols = processed_df.select_dtypes(include='number').columns
    processed_df[numeric_cols] = processed_df[numeric_cols].interpolate(method='linear')
    processed_df[numeric_cols] = processed_df[numeric_cols].bfill().ffill()
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
    '제주도': {'weather': ['제주', '서귀포', '성산', '고산'], 'dust': ['제주', '서귀포', '성산', '고산']}}
 
wind_mapping = {
    '강원도': {'weather': ['대관령', '태백'], 'dust': ['대관령', '태백']}, 
    '제주도': {'weather': ['고산', '성산'], 'dust': ['고산', '성산']},
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

# CSV 저장
solar_integrated.to_csv(os.path.join(DATASET_PATH, 'solar_integrated_dataset.csv'), index=False, encoding='utf-8-sig')
wind_integrated.to_csv(os.path.join(DATASET_PATH, 'wind_integrated_dataset.csv'), index=False, encoding='utf-8-sig')

# 모델링 데이터 준비

print("\n모델링 데이터 준비 중...")
solar_df = pd.read_csv(os.path.join(DATASET_PATH, 'solar_integrated_dataset.csv'), encoding='utf-8-sig')
solar_df['일시'] = pd.to_datetime(solar_df['일시'])

def create_dataset(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

features = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간', '월', '일사(MJ/m2)']
target   = '전력거래량(MWh)'

#태양광 스케일링
# ── 1단계: 지역별 스케일링 ───────────────────────
scalers_X = {}
scalers_y = {}
scaled_X_list = []
scaled_y_list = []
region_labels = []

for region in solar_df['지역'].unique():
    mask = solar_df['지역'] == region
    region_df = solar_df[mask].copy()
    region_df = region_df.sort_values('일시').reset_index(drop=True)
    if len(region_df) < 25:
        continue

    n = len(region_df)
    train_end_r = int(n * 0.7)
    train_df = region_df.iloc[:train_end_r]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(train_df[features])
    scaler_y.fit(train_df[[target]])

    scaled_X = scaler_X.transform(region_df[features])
    scaled_y = scaler_y.transform(region_df[[target]])

    scalers_X[region] = scaler_X
    scalers_y[region] = scaler_y
    scaled_X_list.append(scaled_X)
    scaled_y_list.append(scaled_y)
    region_labels.extend([region] * len(region_df))

all_X = np.concatenate(scaled_X_list, axis=0)
all_y = np.concatenate(scaled_y_list, axis=0)
region_labels = np.array(region_labels)


# ── LSTM 모델 정의 ────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── 하이퍼파라미터 ────────────────────────────────
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
OUTPUT_SIZE = 1
DROPOUT     = 0.3
EPOCHS      = 200
LR          = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")
criterion = nn.MSELoss()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 태양광 지역별 모델 학습 ───────────────────────
print("\n태양광 지역별 모델 학습 시작...")
solar_models              = {}
solar_train_losses        = {}
solar_val_losses          = {}
solar_preds_actual_dict   = {}
solar_actuals_actual_dict = {}
solar_mae_dict, solar_rmse_dict, solar_r2_dict, solar_mape_dict = {}, {}, {}, {}

for region in solar_df['지역'].unique():
    region_mask = (region_labels == region)
    region_X    = all_X[region_mask]
    region_y    = all_y[region_mask]
    if len(region_X) < 25: continue

    X_seq, y_seq = create_dataset(region_X, region_y, 24)
    if len(X_seq) < 10: continue

    n     = len(X_seq)
    t_end = int(n * 0.7)
    v_end = int(n * 0.8)

    train_loader_r = DataLoader(TensorDataset(
        torch.tensor(X_seq[:t_end],      dtype=torch.float32),
        torch.tensor(y_seq[:t_end],      dtype=torch.float32)
    ), batch_size=64, shuffle=True)
    val_loader_r   = DataLoader(TensorDataset(
        torch.tensor(X_seq[t_end:v_end], dtype=torch.float32),
        torch.tensor(y_seq[t_end:v_end], dtype=torch.float32)
    ), batch_size=64, shuffle=False)
    test_loader_r  = DataLoader(TensorDataset(
        torch.tensor(X_seq[v_end:],      dtype=torch.float32),
        torch.tensor(y_seq[v_end:],      dtype=torch.float32)
    ), batch_size=64, shuffle=False)

    model_r     = LSTMModel(len(features), HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    optimizer_r = optim.Adam(model_r.parameters(), lr=LR)
    scheduler_r = optim.lr_scheduler.ReduceLROnPlateau(optimizer_r, mode='min', patience=5, factor=0.5)

    train_losses_r, val_losses_r            = [], []
    best_val_r, counter_r, best_epoch_r     = float('inf'), 0, 0

    for epoch in range(EPOCHS):
        model_r.train()
        train_loss = 0
        for X_batch, y_batch in train_loader_r:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer_r.zero_grad()
            loss = criterion(model_r(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model_r.parameters(), max_norm=1.0)
            optimizer_r.step()
            train_loss += loss.item()

        model_r.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader_r:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model_r(X_batch), y_batch).item()

        train_loss /= len(train_loader_r)
        val_loss   /= len(val_loader_r)
        train_losses_r.append(train_loss)
        val_losses_r.append(val_loss)
        scheduler_r.step(val_loss)

        if val_loss < best_val_r:
            best_val_r, best_epoch_r, counter_r = val_loss, epoch + 1, 0
            torch.save(model_r.state_dict(), os.path.join(DATASET_PATH, f'best_model_solar_{region}.pth'))
        else:
            counter_r += 1
            if counter_r >= 20:
                print(f"  {region} Early Stopping at epoch {epoch+1} (최적: {best_epoch_r})")
                break

    model_r.load_state_dict(torch.load(
        os.path.join(DATASET_PATH, f'best_model_solar_{region}.pth'), map_location=device))
    solar_models[region]       = model_r
    solar_train_losses[region] = train_losses_r
    solar_val_losses[region]   = val_losses_r

    model_r.eval()
    preds_s, actuals_s = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader_r:
            preds_s.append(model_r(X_batch.to(device)).cpu().numpy())
            actuals_s.append(y_batch.numpy())

    preds_s   = scalers_y[region].inverse_transform(np.concatenate(preds_s))
    actuals_s = scalers_y[region].inverse_transform(np.concatenate(actuals_s))

    solar_preds_actual_dict[region]   = preds_s
    solar_actuals_actual_dict[region] = actuals_s

    mask_50 = actuals_s.flatten() > 50
    solar_mae_dict[region]  = mean_absolute_error(actuals_s, preds_s)
    solar_rmse_dict[region] = np.sqrt(mean_squared_error(actuals_s, preds_s))
    solar_r2_dict[region]   = r2_score(actuals_s, preds_s)
    solar_mape_dict[region] = np.mean(np.abs(
        (actuals_s[mask_50] - preds_s[mask_50]) / actuals_s[mask_50]
    )) * 100 if mask_50.sum() > 0 else float('nan')

    print(f"  {region} 완료 | MAE: {solar_mae_dict[region]:.2f} | R²: {solar_r2_dict[region]:.4f} | Best Val: {best_val_r:.4f}")

# ── 스케일러 저장 ─────────────────────────────────
joblib.dump(scalers_X, os.path.join(DATASET_PATH, 'scalers_X_solar.pkl'))
joblib.dump(scalers_y, os.path.join(DATASET_PATH, 'scalers_y_solar.pkl'))

# ── 전체 통합 평가 ────────────────────────────────
all_preds_s   = np.concatenate(list(solar_preds_actual_dict.values()))
all_actuals_s = np.concatenate(list(solar_actuals_actual_dict.values()))
mask_50_all   = all_actuals_s.flatten() > 50

print("\n===== 태양광 전체 평가 결과 =====")
print(f"MAE  : {mean_absolute_error(all_actuals_s, all_preds_s):.4f} MWh")
print(f"RMSE : {np.sqrt(mean_squared_error(all_actuals_s, all_preds_s)):.4f} MWh")
print(f"R²   : {r2_score(all_actuals_s, all_preds_s):.4f}")
print(f"MAPE : {np.mean(np.abs((all_actuals_s[mask_50_all] - all_preds_s[mask_50_all]) / all_actuals_s[mask_50_all])) * 100:.2f}% (50MWh 초과)")

print("\n===== 태양광 지역별 평가 결과 =====")
for region in solar_mae_dict:
    print(f"  {region} | MAE: {solar_mae_dict[region]:.2f} | RMSE: {solar_rmse_dict[region]:.2f} | R²: {solar_r2_dict[region]:.4f} | MAPE: {solar_mape_dict[region]:.2f}%")

# ── 태양광 지역별 시각화 ──────────────────────────
for region in solar_models.keys():
    region_df_full = solar_df[solar_df['지역'] == region].sort_values('일시').reset_index(drop=True)
    n_total        = len(region_df_full)
    test_dates     = region_df_full['일시'].iloc[int(n_total * 0.8):].reset_index(drop=True)
    n_plot         = min(300, len(solar_actuals_actual_dict[region]))
    plot_dates     = test_dates[:n_plot]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(solar_train_losses[region], label="Train Loss")
    axes[0].plot(solar_val_losses[region],   label="Val Loss")
    axes[0].set_title(f"{region} 태양광 학습 손실 곡선")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()

    axes[1].plot(plot_dates, solar_actuals_actual_dict[region][:n_plot], label="실제값", alpha=0.7)
    axes[1].plot(plot_dates, solar_preds_actual_dict[region][:n_plot],   label="예측값", alpha=0.7)
    axes[1].set_title(f"{region} 태양광 실제 vs 예측")
    axes[1].set_xlabel("날짜 (시간 단위)")
    axes[1].set_ylabel("발전량 (MWh)")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_PATH, f"lstm_result_solar_{region}.png"), dpi=150)
    plt.close()
    print(f"{region} 태양광 시각화 저장 완료")


# ── 풍력 XGBoost 데이터 준비 ─────────────────────
import xgboost as xgb

wind_df = pd.read_csv(os.path.join(DATASET_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
wind_df['일시'] = pd.to_datetime(wind_df['일시'])
wind_df = wind_df[wind_df['지역'] != '제주도'].reset_index(drop=True)

print(f"풍력 지역: {wind_df['지역'].unique().tolist()}")
print(f"풍력 전체 데이터 수: {len(wind_df)}행")

features_xgb_wind = [
    '기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)',
    '습도(%)', '현지기압(hPa)', '전운량(10분위)',
    '시간', '월', '요일', '주말여부',
    'lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_168',
    'rolling_mean_6', 'rolling_mean_24', 'rolling_std_24'
]
target_wind = '전력거래량(MWh)'

def make_wind_features_xgb(df):
    df = df.copy().sort_values('일시').reset_index(drop=True)
    df['풍속_세제곱']     = df['풍속(m/s)'] ** 3
    df['시간']            = df['일시'].dt.hour
    df['월']              = df['일시'].dt.month
    df['요일']            = df['일시'].dt.dayofweek
    df['주말여부']         = (df['요일'] >= 5).astype(int)
    df['lag_1']           = df[target_wind].shift(1)
    df['lag_2']           = df[target_wind].shift(2)
    df['lag_3']           = df[target_wind].shift(3)
    df['lag_24']          = df[target_wind].shift(24)
    df['lag_168']         = df[target_wind].shift(168)
    df['rolling_mean_6']  = df[target_wind].shift(1).rolling(6).mean()
    df['rolling_mean_24'] = df[target_wind].shift(1).rolling(24).mean()
    df['rolling_std_24']  = df[target_wind].shift(1).rolling(24).std()
    return df.dropna().reset_index(drop=True)

# ── 풍력 지역별 XGBoost 학습 ─────────────────────
print("\n풍력 XGBoost 지역별 학습 시작...")
xgb_models                                     = {}
xgb_preds_actual_dict, xgb_actuals_actual_dict = {}, {}
xgb_mae_dict, xgb_rmse_dict, xgb_r2_dict, xgb_smape_dict = {}, {}, {}, {}

for region in wind_df['지역'].unique():
    print(f"\n  {region} 학습 중...")
    region_df = make_wind_features_xgb(wind_df[wind_df['지역'] == region].copy())

    if len(region_df) < 200:
        print(f"  {region} 데이터 부족 → 스킵")
        continue

    n     = len(region_df)
    t_end = int(n * 0.7)
    v_end = int(n * 0.8)

    X_train = region_df[features_xgb_wind].iloc[:t_end]
    y_train = region_df[target_wind].iloc[:t_end]
    X_val   = region_df[features_xgb_wind].iloc[t_end:v_end]
    y_val   = region_df[target_wind].iloc[t_end:v_end]
    X_test  = region_df[features_xgb_wind].iloc[v_end:]
    y_test  = region_df[target_wind].iloc[v_end:]

    model_xgb = xgb.XGBRegressor(
        n_estimators          = 1000,
        learning_rate         = 0.05,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        random_state          = 42,
        n_jobs                = -1,
        eval_metric           = 'rmse',
        early_stopping_rounds = 50
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    preds   = np.maximum(model_xgb.predict(X_test), 0)
    actuals = y_test.values

    xgb_models[region]              = model_xgb
    xgb_preds_actual_dict[region]   = preds
    xgb_actuals_actual_dict[region] = actuals
    xgb_mae_dict[region]    = mean_absolute_error(actuals, preds)
    xgb_rmse_dict[region]   = np.sqrt(mean_squared_error(actuals, preds))
    xgb_r2_dict[region]     = r2_score(actuals, preds)
    xgb_smape_dict[region]  = np.mean(2 * np.abs(actuals - preds) /
                              (np.abs(actuals) + np.abs(preds) + 1e-8)) * 100

    joblib.dump(model_xgb, os.path.join(DATASET_PATH, f'xgb_wind_{region}.pkl'))
    print(f"  {region} 완료 | MAE: {xgb_mae_dict[region]:.2f} | R²: {xgb_r2_dict[region]:.4f} | sMAPE: {xgb_smape_dict[region]:.2f}%")

# ── 전체 평가 ─────────────────────────────────────
all_preds_w   = np.concatenate(list(xgb_preds_actual_dict.values()))
all_actuals_w = np.concatenate(list(xgb_actuals_actual_dict.values()))

print("\n===== 풍력 XGBoost 전체 평가 결과 =====")
print(f"MAE  : {mean_absolute_error(all_actuals_w, all_preds_w):.4f} MWh")
print(f"RMSE : {np.sqrt(mean_squared_error(all_actuals_w, all_preds_w)):.4f} MWh")
print(f"R²   : {r2_score(all_actuals_w, all_preds_w):.4f}")
print(f"sMAPE: {np.mean(2 * np.abs(all_actuals_w - all_preds_w) / (np.abs(all_actuals_w) + np.abs(all_preds_w) + 1e-8)) * 100:.2f}%")

print("\n===== 풍력 XGBoost 지역별 평가 결과 =====")
for region in xgb_mae_dict:
    print(f"  {region} | MAE: {xgb_mae_dict[region]:.2f} | RMSE: {xgb_rmse_dict[region]:.2f} | R²: {xgb_r2_dict[region]:.4f} | sMAPE: {xgb_smape_dict[region]:.2f}%")

# ── 풍력 지역별 시각화 ────────────────────────────
for region in xgb_models.keys():
    region_df_full = make_wind_features_xgb(wind_df[wind_df['지역'] == region].copy())
    n_total        = len(region_df_full)
    test_dates     = region_df_full['일시'].iloc[int(n_total * 0.8):].reset_index(drop=True)
    n_plot         = min(300, len(xgb_actuals_actual_dict[region]))
    plot_dates     = test_dates[:n_plot]

    importance = xgb_models[region].feature_importances_
    sorted_idx = np.argsort(importance)[-10:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].barh(np.array(features_xgb_wind)[sorted_idx], importance[sorted_idx], color='steelblue')
    axes[0].set_title(f"{region} 풍력 XGBoost 피처 중요도 (Top 10)")
    axes[0].set_xlabel("Importance")

    axes[1].plot(plot_dates, xgb_actuals_actual_dict[region][:n_plot], label="실제값", alpha=0.7)
    axes[1].plot(plot_dates, xgb_preds_actual_dict[region][:n_plot],   label="예측값", alpha=0.7)
    axes[1].set_title(f"{region} 풍력 XGBoost 실제 vs 예측")
    axes[1].set_xlabel("날짜 (시간 단위)")
    axes[1].set_ylabel("발전량 (MWh)")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_PATH, f"xgb_result_wind_{region}.png"), dpi=150)
    plt.close()
    print(f"{region} 풍력 XGBoost 시각화 저장 완료")

print("\n✅ 전체 학습 완료! (태양광 LSTM 지역별 + 풍력 XGBoost 지역별)")
