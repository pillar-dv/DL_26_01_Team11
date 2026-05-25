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

# ── 2단계: 지역별 시퀀스 생성 후 합치기 ──────────
X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []
X_test_list,  y_test_list  = [], []
train_label_list, val_label_list, test_label_list = [], [], []

for region in solar_df['지역'].unique():
    region_mask = (region_labels == region)
    region_X = all_X[region_mask]
    region_y = all_y[region_mask]

    if len(region_X) < 25:
        continue

    X_seq, y_seq = create_dataset(region_X, region_y, 24)
    if len(X_seq) < 10:
        continue

    n = len(X_seq)
    t_end = int(n * 0.7)
    v_end = int(n * 0.8)

    X_train_list.append(X_seq[:t_end])
    y_train_list.append(y_seq[:t_end])
    X_val_list.append(X_seq[t_end:v_end])
    y_val_list.append(y_seq[t_end:v_end])
    X_test_list.append(X_seq[v_end:])
    y_test_list.append(y_seq[v_end:])

    train_label_list.extend([region] * t_end)
    val_label_list.extend([region]   * (v_end - t_end))
    test_label_list.extend([region]  * (n - v_end))

# ── 3단계: 합치기 & 텐서 변환 ────────────────────
X_train_np = np.concatenate(X_train_list, axis=0)
y_train_np = np.concatenate(y_train_list, axis=0)
X_val_np   = np.concatenate(X_val_list,   axis=0)
y_val_np   = np.concatenate(y_val_list,   axis=0)
X_test_np  = np.concatenate(X_test_list,  axis=0)
y_test_np  = np.concatenate(y_test_list,  axis=0)

test_region_labels = np.array(test_label_list)

X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_np,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val_np,   dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_np,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test_np,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)   # False → True
val_loader   = DataLoader(TensorDataset(X_val_tensor,   y_val_tensor),   batch_size=64, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_tensor,  y_test_tensor),  batch_size=64, shuffle=False)

print(f"지역 수: {len(scalers_X)}개")
print(f"train: {len(X_train_np)} | val: {len(X_val_np)} | test: {len(X_test_np)}")
print("지역별 스케일링 및 DataLoader 생성 완료")

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
        out = self.fc(out[:, -1, :])
        return out

# ── 하이퍼파라미터 ────────────────────────────────
INPUT_SIZE  = len(features)  # 일사량 추가로 7개
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
OUTPUT_SIZE = 1
DROPOUT     = 0.3            # 0.2 → 0.3 과적합 방지
EPOCHS      = 200            # Early Stopping이 알아서 멈추니까 넉넉하게
LR          = 0.001

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
import torch
import torch_directml
'''
'''
try:
    import torch_directml
    device = torch_directml.device()
    print(f"AMD GPU 가속 활성화: {torch_directml.device_name(0)}")
except:
    device = torch.device("cpu")
    print("torch-directml이 설치되지 않아 CPU로 실행됩니다.")
'''

device = torch.device("cpu")
print("CPU 모드로 안전하게 학습/추론을 진행합니다.")


model     = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# ── 학습 + Early Stopping ─────────────────────────
print("\n학습 시작...")
train_losses, val_losses = [], []

best_val_loss = float('inf')
patience      = 20
counter       = 0
best_epoch    = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:  # test_loader → val_loader
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)    # test_loader → val_loader

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f} | Patience: {counter}/{patience}")

    # Early Stopping 체크
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch + 1
        counter       = 0
        torch.save(model.state_dict(), os.path.join(DATASET_PATH, 'best_model.pth'))
    else:
        counter += 1
        if counter >= patience:
            print(f"\nEarly Stopping! epoch {epoch+1}에서 중단 (최적 epoch: {best_epoch})")
            break

# 최적 모델 복원
model.load_state_dict(torch.load(os.path.join(DATASET_PATH, 'best_model.pth'), map_location=device))

# 스케일러 파일로 저장 (추론 시 사용)
joblib.dump(scalers_X, os.path.join(DATASET_PATH, 'scalers_X_solar.pkl'))
joblib.dump(scalers_y, os.path.join(DATASET_PATH, 'scalers_y_solar.pkl'))
print(f"최적 모델 로드 완료 (Best Val Loss: {best_val_loss:.4f})")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 예측 ──────────────────────────────────────────
model.eval()
preds_scaled, actuals_scaled = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch).cpu().numpy()
        preds_scaled.append(pred)
        actuals_scaled.append(y_batch.numpy())

preds_scaled   = np.concatenate(preds_scaled)
actuals_scaled = np.concatenate(actuals_scaled)

# ── 지역별 역변환 ─────────────────────────────────
test_region_labels = np.array(test_label_list)

preds_actual   = np.zeros_like(preds_scaled)
actuals_actual = np.zeros_like(actuals_scaled)

for region, scaler_y in scalers_y.items():
    mask = (test_region_labels == region)
    if mask.sum() == 0:
        continue
    preds_actual[mask]   = scaler_y.inverse_transform(preds_scaled[mask])
    actuals_actual[mask] = scaler_y.inverse_transform(actuals_scaled[mask])

# ── 평가 ──────────────────────────────────────────
mae  = mean_absolute_error(actuals_actual, preds_actual)
rmse = np.sqrt(mean_squared_error(actuals_actual, preds_actual))
r2   = r2_score(actuals_actual, preds_actual)
mask_solar = actuals_actual.flatten() > 50   # 실제값 50 MWh 이하 제외
mape = np.mean(np.abs((actuals_actual[mask_solar] - preds_actual[mask_solar]) / actuals_actual[mask_solar])) * 100

print("\n===== 평가 결과 =====")
print(f"MAE  : {mae:.4f} MWh")
print(f"RMSE : {rmse:.4f} MWh")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}% (실제값 50Mwh 초과 구간 기준)")

# ── 시각화 ────────────────────────────────────────
# ── 태양광 전체 지역 시각화 ───────────────────────
regions_to_plot = [r for r in scalers_y.keys() 
                   if (test_region_labels == r).sum() > 0]

fig, axes = plt.subplots(len(regions_to_plot), 2, figsize=(14, 4 * len(regions_to_plot)))

if len(regions_to_plot) == 1:
    axes = [axes]

for i, region in enumerate(regions_to_plot):
    mask_plot = test_region_labels == region

    axes[i][0].plot(train_losses, label="Train Loss")
    axes[i][0].plot(val_losses,   label="Val Loss")
    axes[i][0].set_title(f"{region} 학습 손실 곡선")
    axes[i][0].set_xlabel("Epoch")
    axes[i][0].set_ylabel("MSE Loss")
    axes[i][0].legend()

    axes[i][1].plot(actuals_actual[mask_plot][:300], label="실제값", alpha=0.7)
    axes[i][1].plot(preds_actual[mask_plot][:300],   label="예측값", alpha=0.7)
    axes[i][1].set_title(f"{region} 태양광 실제 vs 예측 (처음 300개)")
    axes[i][1].set_xlabel("Time Step")
    axes[i][1].set_ylabel("발전량 (MWh)")
    axes[i][1].legend()

plt.tight_layout()
plt.savefig(os.path.join(DATASET_PATH, "lstm_result_solar_all.png"), dpi=150)
plt.close()
print("태양광 전체 지역 시각화 저장 완료")

# ── 풍력 데이터 준비 ──────────────────────────────
wind_df = pd.read_csv(os.path.join(DATASET_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
wind_df['일시'] = pd.to_datetime(wind_df['일시'])

# 제주도 데이터 제외 및 풍속 세제곱 피처 추가
wind_df = wind_df[wind_df['지역'] != '제주도'].reset_index(drop=True)
wind_df['풍속_세제곱'] = wind_df['풍속(m/s)'] ** 3

features_wind = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간', '월']
target_wind   = '전력거래량(MWh)'

# 풍력 지역별 스케일링
scalers_X_wind = {}
scalers_y_wind = {}
scaled_X_wind_list = []
scaled_y_wind_list = []
region_labels_wind = []

for region in wind_df['지역'].unique():
    mask = wind_df['지역'] == region
    region_df = wind_df[mask].copy()
    region_df = region_df.sort_values('일시').reset_index(drop=True)
    if len(region_df) < 25:
        continue

    n = len(region_df)
    train_end_r = int(n * 0.7)

    train_df = region_df.iloc[:train_end_r]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(train_df[features_wind])
    scaler_y.fit(train_df[[target_wind]])

    scaled_X = scaler_X.transform(region_df[features_wind])
    scaled_y = scaler_y.transform(region_df[[target_wind]])

    scalers_X_wind[region] = scaler_X
    scalers_y_wind[region] = scaler_y
    scaled_X_wind_list.append(scaled_X)
    scaled_y_wind_list.append(scaled_y)
    region_labels_wind.extend([region] * len(region_df))

all_X_wind = np.concatenate(scaled_X_wind_list, axis=0)
all_y_wind = np.concatenate(scaled_y_wind_list, axis=0)
region_labels_wind = np.array(region_labels_wind)

print(f"풍력 지역 수: {len(scalers_X_wind)}개")
print(f"풍력 전체 데이터 수: {len(all_X_wind)}행")

# ── 풍력 시퀀스 생성 및 train/val/test 분리 ──────
# ── 풍력 지역별 시퀀스 생성 후 합치기 ────────────
X_train_wind_list, y_train_wind_list = [], []
X_val_wind_list,   y_val_wind_list   = [], []
X_test_wind_list,  y_test_wind_list  = [], []
train_label_wind_list, val_label_wind_list, test_label_wind_list = [], [], []

for region in wind_df['지역'].unique():
    region_mask = (region_labels_wind == region)
    region_X = all_X_wind[region_mask]
    region_y = all_y_wind[region_mask]

    if len(region_X) < 25:
        continue

    X_seq, y_seq = create_dataset(region_X, region_y, 24)
    if len(X_seq) < 10:
        continue

    n = len(X_seq)
    t_end = int(n * 0.7)
    v_end = int(n * 0.8)

    X_train_wind_list.append(X_seq[:t_end])
    y_train_wind_list.append(y_seq[:t_end])
    X_val_wind_list.append(X_seq[t_end:v_end])
    y_val_wind_list.append(y_seq[t_end:v_end])
    X_test_wind_list.append(X_seq[v_end:])
    y_test_wind_list.append(y_seq[v_end:])

    train_label_wind_list.extend([region] * t_end)
    val_label_wind_list.extend([region]   * (v_end - t_end))
    test_label_wind_list.extend([region]  * (n - v_end))

X_train_wind = torch.tensor(np.concatenate(X_train_wind_list), dtype=torch.float32)
y_train_wind = torch.tensor(np.concatenate(y_train_wind_list), dtype=torch.float32)
X_val_wind   = torch.tensor(np.concatenate(X_val_wind_list),   dtype=torch.float32)
y_val_wind   = torch.tensor(np.concatenate(y_val_wind_list),   dtype=torch.float32)
X_test_wind  = torch.tensor(np.concatenate(X_test_wind_list),  dtype=torch.float32)
y_test_wind  = torch.tensor(np.concatenate(y_test_wind_list),  dtype=torch.float32)

test_region_labels_wind = np.array(test_label_wind_list)

train_loader_wind = DataLoader(TensorDataset(X_train_wind, y_train_wind), batch_size=64, shuffle=True)   # False → True
val_loader_wind   = DataLoader(TensorDataset(X_val_wind,   y_val_wind),   batch_size=64, shuffle=False)
test_loader_wind  = DataLoader(TensorDataset(X_test_wind,  y_test_wind),  batch_size=64, shuffle=False)

print(f"풍력 지역 수: {len(scalers_X_wind)}개")
print(f"train: {len(X_train_wind)} | val: {len(X_val_wind)} | test: {len(X_test_wind)}")

# ── 풍력 모델 학습 ────────────────────────────────
print("\n풍력 모델 학습 시작...")
model_wind     = LSTMModel(len(features_wind), HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
optimizer_wind = optim.Adam(model_wind.parameters(), lr=0.0005)  # 0.001 → 0.0005
scheduler_wind = optim.lr_scheduler.ReduceLROnPlateau(optimizer_wind, mode='min', patience=5, factor=0.5)

train_losses_wind, val_losses_wind = [], []
best_val_loss_wind = float('inf')
patience_wind, counter_wind = 20, 0  # 10 → 20
best_epoch_wind = 0

for epoch in range(EPOCHS):
    model_wind.train()
    train_loss = 0
    for X_batch, y_batch in train_loader_wind:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_wind.zero_grad()
        pred = model_wind(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model_wind.parameters(), max_norm=1.0)
        optimizer_wind.step()
        train_loss += loss.item()

    model_wind.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader_wind:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model_wind(X_batch)
            val_loss += criterion(pred, y_batch).item()

    train_loss /= len(train_loader_wind)
    val_loss   /= len(val_loader_wind)
    train_losses_wind.append(train_loss)
    val_losses_wind.append(val_loss)
    scheduler_wind.step(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val_loss_wind:.4f} | Patience: {counter_wind}/{patience_wind}")

    if val_loss < best_val_loss_wind:
        best_val_loss_wind = val_loss
        best_epoch_wind    = epoch + 1
        counter_wind       = 0
        torch.save(model_wind.state_dict(), os.path.join(DATASET_PATH, 'best_model_wind.pth'))
    else:
        counter_wind += 1
        if counter_wind >= patience_wind:
            print(f"\nEarly Stopping! epoch {epoch+1}에서 중단 (최적 epoch: {best_epoch_wind})")
            break

model_wind.load_state_dict(torch.load(os.path.join(DATASET_PATH, 'best_model_wind.pth'), map_location=device))
print(f"풍력 최적 모델 로드 완료 (Best Val Loss: {best_val_loss_wind:.4f})")

# 풍력 최적 모델 스케일러 저장
joblib.dump(scalers_X_wind, os.path.join(DATASET_PATH, 'scalers_X_wind.pkl'))
joblib.dump(scalers_y_wind, os.path.join(DATASET_PATH, 'scalers_y_wind.pkl'))

# ── 풍력 평가 ─────────────────────────────────────
model_wind.eval()
preds_scaled_wind, actuals_scaled_wind = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader_wind:
        X_batch = X_batch.to(device)
        pred = model_wind(X_batch).cpu().numpy()
        preds_scaled_wind.append(pred)
        actuals_scaled_wind.append(y_batch.numpy())

preds_scaled_wind   = np.concatenate(preds_scaled_wind)
actuals_scaled_wind = np.concatenate(actuals_scaled_wind)

test_region_labels_wind = np.array(test_label_wind_list)

preds_actual_wind   = np.zeros_like(preds_scaled_wind)
actuals_actual_wind = np.zeros_like(actuals_scaled_wind)

for region, scaler_y in scalers_y_wind.items():
    mask = (test_region_labels_wind == region)
    if mask.sum() == 0:
        continue
    preds_actual_wind[mask]   = scaler_y.inverse_transform(preds_scaled_wind[mask])
    actuals_actual_wind[mask] = scaler_y.inverse_transform(actuals_scaled_wind[mask])

mae_w  = mean_absolute_error(actuals_actual_wind, preds_actual_wind)
rmse_w = np.sqrt(mean_squared_error(actuals_actual_wind, preds_actual_wind))
r2_w   = r2_score(actuals_actual_wind, preds_actual_wind)
mape_w = np.mean(2 * np.abs(actuals_actual_wind - preds_actual_wind) / 
                 (np.abs(actuals_actual_wind) + np.abs(preds_actual_wind) + 1e-8)) * 100

print("\n===== 풍력 평가 결과 =====")
print(f"MAE  : {mae_w:.4f} MWh")
print(f"RMSE : {rmse_w:.4f} MWh")
print(f"R²   : {r2_w:.4f}")
print(f"sMAPE : {mape_w:.2f}%")

# ── 풍력 시각화 ───────────────────────────────────
# ── 풍력 전체 지역 시각화 ────────────────────────
regions_to_plot_wind = [r for r in scalers_y_wind.keys()
                        if (test_region_labels_wind == r).sum() > 0]

fig, axes = plt.subplots(len(regions_to_plot_wind), 2, figsize=(14, 4 * len(regions_to_plot_wind)))

if len(regions_to_plot_wind) == 1:
    axes = [axes]

for i, region in enumerate(regions_to_plot_wind):
    mask_plot = test_region_labels_wind == region

    axes[i][0].plot(train_losses_wind, label="Train Loss")
    axes[i][0].plot(val_losses_wind,   label="Val Loss")
    axes[i][0].set_title(f"{region} 풍력 학습 손실 곡선")
    axes[i][0].set_xlabel("Epoch")
    axes[i][0].set_ylabel("MSE Loss")
    axes[i][0].legend()

    axes[i][1].plot(actuals_actual_wind[mask_plot][:300], label="실제값", alpha=0.7)
    axes[i][1].plot(preds_actual_wind[mask_plot][:300],   label="예측값", alpha=0.7)
    axes[i][1].set_title(f"{region} 풍력 실제 vs 예측 (처음 300개)")
    axes[i][1].set_xlabel("Time Step")
    axes[i][1].set_ylabel("발전량 (MWh)")
    axes[i][1].legend()

plt.tight_layout()
plt.savefig(os.path.join(DATASET_PATH, "lstm_result_wind_all.png"), dpi=150)
plt.close()
print("풍력 전체 지역 시각화 저장 완료")
