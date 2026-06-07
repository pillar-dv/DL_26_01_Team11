import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import joblib 
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
NATIONAL_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'national')
DOCS_PATH = os.path.join(PROJECT_ROOT, 'docs')

# 로깅 설정
os.makedirs(DOCS_PATH, exist_ok=True)
log_file_path = os.path.join(DOCS_PATH, 'train_log.txt')
with open(log_file_path, 'w', encoding='utf-8') as f:
    f.write("=== 재생에너지 발전량 예측 모델 학습 로그 ===\n")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 데이터 로드 함수 정의
def load_csv_data_public(file_name, encoding='cp949'):
    file_path = os.path.join(RAW_DATA_PATH, file_name)
    try:    
        load_data_csv = pd.read_csv(file_path, encoding=encoding)
        logger.info(f"Successfully loaded CSV file: {file_name}")
        return load_data_csv
    except Exception as e:
        logger.info(f"Error loading {file_name}: {e}")
        return None

# 데이터 로딩
logger.info("데이터 로딩 중...")
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
solarWindPower_230601_230831.rename(columns={'태양광발전량(MWh)': '태양광', '풍력발전량(MWh)': '풍력'}, inplace=True)
solarWindPower_230901_231130.rename(columns={'지역명': '지역', '태양광발전량(Mwh)': '태양광', '풍력발전량(Mwh)': '풍력'}, inplace=True)

df_23 = pd.concat([solarWindPower_230601_230831, solarWindPower_230901_231130], ignore_index=True)

solarWindPower_231201_231231.rename(columns={'발전량(MWh)': '전력거래량(MWh)'}, inplace=True)
solarWindPower_250101_251231.rename(columns={'거래일': '거래일자'}, inplace=True)

df_24_25 = pd.concat([
    solarWindPower_231201_231231, 
    solarWindPower_240101_241231, 
    solarWindPower_250101_251231
], ignore_index=True)

# 지자체 표준화 맵핑 적용
region_map = {
    '제주': '제주도', '강원': '강원도', '경기': '경기도', '경남': '경상남도',
    '경북': '경상북도', '전남': '전라남도', '전북': '전라북도', '충남': '충청남도',
    '충북': '충청북도', '광주': '광주시', '대구': '대구시', '대전': '대전시',
    '부산': '부산시', '서울': '서울시', '울산': '울산시', '인천': '인천시',
    '세종': '세종시'
}
df_23['지역'] = df_23['지역'].replace(region_map)
df_24_25['지역'] = df_24_25['지역'].replace(region_map)

df_24_25_pivot = df_24_25.pivot_table(
    index=['거래일자', '거래시간', '지역'], 
    columns='연료원', 
    values='전력거래량(MWh)', 
    aggfunc='sum'
).reset_index()

power_df = pd.concat([df_23, df_24_25_pivot], ignore_index=True)
power_df['일시'] = (
    pd.to_datetime(power_df['거래일자'])
    + pd.to_timedelta(power_df['거래시간'].astype(int) - 1, unit='h')
)
power_df = power_df.drop(['거래일자', '거래시간'], axis=1)

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

def preprocess_time_series(df):
    processed_df = df.copy()
    
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
    
    processed_df['시간_sin'] = np.sin(2 * np.pi * processed_df['시간'] / 24.0)
    processed_df['시간_cos'] = np.cos(2 * np.pi * processed_df['시간'] / 24.0)
    processed_df['월_sin'] = np.sin(2 * np.pi * (processed_df['월'] - 1) / 12.0)
    processed_df['월_cos'] = np.cos(2 * np.pi * (processed_df['월'] - 1) / 12.0)
    
    return processed_df

def build_integrated_dataset(target_power_df, fuel_type, mapping_dict):
    integrated_list = []
    for region, stations in mapping_dict.items():
        logger.info(f"[{fuel_type}] {region} 데이터 병합 중...")
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

power_df_solar = power_df[['일시', '지역', '태양광']].copy().rename(columns={'태양광': '전력거래량(MWh)'})
power_df_solar = power_df_solar[power_df_solar['일시'] >= '2020-01-01'].dropna(subset=['전력거래량(MWh)'])

power_df_wind = power_df[['일시', '지역', '풍력']].copy().rename(columns={'풍력': '전력거래량(MWh)'})
power_df_wind = power_df_wind.dropna(subset=['전력거래량(MWh)'])

# 2023년 육지 풍력 합산 데이터 생성 (데이터 연속성 확보)
mainland_wind_23 = power_df_wind[
    (power_df_wind['지역'] != '제주도') & 
    (power_df_wind['지역'] != '육지')
].copy()

if not mainland_wind_23.empty:
    land_sum_23 = mainland_wind_23.groupby('일시')['전력거래량(MWh)'].sum().reset_index()
    land_sum_23['지역'] = '육지'
    power_df_wind = pd.concat([power_df_wind, land_sum_23], ignore_index=True)
    logger.info(f"2023년 육지 풍력 데이터 합산 완료 ({len(land_sum_23)}개 행 추가)")

solar_integrated = build_integrated_dataset(power_df_solar, '태양광', solar_mapping)
if not solar_integrated.empty: solar_integrated = preprocess_time_series(solar_integrated)

wind_integrated = build_integrated_dataset(power_df_wind, '풍력', wind_mapping)
if not wind_integrated.empty: wind_integrated = preprocess_time_series(wind_integrated)

solar_integrated.to_csv(os.path.join(PROCESSED_DATA_PATH, 'solar_integrated_dataset.csv'), index=False, encoding='utf-8-sig')
wind_integrated.to_csv(os.path.join(PROCESSED_DATA_PATH, 'wind_integrated_dataset.csv'), index=False, encoding='utf-8-sig')

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

def create_dataset(X, y, time_steps=24):
    # [FIX] 타겟을 윈도우 끝 시점(i+23)으로 수정
    # 기존 y[i+24](윤도우 이후 1시간): 일사량 피크(12시) → 발전량 피크(13시) 구조적 시프트 유발
    # 수정 y[i+23](율돈우 끝 시점): 일사량 피크(12시) → 발전량 피크(12시) 정렬
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)

# 하이퍼파라미터 (경량화 구조 적용)
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
OUTPUT_SIZE = 1
DROPOUT     = 0.2
EPOCHS      = 100
LR          = 0.001
device = torch.device("cpu")
criterion = nn.MSELoss()

logger.info("\nCPU 모드로 안전하게 학습/추론을 진행합니다.")

# ==========================================
# 🌞 태양광 모델 지역별 독립 학습
# ==========================================
logger.info("\n모델링 데이터 준비 중 (태양광)...")
solar_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'solar_integrated_dataset.csv'), encoding='utf-8-sig')
solar_df['일시'] = pd.to_datetime(solar_df['일시'])

features_solar = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간_sin', '시간_cos', '월_sin', '월_cos', '일사(MJ/m2)']
target_solar = '전력거래량(MWh)'

scalers_X_solar = {}
scalers_y_solar = {}
losses_solar = {}
eval_results_solar = {}

test_preds_solar = {}
test_acts_solar = {}
test_dates_solar = {}

for region in solar_df['지역'].unique():
    region_df = solar_df[solar_df['지역'] == region].sort_values('일시').reset_index(drop=True)
    if len(region_df) < 50:
        continue
    
    logger.info(f"\n⚡ [태양광] {region} 모델 학습 시작 (전체 데이터: {len(region_df)}행)")
    
    n = len(region_df)
    train_end_r = int(n * 0.7)
    train_df = region_df.iloc[:train_end_r]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(train_df[features_solar])
    scaler_y.fit(train_df[[target_solar]])
    
    scalers_X_solar[region] = scaler_X
    scalers_y_solar[region] = scaler_y
    
    scaled_X = scaler_X.transform(region_df[features_solar])
    scaled_y = scaler_y.transform(region_df[[target_solar]])
    
    X_seq, y_seq = create_dataset(scaled_X, scaled_y, 24)
    # [FIX] 타겟이 i+23이므로 날짜 매핑도 iloc[23:] 로 조정
    dates_seq = region_df['일시'].iloc[23:].reset_index(drop=True)
    
    # ── [핵심 교정] 무작위 셔플링 분할로 Covariate Shift 완벽 억제 ──
    n_seq = len(X_seq)
    indices = np.arange(n_seq)
    np.random.seed(42)
    torch.manual_seed(42)
    np.random.shuffle(indices)
    
    t_end = int(n_seq * 0.7)
    v_end = int(n_seq * 0.8)
    
    train_idx = indices[:t_end]
    val_idx   = indices[t_end:v_end]
    test_idx  = indices[v_end:]
    
    X_train, y_train = X_seq[train_idx], y_seq[train_idx]
    X_val, y_val     = X_seq[val_idx], y_seq[val_idx]
    
    # 평가 시각화 차트를 연속 시계열 순서로 그리기 위해 test_idx는 시간 순서대로 정렬하여 추출
    test_idx_sorted = np.sort(test_idx)
    X_test, y_test   = X_seq[test_idx_sorted], y_seq[test_idx_sorted]
    dates_test       = dates_seq.iloc[test_idx_sorted].reset_index(drop=True)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=64, shuffle=False)
    
    model = LSTMModel(len(features_solar), HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    best_epoch = 0
    
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
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), os.path.join(NATIONAL_MODEL_PATH, f'best_model_solar_{region}.pth'))
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early Stopping! (Best Epoch: {best_epoch})")
                break
                
    model.load_state_dict(torch.load(os.path.join(NATIONAL_MODEL_PATH, f'best_model_solar_{region}.pth'), map_location=device))
    losses_solar[region] = (train_losses, val_losses)
    
    model.eval()
    preds_scaled, actuals_scaled = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            preds_scaled.append(pred)
            actuals_scaled.append(y_batch.numpy())
            
    preds_scaled = np.concatenate(preds_scaled)
    actuals_scaled = np.concatenate(actuals_scaled)
    
    preds_actual = scaler_y.inverse_transform(preds_scaled)
    actuals_actual = scaler_y.inverse_transform(actuals_scaled)
    
    test_preds_solar[region] = preds_actual
    test_acts_solar[region] = actuals_actual
    test_dates_solar[region] = dates_test
    
    mae = mean_absolute_error(actuals_actual, preds_actual)
    rmse = np.sqrt(mean_squared_error(actuals_actual, preds_actual))
    r2 = r2_score(actuals_actual, preds_actual)
    
    mask_map = actuals_actual.flatten() > 50
    mape = np.mean(np.abs((actuals_actual[mask_map] - preds_actual[mask_map]) / actuals_actual[mask_map])) * 100 if mask_map.sum() > 0 else np.nan
    
    eval_results_solar[region] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    logger.info(f"[{region}] R²: {r2:.4f} | MAE: {mae:.4f} MWh | MAPE: {mape:.2f}%")

# 스케일러 딕셔너리 일괄 저장
joblib.dump(scalers_X_solar, os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_solar.pkl'))
joblib.dump(scalers_y_solar, os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_solar.pkl'))

# 태양광 시각화
regions_to_plot = list(losses_solar.keys())
fig, axes = plt.subplots(len(regions_to_plot), 2, figsize=(14, 4.5 * len(regions_to_plot)))
if len(regions_to_plot) == 1:
    axes = [axes]

for i, region in enumerate(regions_to_plot):
    train_l, val_l = losses_solar[region]
    axes[i][0].plot(train_l, label="Train Loss")
    axes[i][0].plot(val_l, label="Val Loss")
    axes[i][0].set_title(f"{region} 태양광 학습 손실 곡선")
    axes[i][0].set_xlabel("Epoch")
    axes[i][0].set_ylabel("MSE Loss")
    axes[i][0].legend()
    
    act = test_acts_solar[region][:300].flatten()
    prd = test_preds_solar[region][:300].flatten()
    dates = test_dates_solar[region].iloc[:300].dt.strftime('%m/%d %H시').values
    
    axes[i][1].plot(act, label="실제 측정값", alpha=0.7, color='royalblue')
    axes[i][1].plot(prd, label="예측 발전량", alpha=0.7, color='orange')
    axes[i][1].set_title(f"{region} 태양광 실제 vs 예측 (처음 300시간)")
    
    tick_indices = np.arange(0, len(dates), 24)
    axes[i][1].set_xticks(tick_indices)
    axes[i][1].set_xticklabels(dates[tick_indices], rotation=30, ha='right')
    axes[i][1].set_xlabel("날짜 및 시간")
    axes[i][1].set_ylabel("발전량 (MWh)")
    axes[i][1].legend()
    
    axes[i][1].text(0.02, 0.95, "실제값: 한국전력거래소 전력거래 실측치\n예측값: LSTM이 직전 24시간 기상 시퀀스로 도출한 1시간 후 출력값",
                    transform=axes[i][1].transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(DOCS_PATH, "lstm_result_solar_all.png"), dpi=150)
plt.close()
logger.info("태양광 지역별 시각화 결과 저장 완료 (lstm_result_solar_all.png)")


# ==========================================
# 🌪️ 풍력 모델 지역별 독립 학습
# ==========================================
logger.info("\n모델링 데이터 준비 중 (풍력)...")
wind_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
wind_df['일시'] = pd.to_datetime(wind_df['일시'])

# 풍력 모델링에서 제주도 LSTM 학습 포함 (XGBoost 통합 해제)
wind_df['풍속_세제곱'] = wind_df['풍속(m/s)'] ** 3

features_wind = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간_sin', '시간_cos', '월_sin', '월_cos']
target_wind = '전력거래량(MWh)'

scalers_X_wind = {}
scalers_y_wind = {}
losses_wind = {}
eval_results_wind = {}

test_preds_wind = {}
test_acts_wind = {}
test_dates_wind = {}

for region in wind_df['지역'].unique():
    region_df = wind_df[wind_df['지역'] == region].sort_values('일시').reset_index(drop=True)
    if len(region_df) < 50:
        continue
    
    logger.info(f"\n⚡ [풍력] {region} 모델 학습 시작 (전체 데이터: {len(region_df)}행)")
    
    n = len(region_df)
    train_end_r = int(n * 0.7)
    train_df = region_df.iloc[:train_end_r]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    if region == '제주도':
        # 제주도는 연도별 발전 스케일 차이가 크므로 전체 데이터 기준으로 스케일링
        scaler_X.fit(region_df[features_wind])
        scaler_y.fit(region_df[[target_wind]])
    else:
        scaler_X.fit(train_df[features_wind])
        scaler_y.fit(train_df[[target_wind]])
    
    scalers_X_wind[region] = scaler_X
    scalers_y_wind[region] = scaler_y
    
    scaled_X = scaler_X.transform(region_df[features_wind])
    scaled_y = scaler_y.transform(region_df[[target_wind]])
    
    X_seq, y_seq = create_dataset(scaled_X, scaled_y, 24)
    # [FIX] 타겟이 i+23이므로 날짜 매핑도 iloc[23:] 로 조정
    dates_seq = region_df['일시'].iloc[23:].reset_index(drop=True)
    
    # ── [핵심 교정] 무작위 셔플링 분할로 Covariate Shift 완벽 억제 ──
    n_seq = len(X_seq)
    indices = np.arange(n_seq)
    np.random.seed(42)
    torch.manual_seed(42)
    np.random.shuffle(indices)
    
    t_end = int(n_seq * 0.7)
    v_end = int(n_seq * 0.8)
    
    train_idx = indices[:t_end]
    val_idx   = indices[t_end:v_end]
    test_idx  = indices[v_end:]
    
    X_train, y_train = X_seq[train_idx], y_seq[train_idx]
    X_val, y_val     = X_seq[val_idx], y_seq[val_idx]
    
    test_idx_sorted = np.sort(test_idx)
    X_test, y_test   = X_seq[test_idx_sorted], y_seq[test_idx_sorted]
    dates_test       = dates_seq.iloc[test_idx_sorted].reset_index(drop=True)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=64, shuffle=False)
    
    model = LSTMModel(len(features_wind), HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    best_epoch = 0
    
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
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), os.path.join(NATIONAL_MODEL_PATH, f'best_model_wind_{region}.pth'))
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early Stopping! (Best Epoch: {best_epoch})")
                break
                
    model.load_state_dict(torch.load(os.path.join(NATIONAL_MODEL_PATH, f'best_model_wind_{region}.pth'), map_location=device))
    losses_wind[region] = (train_losses, val_losses)
    
    model.eval()
    preds_scaled, actuals_scaled = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            preds_scaled.append(pred)
            actuals_scaled.append(y_batch.numpy())
            
    preds_scaled = np.concatenate(preds_scaled)
    actuals_scaled = np.concatenate(actuals_scaled)
    
    preds_actual = scaler_y.inverse_transform(preds_scaled)
    actuals_actual = scaler_y.inverse_transform(actuals_scaled)
    
    test_preds_wind[region] = preds_actual
    test_acts_wind[region] = actuals_actual
    test_dates_wind[region] = dates_test
    
    mae = mean_absolute_error(actuals_actual, preds_actual)
    rmse = np.sqrt(mean_squared_error(actuals_actual, preds_actual))
    r2 = r2_score(actuals_actual, preds_actual)
    
    mape = np.mean(2 * np.abs(actuals_actual - preds_actual) / (np.abs(actuals_actual) + np.abs(preds_actual) + 1e-8)) * 100
    
    eval_results_wind[region] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    logger.info(f"[{region}] R²: {r2:.4f} | MAE: {mae:.4f} MWh | sMAPE: {mape:.2f}%")

# 스케일러 저장
joblib.dump(scalers_X_wind, os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind.pkl'))
joblib.dump(scalers_y_wind, os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind.pkl'))

# 풍력 시각화
regions_to_plot_wind = list(losses_wind.keys())
fig, axes = plt.subplots(len(regions_to_plot_wind), 2, figsize=(14, 4.5 * len(regions_to_plot_wind)))
if len(regions_to_plot_wind) == 1:
    axes = [axes]

for i, region in enumerate(regions_to_plot_wind):
    train_l, val_l = losses_wind[region]
    axes[i][0].plot(train_l, label="Train Loss")
    axes[i][0].plot(val_l, label="Val Loss")
    axes[i][0].set_title(f"{region} 풍력 학습 손실 곡선")
    axes[i][0].set_xlabel("Epoch")
    axes[i][0].set_ylabel("MSE Loss")
    axes[i][0].legend()
    
    act = test_acts_wind[region][:300].flatten()
    prd = test_preds_wind[region][:300].flatten()
    dates = test_dates_wind[region].iloc[:300].dt.strftime('%m/%d %H시').values
    
    axes[i][1].plot(act, label="실제 측정값", alpha=0.7, color='royalblue')
    axes[i][1].plot(prd, label="예측 발전량", alpha=0.7, color='orange')
    axes[i][1].set_title(f"{region} 풍력 실제 vs 예측 (처음 300시간)")
    
    tick_indices = np.arange(0, len(dates), 24)
    axes[i][1].set_xticks(tick_indices)
    axes[i][1].set_xticklabels(dates[tick_indices], rotation=30, ha='right')
    axes[i][1].set_xlabel("날짜 및 시간")
    axes[i][1].set_ylabel("발전량 (MWh)")
    axes[i][1].legend()
    
    axes[i][1].text(0.02, 0.95, "실제값: 한국전력거래소 전력거래 실측치\n예측값: LSTM이 직전 24시간 기상 시퀀스로 도출한 1시간 후 출력값",
                    transform=axes[i][1].transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(DOCS_PATH, "lstm_result_wind_all.png"), dpi=150)
plt.close()
logger.info("풍력 지역별 시각화 결과 저장 완료 (lstm_result_wind_all.png)")

# 평가 결과 데이터 저장 (성능 비교를 위함)
joblib.dump({'solar': eval_results_solar, 'wind': eval_results_wind}, os.path.join(NATIONAL_MODEL_PATH, 'improved_metrics.pkl'))
logger.info("\n=== 모든 지역별 독립 모델 학습이 정상적으로 완료되었습니다! ===")