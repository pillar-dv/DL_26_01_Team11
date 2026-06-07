import pandas as pd
import numpy as np
import os
import glob
import sys
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
GPS_OUTPUT_PATH = os.path.join(PROCESSED_DATA_PATH, 'gps_micro')
MICRO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'micro_gps')
DOCS_PATH = os.path.join(PROJECT_ROOT, 'docs')

os.makedirs(GPS_OUTPUT_PATH, exist_ok=True)
os.makedirs(MICRO_MODEL_PATH, exist_ok=True)
os.makedirs(DOCS_PATH, exist_ok=True)

# 로깅 설정
log_file_path = os.path.join(DOCS_PATH, 'train_gps_log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# GRS80 및 위경도 최단 거리 공간 조인 모듈
STATIONS_COORDS = {
    184: {'name': '제주', 'lat': 33.5141, 'lon': 126.5297},
    185: {'name': '고산', 'lat': 33.2938, 'lon': 126.1628},
    188: {'name': '성산', 'lat': 33.3868, 'lon': 126.8806},
    189: {'name': '서귀포', 'lat': 33.2462, 'lon': 126.5654}
}

def find_nearest_station(target_lat, target_lon):
    min_dist = float('inf')
    best_station = None
    for code, coords in STATIONS_COORDS.items():
        # 단순 유클리드 평면 거리 공식 사용
        dist = np.sqrt((coords['lat'] - target_lat)**2 + (coords['lon'] - target_lon)**2)
        if dist < min_dist:
            min_dist = dist
            best_station = (code, coords['name'])
    return best_station

# 기상 원천 데이터 로드
logger.info("기상 데이터 통합 중...")
weather_files = glob.glob(os.path.join(RAW_DATA_PATH, '*_기상.csv'))
weather_dfs = []
for wf in weather_files:
    try:
        w_df = pd.read_csv(wf, encoding='cp949')
        weather_dfs.append(w_df)
    except:
        w_df = pd.read_csv(wf, encoding='utf-8')
        weather_dfs.append(w_df)
        
weather_df = pd.concat(weather_dfs, ignore_index=True)
weather_df.columns = [
    '지점', '지점명', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', 
    '풍향(16방위)', '습도(%)', '현지기압(hPa)', '일사(MJ/m2)', '전운량(10분위)'
]
weather_df['일시'] = pd.to_datetime(weather_df['일시'])

# GPS 기반 기상 지점 공간 조인 처리
logger.info("GPS 좌표기반 최인접 기상관측 지점 공간 조인 연산 구동...")
loc_file = glob.glob(os.path.join(RAW_DATA_PATH, '*위치정보*.csv'))[0]
loc_df = pd.read_csv(loc_file, encoding='cp949')

# 단지 단계별 평균 GPS 계산
stages_map = {
    '한경1': '한경1',
    '한경2': '한경2',
    '성산1': '성산1',
    '삼달': '성산2'  # 성산 2단계 매핑
}

farm_gps = {}
for farm_name, target_key in stages_map.items():
    sub = loc_df[loc_df['단지명'] == farm_name]
    if not sub.empty:
        avg_lat = sub['위도(lat)'].mean()
        avg_lon = sub['경도(lon)'].mean()
        code, s_name = find_nearest_station(avg_lat, avg_lon)
        farm_gps[target_key] = {'lat': avg_lat, 'lon': avg_lon, 'station_code': code, 'station_name': s_name}
        logger.info(f"단지 [{target_key}] 위치 (lat:{avg_lat:.4f}, lon:{avg_lon:.4f}) ➔ 인접 기상관측소 [{s_name}({code})] 매핑 완료")

# 남부발전 실적 데이터 변환 함수 (kWh -> MWh 보정 탑재)
def process_plant_stage(file_pattern, stage, plant_name):
    files = glob.glob(os.path.join(RAW_DATA_PATH, file_pattern))
    if not files:
        return None
    df = pd.read_csv(files[0], encoding='cp949')
    sub_df = df[df['단계'] == stage].copy()
    
    melted = pd.melt(
        sub_df, 
        id_vars=['년월일'], 
        value_vars=[str(i) for i in range(1, 25)], 
        var_name='시간', 
        value_name='전력거래량(MWh)'
    )
    melted['일시'] = (
        pd.to_datetime(melted['년월일'])
        + pd.to_timedelta(melted['시간'].astype(int) - 1, unit='h')
    )
    # 안전한 수치 변환 (문자열 결측치 처리)
    melted['전력거래량(MWh)'] = pd.to_numeric(melted['전력거래량(MWh)'], errors='coerce')
    melted['전력거래량(MWh)'] = melted['전력거래량(MWh)'].fillna(0.0) / 1000.0
    return melted[['일시', '전력거래량(MWh)']].sort_values('일시').reset_index(drop=True)

# 초국소 데이터 빌드 및 정합
gps_datasets = {}
for target_key, info in farm_gps.items():
    if '한경' in target_key:
        stage = 1 if '1' in target_key else 2
        p_df = process_plant_stage('*한경풍력*.csv', stage, '한경')
    else:
        stage = 1 if '1' in target_key else 2
        p_df = process_plant_stage('*성산풍력*.csv', stage, '성산')
        
    if p_df is not None:
        # 매핑된 기상소 데이터 추출
        s_code = info['station_code']
        station_weather = weather_df[weather_df['지점'] == s_code].copy()
        
        # 병합
        merged = pd.merge(p_df, station_weather, on='일시', how='inner')
        
        # 기상 피처 전처리 (보간 및 삼각함수 시간 변수 생성)
        merged['풍속_세제곱'] = merged['풍속(m/s)'] ** 3
        numeric_cols = merged.select_dtypes(include='number').columns
        merged[numeric_cols] = merged[numeric_cols].interpolate(method='linear').bfill().ffill()
        
        merged['시간'] = merged['일시'].dt.hour
        merged['월'] = merged['일시'].dt.month
        merged['시간_sin'] = np.sin(2 * np.pi * merged['시간'] / 24.0)
        merged['시간_cos'] = np.cos(2 * np.pi * merged['시간'] / 24.0)
        merged['월_sin'] = np.sin(2 * np.pi * (merged['월'] - 1) / 12.0)
        merged['월_cos'] = np.cos(2 * np.pi * (merged['월'] - 1) / 12.0)
        
        gps_datasets[target_key] = merged
        merged.to_csv(os.path.join(GPS_OUTPUT_PATH, f'wind_dataset_{target_key}.csv'), index=False, encoding='utf-8-sig')
        logger.info(f"초국소 GPS 데이터셋 생성 완료: wind_dataset_{target_key}.csv (행 개수: {len(merged)})")

# ── LSTM 모델 정의 ──
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_dataset(X, y, time_steps=24):
    # [FIX] 타겟을 요도우 끝 시점(i+23)으로 수정 (일사량/풍속 피크 시간 정렬)
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)

features_wind = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간_sin', '시간_cos', '월_sin', '월_cos']
target_wind = '전력거래량(MWh)'

scalers_X_wind = {}
scalers_y_wind = {}
eval_results_wind = {}

# 단지 단계별 학습 루프 돌리기 (신속 학습을 위해 Epochs 20 으로 경량화)
EPOCHS = 20
BATCH_SIZE = 64
device = torch.device("cpu")

for farm_key, data in gps_datasets.items():
    logger.info(f"\n⚡ [GPS 초국소 모델] {farm_key} LSTM 모델 학습 시작...")
    
    n = len(data)
    train_end_r = int(n * 0.7)
    train_df = data.iloc[:train_end_r]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(data[features_wind])
    scaler_y.fit(data[[target_wind]])
    
    scalers_X_wind[farm_key] = scaler_X
    scalers_y_wind[farm_key] = scaler_y
    
    scaled_X = scaler_X.transform(data[features_wind])
    scaled_y = scaler_y.transform(data[[target_wind]])
    
    X_seq, y_seq = create_dataset(scaled_X, scaled_y, 24)
    
    # 셔플링 분할 기법 적용
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
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
    
    model = LSTMModel(len(features_wind), 64, 1, 1, 0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MICRO_MODEL_PATH, f'best_model_wind_{farm_key}.pth'))
            
    # 최종 평가
    model.load_state_dict(torch.load(os.path.join(MICRO_MODEL_PATH, f'best_model_wind_{farm_key}.pth')))
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            pred = model(X_batch).numpy()
            preds_scaled.append(pred)
    preds_scaled = np.concatenate(preds_scaled)
    
    preds_actual = scaler_y.inverse_transform(preds_scaled)
    actuals_actual = scaler_y.inverse_transform(y_test)
    
    mae = mean_absolute_error(actuals_actual, preds_actual)
    rmse = np.sqrt(mean_squared_error(actuals_actual, preds_actual))
    r2 = r2_score(actuals_actual, preds_actual)
    
    eval_results_wind[farm_key] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    logger.info(f"🚀 [{farm_key} 결과] R²: {r2:.4f} | MAE: {mae:.4f} MWh | RMSE: {rmse:.4f} MWh")

# 스케일러 및 최종 메트릭 격리 덤프
joblib.dump(scalers_X_wind, os.path.join(MICRO_MODEL_PATH, 'scalers_X_wind.pkl'))
joblib.dump(scalers_y_wind, os.path.join(MICRO_MODEL_PATH, 'scalers_y_wind.pkl'))
joblib.dump(eval_results_wind, os.path.join(MICRO_MODEL_PATH, 'gps_metrics.pkl'))
logger.info("\n=== [성공] 모든 GPS 초국소 단지별 모델 학습 완료 및 models/micro_gps/ 에 분리 저장 완료! ===")
