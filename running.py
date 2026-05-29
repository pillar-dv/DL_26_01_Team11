import os
import torch
import numpy as np
import pandas as pd
import joblib
import torch.nn as nn
from datetime import datetime

# ── LSTM 모델 정의 (태양광용) ─────────────────────
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

DATASET_PATH   = r'./dataset'
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_solar = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간', '월', '일사(MJ/m2)']
features_xgb   = [
    '기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)',
    '습도(%)', '현지기압(hPa)', '전운량(10분위)',
    '시간', '월', '요일', '주말여부',
    'lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_168',
    'rolling_mean_6', 'rolling_mean_24', 'rolling_std_24'
]
target        = '전력거래량(MWh)'
target_wind   = '전력거래량(MWh)'

# ── 예측 지역 설정 ────────────────────────────────
target_region_solar = '경상북도'  # 태양광 예측 지역
target_region_wind  = '강원도'    # 풍력 예측 지역

# ── 태양광 LSTM 모델 로드 ─────────────────────────
model_solar = LSTMModel(len(features_solar), 128, 2, 1, 0.3).to(device)
model_solar.load_state_dict(torch.load(
    os.path.join(DATASET_PATH, f'best_model_solar_{target_region_solar}.pth'),
    map_location=device
))
model_solar.eval()
scalers_X_solar = joblib.load(os.path.join(DATASET_PATH, 'scalers_X_solar.pkl'))
scalers_y_solar = joblib.load(os.path.join(DATASET_PATH, 'scalers_y_solar.pkl'))

# ── 풍력 XGBoost 모델 로드 ────────────────────────
model_wind_xgb = joblib.load(os.path.join(DATASET_PATH, f'xgb_wind_{target_region_wind}.pkl'))

print("모델 로드 완료")

# ── 데이터 로드 ───────────────────────────────────
solar_df = pd.read_csv(os.path.join(DATASET_PATH, 'solar_integrated_dataset.csv'), encoding='utf-8-sig')
solar_df['일시'] = pd.to_datetime(solar_df['일시'])

wind_df = pd.read_csv(os.path.join(DATASET_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
wind_df['일시'] = pd.to_datetime(wind_df['일시'])
wind_df         = wind_df[wind_df['지역'] != '제주도'].reset_index(drop=True)

print(f"태양광 가용 지역: {sorted(solar_df['지역'].unique().tolist())}")
print(f"풍력   가용 지역: {sorted(wind_df['지역'].unique().tolist())}")

# ── XGBoost용 풍력 피처 생성 함수 ────────────────
def make_wind_features_predict(df):
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
    df = df.dropna().reset_index(drop=True)
    return df

# ── 태양광 LSTM 예측 함수 ─────────────────────────
def predict_solar(region):
    if region not in scalers_X_solar:
        print(f"{region} 태양광 스케일러가 없습니다.")
        return 0.0

    region_df = solar_df[solar_df['지역'] == region].copy()
    region_df = region_df.sort_values('일시').reset_index(drop=True)

    if len(region_df) < 24:
        print(f"{region} 24시간 데이터 부족")
        return 0.0

    # 야간 시간대 처리
    current_hour = datetime.now().hour
    if not (6 <= current_hour <= 19):
        print(f"태양광: 야간 시간대 ({current_hour}시) → 0 MWh")
        return 0.0

    scaled_X     = scalers_X_solar[region].transform(region_df[features_solar])
    input_tensor = torch.tensor(scaled_X[-24:], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model_solar(input_tensor).cpu().numpy()
        pred_actual = scalers_y_solar[region].inverse_transform(pred_scaled)

    return float(np.maximum(pred_actual[0][0], 0))

# ── 풍력 XGBoost 예측 함수 ───────────────────────
def predict_wind_xgb(region):
    region_df = wind_df[wind_df['지역'] == region].copy()

    if len(region_df) < 200:
        print(f"{region} 풍력 데이터 부족")
        return 0.0

    region_df = make_wind_features_predict(region_df)

    if len(region_df) == 0:
        print(f"{region} 피처 생성 후 데이터 없음")
        return 0.0

    last_row = region_df[features_xgb].iloc[[-1]]
    pred     = np.maximum(model_wind_xgb.predict(last_row)[0], 0)
    return float(pred)

# ── 예측 실행 ─────────────────────────────────────
solar_pred = predict_solar(target_region_solar)
wind_pred  = predict_wind_xgb(target_region_wind)

current_hour = datetime.now().hour
print(f"\n===== 예측 결과 =====")
print(f"현재 시각  : {current_hour}시")
print(f"태양광 ({target_region_solar}) [LSTM]    : {solar_pred:.2f} MWh")
print(f"풍력   ({target_region_wind}) [XGBoost] : {wind_pred:.2f} MWh")
print(f"합계                                    : {solar_pred + wind_pred:.2f} MWh")
