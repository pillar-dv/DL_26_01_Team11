import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import os
import torch
import numpy as np
import pandas as pd
import joblib
import torch.nn as nn

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
NATIONAL_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'national')
device = torch.device("cpu")  # CPU 모드 동기화

# 삼각함수 피처 반영 완료
features_solar = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간_sin', '시간_cos', '월_sin', '월_cos', '일사(MJ/m2)']
features_wind  = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간_sin', '시간_cos', '월_sin', '월_cos']
target   = '전력거래량(MWh)'

# 스케일러 사전 로드
scalers_X_solar = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_solar.pkl'))
scalers_y_solar = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_solar.pkl'))
scalers_X_wind = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind.pkl'))
scalers_y_wind = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind.pkl'))

# 제주도 풍력 스케일러 병합
if os.path.exists(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind_jeju.pkl')):
    scalers_X_wind.update(joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind_jeju.pkl')))
if os.path.exists(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind_jeju.pkl')):
    scalers_y_wind.update(joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind_jeju.pkl')))

print("스케일러 딕셔너리 로드 완료")

# ── 최근 데이터 로드 ──
solar_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'solar_integrated_dataset.csv'), encoding='utf-8-sig')
solar_df['일시'] = pd.to_datetime(solar_df['일시'])

wind_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
wind_df['일시'] = pd.to_datetime(wind_df['일시'])
wind_df['풍속_세제곱'] = wind_df['풍속(m/s)'] ** 3

# ── 예측 함수 (지자체별 모델 동적 로드 구조) ──
def predict(fuel_type, df, scalers_X, scalers_y, region, features):
    if region not in scalers_X:
        print(f"{region} 데이터 또는 스케일러가 없습니다.")
        return 0.0
    
    region_df = df[df['지역'] == region].copy()
    region_df = region_df.sort_values('일시').reset_index(drop=True)
    
    if len(region_df) < 24:
        print(f"{region}의 24시간 데이터가 부족합니다.")
        return 0.0
        
    # 지역별 특화 가중치 모델 동적 로드
    model_path = os.path.join(NATIONAL_MODEL_PATH, f'best_model_{fuel_type}_{region}.pth')
    if not os.path.exists(model_path):
        print(f"{region} {fuel_type} 가중치 파일이 존재하지 않습니다: {model_path}")
        return 0.0
        
    model = LSTMModel(len(features), 64, 1, 1, 0.2).to(device)  # 64, 1, 0.2 규격 동기화
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # transform만 수행하여 입력 텐서 준비
    scaled_X  = scalers_X[region].transform(region_df[features])
    input_tensor = torch.tensor(scaled_X[-24:], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()
        pred_actual = scalers_y[region].inverse_transform(pred_scaled)
        
    return float(np.maximum(pred_actual[0][0], 0))

# ── 예측 실행 ──
target_region_solar = '제주도'
target_region_wind  = '제주도'

solar_pred = predict('solar', solar_df, scalers_X_solar, scalers_y_solar, target_region_solar, features_solar)
wind_pred  = predict('wind',  wind_df,  scalers_X_wind,  scalers_y_wind,  target_region_wind,  features_wind)

print(f"\n===== 예측 결과 =====")
print(f"태양광 ({target_region_solar}): {solar_pred:.2f} MWh")
print(f"풍력   ({target_region_wind }) : {wind_pred:.2f} MWh")
print(f"합계                          : {solar_pred + wind_pred:.2f} MWh")