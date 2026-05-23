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

DATASET_PATH = r'./dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 피처 정의 (풍속 세제곱 추가됨)
features_solar = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간', '월', '일사(MJ/m2)']
features_wind  = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간', '월']
target   = '전력거래량(MWh)'

# ── 모델 및 스케일러 로드 (클라우드 환경 대응 os.path.join 적용) ──
# 1. 태양광
model_solar = LSTMModel(len(features_solar), 128, 2, 1, 0.3).to(device)
model_solar.load_state_dict(torch.load(os.path.join(DATASET_PATH, 'best_model.pth'), map_location=device))
model_solar.eval()

scalers_X_solar = joblib.load(os.path.join(DATASET_PATH, 'scalers_X_solar.pkl'))
scalers_y_solar = joblib.load(os.path.join(DATASET_PATH, 'scalers_y_solar.pkl'))

# 2. 풍력
model_wind = LSTMModel(len(features_wind), 128, 2, 1, 0.3).to(device)
model_wind.load_state_dict(torch.load(os.path.join(DATASET_PATH, 'best_model_wind.pth'), map_location=device))
model_wind.eval()

scalers_X_wind = joblib.load(os.path.join(DATASET_PATH, 'scalers_X_wind.pkl'))
scalers_y_wind = joblib.load(os.path.join(DATASET_PATH, 'scalers_y_wind.pkl'))

print("모델 및 스케일러 로드 완료")

# ── 최근 데이터 로드 (추론용 과거 24시간 추출) ──
solar_df = pd.read_csv(os.path.join(DATASET_PATH, 'solar_integrated_dataset.csv'), encoding='utf-8-sig')
solar_df['일시'] = pd.to_datetime(solar_df['일시'])

wind_df = pd.read_csv(os.path.join(DATASET_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
wind_df['일시'] = pd.to_datetime(wind_df['일시'])
wind_df['풍속_세제곱'] = wind_df['풍속(m/s)'] ** 3  # 추론 시에도 파생 변수 생성 필수

# ── 예측 함수 (fit 제거, transform만 수행) ──
def predict(model, df, scalers_X, scalers_y, region, features):
    if region not in scalers_X:
        print(f"{region} 데이터 또는 스케일러가 없습니다.")
        return 0.0
    
    region_df = df[df['지역'] == region].copy()
    region_df = region_df.sort_values('일시').reset_index(drop=True)
    
    if len(region_df) < 24:
        print(f"{region}의 24시간 데이터가 부족합니다.")
        return 0.0
        
    # 이미 학습된 스케일러로 transform만 수행 (미래 데이터 누수 및 잣대 틀어짐 방지)
    scaled_X  = scalers_X[region].transform(region_df[features])
    input_tensor = torch.tensor(scaled_X[-24:], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()
        pred_actual = scalers_y[region].inverse_transform(pred_scaled)
        
    return float(np.maximum(pred_actual[0][0], 0))

# ── 예측 실행 ──
target_region_solar = '제주도'
target_region_wind  = '전라북도'

solar_pred = predict(model_solar, solar_df, scalers_X_solar, scalers_y_solar, target_region_solar, features_solar)
wind_pred  = predict(model_wind,  wind_df,  scalers_X_wind,  scalers_y_wind,  target_region_wind,  features_wind)

print(f"\n===== 예측 결과 =====")
print(f"태양광 ({target_region_solar}): {solar_pred:.2f} MWh")
print(f"풍력   ({target_region_wind }) : {wind_pred:.2f} MWh")
print(f"합계                          : {solar_pred + wind_pred:.2f} MWh")