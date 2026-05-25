import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

DATASET_PATH = r'./dataset'

print("데이터 로딩 중...")
wind_df = pd.read_csv(os.path.join(DATASET_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
wind_df['일시'] = pd.to_datetime(wind_df['일시'])

# 풍속 세제곱 파생 변수 추가
if '풍속_세제곱' not in wind_df.columns:
    wind_df['풍속_세제곱'] = wind_df['풍속(m/s)'] ** 3

# 🔥 제주도 데이터만 분리
jeju_df = wind_df[wind_df['지역'] == '제주도'].sort_values('일시').reset_index(drop=True)

# 피처 및 타겟 설정 (발전량 컬럼 자동 탐색)
features_wind = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간', '월']
target_cols = [col for col in jeju_df.columns if '발전량' in col]
target_col = target_cols[0] if target_cols else jeju_df.columns[-1]

# 스케일링
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_X = scaler_X.fit_transform(jeju_df[features_wind])
scaled_y = scaler_y.fit_transform(jeju_df[[target_col]])

# 24시간 시퀀스 데이터를 1차원(Flatten)으로 변환하여 XGBoost에 맞춤
seq_length = 24
X_seq, y_seq = [], []
for i in range(len(scaled_X) - seq_length):
    X_seq.append(scaled_X[i:i+seq_length].flatten())
    y_seq.append(scaled_y[i+seq_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Train/Test 분할 (8:2)
split_idx = int(len(X_seq) * 0.8)
X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
X_test, y_test = X_seq[split_idx:], y_seq[split_idx:]

print("제주도 전용 XGBoost 모델 학습 시작...")
model_jeju = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model_jeju.fit(X_train, y_train.ravel())

# 평가
preds = model_jeju.predict(X_test).reshape(-1, 1)
preds_actual = scaler_y.inverse_transform(preds)
y_test_actual = scaler_y.inverse_transform(y_test)

print(f"✅ 제주도 전용 모델 R² Score: {r2_score(y_test_actual, preds_actual):.4f}")

# 결과물 저장
joblib.dump(model_jeju, os.path.join(DATASET_PATH, 'best_model_wind_jeju_xgb.pkl'))
joblib.dump({'제주도': scaler_X}, os.path.join(DATASET_PATH, 'scalers_X_wind_jeju.pkl'))
joblib.dump({'제주도': scaler_y}, os.path.join(DATASET_PATH, 'scalers_y_wind_jeju.pkl'))
print("🎉 제주도 모델 및 스케일러 저장 완료!")