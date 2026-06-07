import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
device = torch.device("cpu")

# 난수 시드 고정
torch.manual_seed(42)
np.random.seed(42)

# LSTM 모델 정의 (동일 구조 적용)
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
    # [FIX] 타겟을 윈돈우 끝 시점(i+23)으로 수정
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)

print("데이터 로딩 중...")
wind_df = pd.read_csv(os.path.join(DATASET_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
wind_df['일시'] = pd.to_datetime(wind_df['일시'])

if '풍속_세제곱' not in wind_df.columns:
    wind_df['풍속_세제곱'] = wind_df['풍속(m/s)'] ** 3

# 제주도 데이터만 분리
jeju_df = wind_df[wind_df['지역'] == '제주도'].sort_values('일시').reset_index(drop=True)

# 주기적 특성 인코딩 전처리 추가
jeju_df['시간'] = jeju_df['일시'].dt.hour
jeju_df['월'] = jeju_df['일시'].dt.month

jeju_df['시간_sin'] = np.sin(2 * np.pi * jeju_df['시간'] / 24.0)
jeju_df['시간_cos'] = np.cos(2 * np.pi * jeju_df['시간'] / 24.0)
jeju_df['월_sin'] = np.sin(2 * np.pi * (jeju_df['월'] - 1) / 12.0)
jeju_df['월_cos'] = np.cos(2 * np.pi * (jeju_df['월'] - 1) / 12.0)

# 피처 및 타겟 설정
features_wind = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간_sin', '시간_cos', '월_sin', '월_cos']
target_col = '전력거래량(MWh)'

# 스케일러 학습 (70% 기점)
n = len(jeju_df)
train_end_r = int(n * 0.7)
train_df = jeju_df.iloc[:train_end_r]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(jeju_df[features_wind])
scaler_y.fit(jeju_df[[target_col]])

scaled_X = scaler_X.transform(jeju_df[features_wind])
scaled_y = scaler_y.transform(jeju_df[[target_col]])

# 시퀀스 데이터 생성
X_seq, y_seq = create_dataset(scaled_X, scaled_y, 24)

# ── [핵심 교정] 무작위 셔플링 분할로 Covariate Shift 완벽 억제 ──
n_seq = len(X_seq)
indices = np.arange(n_seq)
np.random.seed(42)
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

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=64, shuffle=False)

# 모델 정의 및 하이퍼파라미터 규격 싱크
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 1
DROPOUT = 0.2
EPOCHS = 100
LR = 0.001

model = LSTMModel(len(features_wind), HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
criterion = nn.MSELoss()

print("제주도 전용 LSTM 모델 학습 시작...")
best_val_loss = float('inf')
patience = 15
counter = 0

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
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), os.path.join(os.path.join(PROJECT_ROOT, 'models', 'national'), 'best_model_wind_제주도.pth'))
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 복원 및 평가
model.load_state_dict(torch.load(os.path.join(os.path.join(PROJECT_ROOT, 'models', 'national'), 'best_model_wind_제주도.pth'), map_location=device))
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
y_test_actual = scaler_y.inverse_transform(actuals_scaled)

print(f"[Jeju Wind LSTM] R2 Score: {r2_score(y_test_actual, preds_actual):.4f}")

# 스케일러 저장 (기존 scalers_X_wind.pkl와 호환성 맞춤을 위해 scalers_X_wind_jeju.pkl로 저장)
# app.py와 running.py에서 스케일러 딕셔너리에 병합할 수 있도록 딕셔너리 형태로 덤프
joblib.dump({'제주도': scaler_X}, os.path.join(os.path.join(PROJECT_ROOT, 'models', 'national'), 'scalers_X_wind_jeju.pkl'))
joblib.dump({'제주도': scaler_y}, os.path.join(os.path.join(PROJECT_ROOT, 'models', 'national'), 'scalers_y_wind_jeju.pkl'))
print("Jeju LSTM model and scalers successfully saved!")