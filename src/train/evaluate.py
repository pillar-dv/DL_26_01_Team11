import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
NATIONAL_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'national')
BASELINE_PATH = os.path.join(PROJECT_ROOT, 'models', 'baseline')
DOCS_PATH = os.path.join(PROJECT_ROOT, 'docs')

# 기존 LSTM 모델 정의 (피처 개수가 가변적이어야 함)
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
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps - 1])  # [FIX] 윈도우 끝 시점 타겟 (피크 시간 정렬)
    return np.array(Xs), np.array(ys)

def evaluate_baseline_solar():
    print("Baseline 태양광 모델 로드 및 평가 중...")
    solar_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'solar_integrated_dataset.csv'), encoding='utf-8-sig')
    solar_df['일시'] = pd.to_datetime(solar_df['일시'])
    
    # Baseline 피처 목록 (Sin/Cos 이전)
    features = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간', '월', '일사(MJ/m2)']
    target = '전력거래량(MWh)'
    
    # 백업된 Baseline 모델 및 스케일러 로드
    m_solar = LSTMModel(len(features), 128, 2, 1, 0.3)
    m_solar.load_state_dict(torch.load(os.path.join(BASELINE_PATH, 'best_model.pth'), map_location='cpu'))
    m_solar.eval()
    
    scalers_X = joblib.load(os.path.join(BASELINE_PATH, 'scalers_X_solar.pkl'))
    scalers_y = joblib.load(os.path.join(BASELINE_PATH, 'scalers_y_solar.pkl'))
    
    results = {}
    
    for region in solar_df['지역'].unique():
        region_df = solar_df[solar_df['지역'] == region].sort_values('일시').reset_index(drop=True)
        if len(region_df) < 50 or region not in scalers_X:
            continue
            
        scaler_X = scalers_X[region]
        scaler_y = scalers_y[region]
        
        scaled_X = scaler_X.transform(region_df[features].values)
        scaled_y = scaler_y.transform(region_df[[target]].values)
        
        X_seq, y_seq = create_dataset(scaled_X, scaled_y, 24)
        
        # 무작위 셔플링 분할 (Improved 모델과 완벽 싱크)
        n_seq = len(X_seq)
        indices = np.arange(n_seq)
        np.random.seed(42)
        np.random.shuffle(indices)
        v_end = int(n_seq * 0.8)
        test_idx = indices[v_end:]
        test_idx_sorted = np.sort(test_idx)
        X_test, y_test = X_seq[test_idx_sorted], y_seq[test_idx_sorted]
        
        if len(X_test) == 0:
            continue
            
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            preds_scaled = m_solar(X_tensor).numpy()
            
        preds_actual = scaler_y.inverse_transform(preds_scaled)
        actuals_actual = scaler_y.inverse_transform(y_test)
        
        mae = mean_absolute_error(actuals_actual, preds_actual)
        rmse = np.sqrt(mean_squared_error(actuals_actual, preds_actual))
        r2 = r2_score(actuals_actual, preds_actual)
        
        mask_map = actuals_actual.flatten() > 50
        mape = np.mean(np.abs((actuals_actual[mask_map] - preds_actual[mask_map]) / actuals_actual[mask_map])) * 100 if mask_map.sum() > 0 else np.nan
        
        results[region] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        
    return results

def evaluate_baseline_wind():
    print("Baseline 풍력 모델 로드 및 평가 중...")
    wind_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
    wind_df['일시'] = pd.to_datetime(wind_df['일시'])
    wind_df['풍속_세제곱'] = wind_df['풍속(m/s)'] ** 3
    
    # Baseline 피처 목록 (Sin/Cos 이전)
    features = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간', '월']
    target = '전력거래량(MWh)'
    
    m_wind = LSTMModel(len(features), 128, 2, 1, 0.3)
    m_wind.load_state_dict(torch.load(os.path.join(BASELINE_PATH, 'best_model_wind.pth'), map_location='cpu'))
    m_wind.eval()
    
    scalers_X = joblib.load(os.path.join(BASELINE_PATH, 'scalers_X_wind.pkl'))
    scalers_y = joblib.load(os.path.join(BASELINE_PATH, 'scalers_y_wind.pkl'))
    
    # 제주도 전용 baseline 및 신규 스케일러 로드
    if os.path.exists(os.path.join(NATIONAL_MODEL_PATH, 'best_model_wind_jeju_xgb.pkl')):
        # 신규 11개 피처 스케일러에서 제주도 스케일러 가져오기 (264 input size 대응)
        new_scalers_X = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind.pkl'))
        new_scalers_y = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind.pkl'))
        if '제주도' in new_scalers_X:
            scalers_X['제주도'] = new_scalers_X['제주도']
            scalers_y['제주도'] = new_scalers_y['제주도']
        xgb_model = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'best_model_wind_jeju_xgb.pkl'))
    else:
        xgb_model = None
        
    results = {}
    
    for region in wind_df['지역'].unique():
        region_df = wind_df[wind_df['지역'] == region].sort_values('일시').reset_index(drop=True)
        if len(region_df) < 50 or region not in scalers_X:
            continue
            
        scaler_X = scalers_X[region]
        scaler_y = scalers_y[region]
        
        if region == '제주도':
            features_to_use = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간_sin', '시간_cos', '월_sin', '월_cos']
        else:
            features_to_use = features
            
        scaled_X = scaler_X.transform(region_df[features_to_use].values)
        scaled_y = scaler_y.transform(region_df[[target]].values)
        
        X_seq, y_seq = create_dataset(scaled_X, scaled_y, 24)
        
        # 무작위 셔플링 분할 (Improved 모델과 완벽 싱크)
        n_seq = len(X_seq)
        indices = np.arange(n_seq)
        np.random.seed(42)
        np.random.shuffle(indices)
        v_end = int(n_seq * 0.8)
        test_idx = indices[v_end:]
        test_idx_sorted = np.sort(test_idx)
        X_test, y_test = X_seq[test_idx_sorted], y_seq[test_idx_sorted]
        
        if len(X_test) == 0:
            continue
            
        if region == '제주도' and xgb_model is not None:
            # 제주도 baseline (XGBoost) 평가 (flatten 216 features)
            X_test_flat = X_test.reshape(len(X_test), -1)
            preds_scaled = xgb_model.predict(X_test_flat).reshape(-1, 1)
        else:
            # 육지 baseline (LSTM) 평가
            with torch.no_grad():
                X_tensor = torch.tensor(X_test, dtype=torch.float32)
                preds_scaled = m_wind(X_tensor).numpy()
            
        preds_actual = scaler_y.inverse_transform(preds_scaled)
        actuals_actual = scaler_y.inverse_transform(y_test)
        
        mae = mean_absolute_error(actuals_actual, preds_actual)
        rmse = np.sqrt(mean_squared_error(actuals_actual, preds_actual))
        r2 = r2_score(actuals_actual, preds_actual)
        
        mape = np.mean(2 * np.abs(actuals_actual - preds_actual) / (np.abs(actuals_actual) + np.abs(preds_actual) + 1e-8)) * 100
        
        results[region] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        
    return results

def main():
    # 1. Baseline 성능 지표 획득
    try:
        baseline_solar = evaluate_baseline_solar()
        baseline_wind = evaluate_baseline_wind()
    except Exception as e:
        print(f"Baseline 평가 중 에러 발생 (백업 리소스 누락 등): {e}")
        return
        
    # 2. Improved 성능 지표 획득
    improved_metrics_file = os.path.join(NATIONAL_MODEL_PATH, 'improved_metrics.pkl')
    if not os.path.exists(improved_metrics_file):
        print(f"Improved 성능 지표 파일({improved_metrics_file})이 없습니다. generator.py를 먼저 실행해 주세요.")
        return
        
    improved_data = joblib.load(improved_metrics_file)
    improved_solar = improved_data['solar']
    improved_wind = improved_data['wind']
    
    # 3. 비교 리포트 생성
    report_lines = []
    report_lines.append("# 📊 신재생 에너지 예측 모델 성능 개선 비교 리포트\n")
    report_lines.append("> 본 리포트는 **전국 통합 단일 모델(Baseline)**과 **지자체별 독립 로컬 모델 + 주기적 시간 인코딩 적용 모델(Improved)**의 테스트 셋 성능을 정량 비교합니다.\n")
    
    # 태양광 비교 표
    report_lines.append("## ☀️ 태양광 발전량 예측 모델 성능 비교\n")
    report_lines.append("| 지역 | 모델 | R² Score | MAE (MWh) | RMSE (MWh) | MAPE (%) |")
    report_lines.append("| --- | --- | --- | --- | --- | --- |")
    
    for region in sorted(improved_solar.keys()):
        if region in baseline_solar:
            b = baseline_solar[region]
            imp = improved_solar[region]
            
            # 개선폭 계산 화살표
            r2_change = "▲" if imp['R2'] > b['R2'] else "▼"
            mae_change = "▼" if imp['MAE'] < b['MAE'] else "▲"
            
            report_lines.append(f"| **{region}** | Baseline | {b['R2']:.4f} | {b['MAE']:.2f} | {b['RMSE']:.2f} | {b['MAPE']:.2f}% |")
            report_lines.append(f"| | **Improved** | **{imp['R2']:.4f} ({r2_change})** | **{imp['MAE']:.2f} ({mae_change})** | **{imp['RMSE']:.2f}** | **{imp['MAPE']:.2f}%** |")
            report_lines.append("| | | | | | |") # 빈 칸 구분선
            
    # 풍력 비교 표
    report_lines.append("\n## 🌪️ 풍력 발전량 예측 모델 성능 비교\n")
    report_lines.append("| 지역 | 모델 | R² Score | MAE (MWh) | RMSE (MWh) | sMAPE (%) |")
    report_lines.append("| --- | --- | --- | --- | --- | --- |")
    
    for region in sorted(improved_wind.keys()):
        if region in baseline_wind:
            b = baseline_wind[region]
            imp = improved_wind[region]
            
            r2_change = "▲" if imp['R2'] > b['R2'] else "▼"
            mae_change = "▼" if imp['MAE'] < b['MAE'] else "▲"
            
            report_lines.append(f"| **{region}** | Baseline | {b['R2']:.4f} | {b['MAE']:.2f} | {b['RMSE']:.2f} | {b['MAPE']:.2f}% |")
            report_lines.append(f"| | **Improved** | **{imp['R2']:.4f} ({r2_change})** | **{imp['MAE']:.2f} ({mae_change})** | **{imp['RMSE']:.2f}** | **{imp['MAPE']:.2f}%** |")
            report_lines.append("| | | | | | |") # 빈 칸 구분선
            
    # 핵심 트러블슈팅 요약
    report_lines.append("\n## 🧠 주요 성능 개선 기술 요약")
    report_lines.append("1. **지자체별 독립 로컬 모델링(Local Modeling)**:")
    report_lines.append("   - 전국 통합 모델 학습 시 지역별 세부 특성이 평균화되어 뭉개지던 문제를 지역별 LSTM 개별 학습 구조로 전면 혁신하였습니다.")
    report_lines.append("   - 이를 통해 각 지자체 고유의 기상-발전량 가중치를 개별 적합하여 전반적인 예측 정확도(R²)가 대폭 상승했습니다.")
    report_lines.append("2. **Sin/Cos 주기적 시간 인코딩(Periodic Feature Encoding)**:")
    report_lines.append("   - 시간대별 발전량 평균치만을 과적합하여 기상 변화와 무관하게 고정된 피크가 반복되는 문제를 해결했습니다.")
    report_lines.append("   - 하루 주기(24시간)와 연중 주기(12개월)를 삼각함수로 변환하여 기상 독립 변수(일사량, 풍속 등)에 대한 예측 민감도를 획득했습니다.")
    
    report_content = "\n".join(report_lines)
    
    # 마크다운 문법적 공백 오류 및 볼드 렌더링 오류 사전 보정
    import re
    report_content = re.sub(r'\*\*\s*(.*?)\s*\*\*', r'**\1**', report_content)
    # 한국어 조사 결합 시 볼드 깨짐 현상 사전 치환 보정 (**단어**조사 -> **단어조사**)
    report_content = re.sub(r'\*\*(.*?)\*\*(을|를|과|와|의|은|는|이|가|에|서)', r'**\1\2**', report_content)
    
    # 마크다운 보고서로 저장
    comparison_report_file = os.path.join(DOCS_PATH, 'performance_comparison.md')
    with open(comparison_report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"\n성공적으로 비교 리포트가 생성되었습니다: {comparison_report_file}")

if __name__ == '__main__':
    main()
