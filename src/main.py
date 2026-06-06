import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import datetime

# 런타임 모듈 검색 경로 가드 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ui.anomaly_detection import render_anomaly_tab
from ui.seq2seq_prediction import render_seq2seq_tab
from ui.report_generator import render_report_tab


st.set_page_config(page_title="재생에너지 미래 발전량 시뮬레이터", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    /* 기본 주버튼 스타일링 */
    div.stButton > button[kind="primary"] { background-color: #FFD700; color: #000000; border: none; font-weight: bold; }
    div.stButton > button[kind="primary"]:hover { background-color: #FFC107; color: #000000; }
    
    /* 파일철 구분 탭 커스텀 CSS 스타일링 */
    div.stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
        border-bottom: 2px solid #000000 !important;
        padding-bottom: 0px !important;
    }
    
    div.stTabs [data-baseweb="tab"] {
        border: 2px solid #000000 !important;
        border-bottom: none !important;
        border-radius: 12px 12px 0px 0px !important;
        background-color: #f8f9fa !important;
        color: #333333 !important;
        padding: 10px 22px !important;
        font-weight: bold !important;
        margin-bottom: -2px !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: none !important;
    }
    
    div.stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef !important;
        color: #000000 !important;
    }
    
    div.stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1f77b4 !important;
        border: 2px solid #000000 !important;
        border-bottom: 3px solid #ffffff !important; /* 아래 보더를 흰색으로 채워 파일철 속지와 일치감 부여 */
        padding-top: 12px !important; /* 활성화 탭을 살짝 위로 솟아오르게 처리 */
    }
    
    /* 탭 내부 텍스트 스타일 */
    div.stTabs [data-baseweb="tab"] p {
        font-size: 14px !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

KOR_COORDS = {
    '서울시': [37.5665, 126.9780], '부산시': [35.1796, 129.0756], '대구시': [35.8714, 128.6014], '인천시': [37.4563, 126.7052],
    '광주시': [35.1595, 126.8526], '대전시': [36.3504, 127.3845], '울산시': [35.5384, 129.3114], '세종시': [36.4801, 127.2890],
    '경기도': [37.2636, 127.0286], '강원도': [37.8228, 128.1555], '충청북도': [36.6358, 127.4913], '충청남도': [36.6583, 126.6736],
    '전라북도': [35.8203, 127.1088], '전라남도': [34.8161, 126.4629], '경상북도': [36.5760, 128.5056], '경상남도': [35.2383, 128.6925],
    '제주도': [33.4890, 126.4983], '육지': [36.5, 127.5]
}

GPS_FARM_INFO = {
    '한경1': {'lat': 33.3371, 'lon': 126.1630, 'station': '고산(185)', 'mae': 0.5751, 'r2': 0.5086, 'rmse': 0.8370},
    '한경2': {'lat': 33.3369, 'lon': 126.1675, 'station': '고산(185)', 'mae': 1.4418, 'r2': 0.5594, 'rmse': 2.0926},
    '성산1': {'lat': 33.4426, 'lon': 126.8339, 'station': '성산(188)', 'mae': 1.3747, 'r2': 0.4692, 'rmse': 1.9121},
    '성산2': {'lat': 33.3969, 'lon': 126.8222, 'station': '성산(188)', 'mae': 1.0427, 'r2': 0.4777, 'rmse': 1.4194}
}


# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'metadata')
NATIONAL_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'national')
MICRO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'micro_gps')
DOCS_PATH = os.path.join(PROJECT_ROOT, 'docs')

device = torch.device("cpu")

features_solar = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간_sin', '시간_cos', '월_sin', '월_cos', '일사(MJ/m2)']
features_wind  = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간_sin', '시간_cos', '월_sin', '월_cos']

from engine.stochastic_weather import generate_stochastic_weather

@st.cache_resource
def load_scalers():
    sx_solar = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_solar.pkl'))
    sy_solar = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_solar.pkl'))
    sx_wind = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind.pkl'))
    sy_wind = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind.pkl'))
    
    # 제주도 풍력 스케일러 병합 (존재하는 경우)
    if os.path.exists(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind_jeju.pkl')):
        sx_wind.update(joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind_jeju.pkl')))
    if os.path.exists(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind_jeju.pkl')):
        sy_wind.update(joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind_jeju.pkl')))
        
    # 기존 XGBoost 제주도 가중치 파일 호환성 유지
    if os.path.exists(os.path.join(NATIONAL_MODEL_PATH, 'best_model_wind_jeju_xgb.pkl')):
        m_wind_jeju = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'best_model_wind_jeju_xgb.pkl'))
    else:
        m_wind_jeju = None
        
    # GPS 초국소 단지별 스케일러 분리 로드
    if os.path.exists(os.path.join(MICRO_MODEL_PATH, 'scalers_X_wind.pkl')):
        sx_wind_gps = joblib.load(os.path.join(MICRO_MODEL_PATH, 'scalers_X_wind.pkl'))
        sy_wind_gps = joblib.load(os.path.join(MICRO_MODEL_PATH, 'scalers_y_wind.pkl'))
    else:
        sx_wind_gps, sy_wind_gps = {}, {}
        
    return sx_solar, sy_solar, sx_wind, sy_wind, m_wind_jeju, sx_wind_gps, sy_wind_gps

@st.cache_resource
def load_solar_model(region):
    model_path = os.path.join(NATIONAL_MODEL_PATH, f'best_model_solar_{region}.pth')
    if not os.path.exists(model_path):
        # 만약 개별 모델이 없다면 기본 모델로 폴백
        model_path = os.path.join(NATIONAL_MODEL_PATH, 'best_model.pth')
        if not os.path.exists(model_path):
            return None
    model = LSTMModel(len(features_solar), 64, 1, 1, 0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

@st.cache_resource
def load_wind_model(region):
    model_path = os.path.join(NATIONAL_MODEL_PATH, f'best_model_wind_{region}.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(NATIONAL_MODEL_PATH, 'best_model_wind.pth')
        if not os.path.exists(model_path):
            return None
    model = LSTMModel(len(features_wind), 64, 1, 1, 0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

@st.cache_resource
def load_wind_gps_model(stage):
    model_path = os.path.join(MICRO_MODEL_PATH, f'best_model_wind_{stage}.pth')
    if not os.path.exists(model_path):
        return None
    model = LSTMModel(len(features_wind), 64, 1, 1, 0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

@st.cache_data
def load_data():
    s_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'solar_integrated_dataset.csv'), encoding='utf-8-sig')
    w_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
    s_df['일시'] = pd.to_datetime(s_df['일시'])
    w_df['일시'] = pd.to_datetime(w_df['일시'])
    if '풍속(m/s)' in w_df.columns:
        w_df['풍속_세제곱'] = w_df['풍속(m/s)'] ** 3
    return s_df, w_df

sx_solar, sy_solar, sx_wind, sy_wind, m_wind_jeju, sx_wind_gps, sy_wind_gps = load_scalers()
solar_df, wind_df = load_data()

st.title("⚡ AI 기반 신재생 에너지 미래 시뮬레이터")
st.markdown("수집된 과거 데이터를 기반으로 선택하신 날짜의 평균 기상을 자동 추출하여 미래 발전량을 시뮬레이션합니다.")

tab_wind, tab_solar, tab_performance, tab_anomaly, tab_seq2seq, tab_report = st.tabs([
    "🌪️ 풍력 발전량 예측", 
    "☀️ 태양광 발전량 예측", 
    "📊 AI 성능 분석 및 트러블슈팅",
    "⚠️ 발전 이상 감지 및 출력제어",
    "📈 Seq2Seq 24시간 일괄 예측",
    "📋 일일 보고서 생성"
])

# ==========================================
# 탭 1: 풍력 발전량 시뮬레이터
# ==========================================
with tab_wind:
    col_w1, col_w2 = st.columns([1, 2])
    
    with col_w1:
        st.subheader("⚙️ 풍력 시뮬레이션 설정")
        future_date_w = st.date_input("🚀 예측할 미래 날짜", value=datetime.date.today() + datetime.timedelta(days=1), min_value=datetime.date.today(), max_value=datetime.date(2027, 12, 31), key="date_wind")
        sim_region_wind = st.selectbox("🌪️ 예측 지역 (풍력)", list(sx_wind.keys()), key="region_wind")
        
        target_month_w, target_day_w = future_date_w.month, future_date_w.day
        hist_df_w = wind_df[(wind_df['지역'] == sim_region_wind) & (wind_df['일시'].dt.month == target_month_w) & (wind_df['일시'].dt.day == target_day_w)].copy()
        
        agg_features_w = [col for col in features_wind if col not in ['시간', '시간_sin', '시간_cos', '월_sin', '월_cos']]
        if not hist_df_w.empty:
            base_profile_w = hist_df_w.groupby('시간')[agg_features_w].mean().reset_index()
        else:
            base_profile_w = wind_df[wind_df['지역'] == sim_region_wind].groupby('시간')[agg_features_w].mean().reset_index()
        default_wind = float(base_profile_w['풍속(m/s)'].mean())
        default_temp_w = float(base_profile_w['기온(°C)'].mean())

        # AI 확률론적 기상 자동생성 체크박스
        use_stochastic_wind = st.checkbox("🎲 AI 확률론적 기상 자동생성", value=False, key="stochastic_wind")
        
        # GPS 초국소 단지별 합산 모델 적용 체크박스 (전 지역 노출로 확대)
        use_gps_model = st.checkbox("📡 GPS 초국소 단지별 합산 모델 적용 (v2)", value=False, key="gps_model_wind")

        sim_wind_speed = st.slider("💨 예상 평균 풍속 (m/s)", 0.0, 30.0, float(round(default_wind, 1)), 0.1, key=f"w_spd_{sim_region_wind}_{future_date_w}", disabled=use_stochastic_wind)
        sim_temp_w = st.slider("🌡️ 예상 평균 기온 (°C)", -15.0, 40.0, float(round(default_temp_w, 1)), 0.5, key=f"w_tmp_{sim_region_wind}_{future_date_w}", disabled=use_stochastic_wind)
        
        season_w = "겨울"
        if target_month_w in [3, 4]: season_w = "봄"
        elif 5 <= target_month_w <= 8: season_w = "여름"
        elif target_month_w in [9, 10]: season_w = "가을"
            
        if use_stochastic_wind:
            st.caption(f"🎲 **[확률론적 기상 자동생성]** {sim_region_wind} 지역의 {target_month_w}월 기상 통계 분포와 마르코프 체인 AR(1) 노이즈를 결합한 24시간 가상 시퀀스가 실시간 합성됩니다.")
        else:
            st.caption(f"💡 **[{season_w}철 참고]** 지난 6년간 **{target_month_w}월 {target_day_w}일** 평균 풍속은 **{default_wind:.1f} m/s**, 기온은 **{default_temp_w:.1f} °C** 였습니다.")
        btn_sim_wind = st.button("풍력 시뮬레이션 실행", type="primary", use_container_width=True, key="btn_wind")

    with col_w2:
        if btn_sim_wind:
            with st.spinner(f"AI가 {sim_region_wind} 전용 모델을 사용하여 시뮬레이션 중입니다..."):
                if use_stochastic_wind:
                    sim_df_w = generate_stochastic_weather('wind', sim_region_wind, target_month_w)
                else:
                    sim_df_w = base_profile_w.copy()
                    if default_wind > 0: sim_df_w['풍속(m/s)'] = sim_df_w['풍속(m/s)'] * (sim_wind_speed / default_wind)
                    else: sim_df_w['풍속(m/s)'] = sim_wind_speed
                    sim_df_w['기온(°C)'] = sim_df_w['기온(°C)'] + (sim_temp_w - default_temp_w)
                    
                # 강제 정렬 및 인덱스 초기화로 시간대 매핑 붕괴 방지
                sim_df_w = sim_df_w.sort_values('시간').reset_index(drop=True)
                    
                # 롤링 예측을 위한 48시간 기상 확장 생성
                sim_df_extended = pd.concat([sim_df_w, sim_df_w], ignore_index=True)
                sim_df_extended['시간'] = np.arange(48) % 24
                
                # 피처 추가 생성 및 주기적 시간 변수 재동기화 (extended 데이터 기준)
                sim_df_extended['풍속_세제곱'] = sim_df_extended['풍속(m/s)'] ** 3
                sim_df_extended['시간_sin'] = np.sin(2 * np.pi * sim_df_extended['시간'] / 24.0)
                sim_df_extended['시간_cos'] = np.cos(2 * np.pi * sim_df_extended['시간'] / 24.0)
                sim_df_extended['월_sin'] = np.sin(2 * np.pi * (target_month_w - 1) / 12.0)
                sim_df_extended['월_cos'] = np.cos(2 * np.pi * (target_month_w - 1) / 12.0)
                
                hourly_preds_w = []
                
                if use_gps_model:
                    preds_gps = {}
                    pred_sum_w = 0.0
                    
                    # 4개 단지별 24시간 롤링 발전량 시퀀스를 각각 계산
                    stage_hourly_series = {stage: [] for stage in ['한경1', '한경2', '성산1', '성산2']}
                    for stage in ['한경1', '한경2', '성산1', '성산2']:
                        m_wind_gps = load_wind_gps_model(stage)
                        if m_wind_gps is not None and stage in sx_wind_gps and stage in sy_wind_gps:
                            for t in range(24):
                                window_df = sim_df_extended.iloc[t : t+24].copy()
                                scaled_window = sx_wind_gps[stage].transform(window_df[features_wind])
                                input_tensor_stage = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    pred_scaled_stage = m_wind_gps(input_tensor_stage).cpu().numpy()
                                pred_actual_stage = sy_wind_gps[stage].inverse_transform(pred_scaled_stage)
                                pred_val = float(np.maximum(pred_actual_stage[0][0], 0))
                                stage_hourly_series[stage].append(pred_val)
                            
                            stage_sum = sum(stage_hourly_series[stage])
                            preds_gps[stage] = stage_sum
                            pred_sum_w += stage_sum
                        else:
                            preds_gps[stage] = 0.0
                            stage_hourly_series[stage] = [0.0] * 24
                            
                    sim_result_w = pred_sum_w
                    
                    # 4개 단지의 시간대별 발전량 합산 프로파일 생성
                    hourly_preds_w = [
                        sum(stage_hourly_series[stg][t] for stg in ['한경1', '한경2', '성산1', '성산2'])
                        for t in range(24)
                    ]
                else:
                    m_wind = load_wind_model(sim_region_wind)
                    if m_wind is not None:
                        for t in range(24):
                            window_df = sim_df_extended.iloc[t : t+24].copy()
                            scaled_window = sx_wind[sim_region_wind].transform(window_df[features_wind])
                            input_tensor_w = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
                            with torch.no_grad():
                                pred_scaled_w = m_wind(input_tensor_w).cpu().numpy()
                            pred_actual_w = sy_wind[sim_region_wind].inverse_transform(pred_scaled_w)
                            pred_val = float(np.maximum(pred_actual_w[0][0], 0))
                            hourly_preds_w.append(pred_val)
                        
                        # [FIX] 풍력 사후 물리 보정: 풍속³ 법칙 역전 구간 감지 및 교체
                        # 풍속이 25% 이상 하락했는데 발전량이 10% 이상 증가한 경우는 물리 불가
                        for h in range(1, 24):
                            v_curr = float(sim_df_extended.iloc[h]['풍속(m/s)'])
                            v_prev = float(sim_df_extended.iloc[h - 1]['풍속(m/s)'])
                            p_curr = hourly_preds_w[h]
                            p_prev = hourly_preds_w[h - 1]
                            if v_prev > 0.5 and v_curr < v_prev * 0.75 and p_curr > p_prev * 1.10:
                                # 풍속³ 비율로 발전량 보정
                                phys_ratio = (v_curr ** 3) / (v_prev ** 3 + 1e-6)
                                hourly_preds_w[h] = max(p_prev * phys_ratio, 0.0)
                        sim_result_w = sum(hourly_preds_w)
                    elif sim_region_wind == '제주도' and m_wind_jeju is not None:
                        # XGBoost 폴백 (기존 24시간 통짜 데이터 기반 단일 추론 호환성 유지)
                        scaled_X_w = sx_wind[sim_region_wind].transform(sim_df_w[features_wind])
                        input_flat = scaled_X_w.flatten().reshape(1, -1)
                        pred_scaled_w = m_wind_jeju.predict(input_flat).reshape(-1, 1)
                        pred_actual_w = sy_wind[sim_region_wind].inverse_transform(pred_scaled_w)
                        sim_result_w = float(np.maximum(pred_actual_w[0][0], 0))
                        hourly_preds_w = [sim_result_w / 24.0] * 24
                    else:
                        st.error("풍력 모델을 찾을 수 없습니다. generator.py 및 train_jeju.py 학습 상태를 확인하세요.")
                        sim_result_w = 0.0
                        hourly_preds_w = [0.0] * 24
                
                st.success(f"🗓️ {future_date_w.strftime('%Y년 %m월 %d일')} {sim_region_wind} 풍력 예측 완료!")
                
                # 램핑률 계산
                ramping_rates_w = [abs(hourly_preds_w[i] - hourly_preds_w[i-1]) for i in range(1, len(hourly_preds_w))]
                max_ramping_w = max(ramping_rates_w) if ramping_rates_w else 0.0
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric(label=f"🌪️ 예상 일일 총 풍력 발전량", value=f"{sim_result_w:.2f} MWh")
                
                max_wind = float(sim_df_w['풍속(m/s)'].max()) if '풍속(m/s)' in sim_df_w.columns else 0.0
                if max_wind >= 25.0:
                    m_col2.error("❌ 경고: 태풍급 강풍 발생! 물리적 터빈 보호 강제 차단(Cut-out) 확정적 위협")
                elif 15.0 <= max_wind < 25.0:
                    m_col2.warning("⚠️ 주의: 강풍 발생! 풍력 터빈 일부 자동 차단 및 계통 요동 우려")
                else:
                    m_col2.success(f"✅ 안정: 특이 {sim_region_wind} 풍력 계통 불안정 징후 없음")
                
                # 3단계 램핑 알림 연동
                if max_ramping_w < 30.0:
                    m_col3.success(f"✅ 램핑 안정 ({max_ramping_w:.2f} MWh/hr)\n계통 예비력 안정 범위")
                elif 30.0 <= max_ramping_w <= 80.0:
                    m_col3.warning(f"⚠️ 램핑 주의 ({max_ramping_w:.2f} MWh/hr)\n가스터빈/양수 발전기 예비 시동 권고")
                else: # > 80.0
                    m_col3.error(f"🚨 램핑 위험 ({max_ramping_w:.2f} MWh/hr)\n양수발전기 즉시 기동 및 출력제어 준비 필수")
                
                st.divider()
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    if use_gps_model:
                        st.markdown("##### 📡 GPS 초국소 단지별 매핑 위치")
                        map_data = []
                        for stage, coords in GPS_FARM_INFO.items():
                            map_data.append({
                                'lat': coords['lat'], 
                                'lon': coords['lon'], 
                                'name': f"{stage} ({preds_gps[stage]:.2f} MWh)"
                            })
                        map_df = pd.DataFrame(map_data)
                        st.map(map_df, zoom=9)
                    else:
                        st.markdown("##### 🗺️ 타겟 지역")
                        map_data = []
                        if sim_region_wind in KOR_COORDS: map_data.append({'lat': KOR_COORDS[sim_region_wind][0], 'lon': KOR_COORDS[sim_region_wind][1]})
                        map_df = pd.DataFrame(map_data)
                        if not map_df.empty: st.map(map_df, zoom=6)
                with v_col2:
                    if use_gps_model:
                        st.markdown("##### 📊 단지별 발전 기여도 비교")
                        import plotly.graph_objects as go
                        stages = list(preds_gps.keys())
                        vals = list(preds_gps.values())
                        colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
                        
                        fig = go.Figure(go.Bar(
                            x=vals,
                            y=stages,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{v:.2f} MWh" for v in vals],
                            textposition='auto',
                        ))
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=10, b=10),
                            height=250,
                            xaxis_title="예상 발전량 (MWh)",
                            yaxis=dict(autorange="reversed")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("##### 📋 단지별 GPS 초국소 LSTM 모델 성능 (R² / MAE)")
                        cols_spec = st.columns(4)
                        for idx, stage in enumerate(['한경1', '한경2', '성산1', '성산2']):
                            c_info = GPS_FARM_INFO[stage]
                            with cols_spec[idx]:
                                st.metric(
                                    label=f"📍 {stage}", 
                                    value=f"{c_info['r2']:.2f}", 
                                    delta=f"MAE {c_info['mae']:.2f}", 
                                    delta_color="off"
                                )
                    else:
                        st.markdown("##### 📈 24시간 예상 풍력 발전량 및 풍속 추이")
                        import matplotlib.pyplot as plt
                        import matplotlib.font_manager as fm
                        local_font = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fonts', 'malgun.ttf'))
                        sys_font = r'C:\Windows\Fonts\malgun.ttf'
                        font_name = None
                        font_prop = None
                        if os.path.exists(local_font):
                            try:
                                font_name = fm.FontProperties(fname=local_font).get_name()
                                fm.fontManager.addfont(local_font)
                                font_prop = fm.FontProperties(fname=local_font)
                            except Exception: pass
                        if font_prop is None and os.path.exists(sys_font):
                            try:
                                font_name = fm.FontProperties(fname=sys_font).get_name()
                                fm.fontManager.addfont(sys_font)
                                font_prop = fm.FontProperties(fname=sys_font)
                            except Exception: pass
                        
                        if font_name:
                            plt.rcParams['font.family'] = font_name
                        else:
                            plt.rcParams['font.family'] = 'sans-serif'
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        fig, ax1 = plt.subplots(figsize=(6, 3.5))
                        line1 = ax1.plot(np.arange(24), hourly_preds_w, color='#1f77b4', linewidth=2.0, marker='o', markersize=3, label='예상 발전량')
                        ax1.fill_between(np.arange(24), hourly_preds_w, color='#1f77b4', alpha=0.15)
                        ax1.set_xlabel('시간 (시)', fontsize=8)
                        ax1.set_ylabel('발전량 (MWh)', color='#1f77b4', fontsize=8)
                        ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=8)
                        ax1.set_xticks(np.arange(0, 24, 3))
                        
                        ax2 = ax1.twinx()
                        wind_speeds = sim_df_w['풍속(m/s)'].values
                        line2 = ax2.plot(np.arange(24), wind_speeds, color='#2ca02c', linewidth=1.5, linestyle='--', marker='s', markersize=3, label='예측 풍속')
                        ax2.set_ylabel('풍속 (m/s)', color='#2ca02c', fontsize=8)
                        ax2.tick_params(axis='y', labelcolor='#2ca02c', labelsize=8)
                        
                        lines = line1 + line2
                        labels = [l.get_label() for l in lines]
                        if font_prop:
                            ax1.legend(lines, labels, fontsize=8, loc='upper left', prop=font_prop)
                            ax1.set_title("24시간 예상 발전량 및 풍속 시계열 흐름", fontsize=10, fontweight='bold', fontproperties=font_prop)
                        else:
                            ax1.legend(lines, labels, fontsize=8, loc='upper left')
                            ax1.set_title("24시간 예상 발전량 및 풍속 시계열 흐름", fontsize=10, fontweight='bold')
                        ax1.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig)

# ==========================================
# 탭 2: 태양광 발전량 시뮬레이터
# ==========================================
with tab_solar:
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.subheader("⚙️ 태양광 시뮬레이션 설정")
        future_date_s = st.date_input("🚀 예측할 미래 날짜", value=datetime.date.today() + datetime.timedelta(days=1), min_value=datetime.date.today(), max_value=datetime.date(2027, 12, 31), key="date_solar")
        sim_region_solar = st.selectbox("☀️ 예측 지역 (태양광)", list(sx_solar.keys()), key="region_solar")
        
        target_month_s, target_day_s = future_date_s.month, future_date_s.day
        hist_df_s = solar_df[(solar_df['지역'] == sim_region_solar) & (solar_df['일시'].dt.month == target_month_s) & (solar_df['일시'].dt.day == target_day_s)].copy()
        
        agg_features_s = [col for col in features_solar if col not in ['시간', '시간_sin', '시간_cos', '월_sin', '월_cos']]
        if not hist_df_s.empty:
            base_profile_s = hist_df_s.groupby('시간')[agg_features_s].mean().reset_index()
        else:
            base_profile_s = solar_df[solar_df['지역'] == sim_region_solar].groupby('시간')[agg_features_s].mean().reset_index()
        default_insol = float(base_profile_s['일사(MJ/m2)'].mean())
        default_temp_s = float(base_profile_s['기온(°C)'].mean())

        # AI 확률론적 기상 자동생성 체크박스
        use_stochastic_solar = st.checkbox("🎲 AI 확률론적 기상 자동생성", value=False, key="stochastic_solar")

        sim_insol = st.slider("☀️ 예상 평균 일사량 (MJ/m2)", 0.0, 5.0, float(round(default_insol, 2)), 0.05, key=f"s_ins_{sim_region_solar}_{future_date_s}", disabled=use_stochastic_solar)
        sim_temp_s = st.slider("🌡️ 예상 평균 기온 (°C)", -15.0, 40.0, float(round(default_temp_s, 1)), 0.5, key=f"s_tmp_{sim_region_solar}_{future_date_s}", disabled=use_stochastic_solar)
        
        season_s = "겨울"
        if target_month_s in [3, 4]: season_s = "봄"
        elif 5 <= target_month_s <= 8: season_s = "여름"
        elif target_month_s in [9, 10]: season_s = "가을"
            
        if use_stochastic_solar:
            st.caption(f"🎲 **[확률론적 기상 자동생성]** {sim_region_solar} 지역의 {target_month_s}월 기상 통계 분포와 마르코프 체인 AR(1) 노이즈를 결합한 24시간 가상 시퀀스가 실시간 합성됩니다.")
        else:
            st.caption(f"💡 **[{season_s}철 참고]** 지난 6년간 **{target_month_s}월 {target_day_s}일** 평균 일사량은 **{default_insol:.2f} MJ/m2**, 기온은 **{default_temp_s:.1f} °C** 였습니다.")
        btn_sim_solar = st.button("태양광 시뮬레이션 실행", type="primary", use_container_width=True, key="btn_solar")

    with col_s2:
        if btn_sim_solar:
            with st.spinner('AI가 태양광 시나리오를 시뮬레이션 중입니다...'):
                if use_stochastic_solar:
                    sim_df_s = generate_stochastic_weather('solar', sim_region_solar, target_month_s)
                else:
                    sim_df_s = base_profile_s.copy()
                    # [FIX] 일사량 비율 조정: 분모 0 방지 + 물리 최대치(6 MJ/m2) 클리핑
                    if default_insol > 0.01:
                        ratio_s = sim_insol / default_insol
                        sim_df_s['일사(MJ/m2)'] = (sim_df_s['일사(MJ/m2)'] * ratio_s).clip(lower=0.0, upper=6.0)
                    else:
                        sim_df_s['일사(MJ/m2)'] = sim_insol * (sim_df_s['일사(MJ/m2)'] > 0).astype(float)
                    sim_df_s['기온(°C)'] = sim_df_s['기온(°C)'] + (sim_temp_s - default_temp_s)
                
                # 강제 정렬 및 인덱스 초기화로 시간대 매핑 붕괴 방지
                sim_df_s = sim_df_s.sort_values('시간').reset_index(drop=True)
                # [REMOVED] 스왑 가드 제거: 한국 지상 일사량은 물리적으로 6 MJ/m2를 초과하지 않으므로
                # avg_insol > 10.0 조건은 절대 발동하지 않으며, 오히려 정상 데이터를 파괴할 위험이 있음
                            
                # 롤링 예측을 위한 48시간 기상 확장 생성
                sim_df_extended = pd.concat([sim_df_s, sim_df_s], ignore_index=True)
                sim_df_extended['시간'] = np.arange(48) % 24
                
                # 피처 추가 생성 및 주기적 시간 변수 재동기화 (extended 데이터 기준)
                sim_df_extended['시간_sin'] = np.sin(2 * np.pi * sim_df_extended['시간'] / 24.0)
                sim_df_extended['시간_cos'] = np.cos(2 * np.pi * sim_df_extended['시간'] / 24.0)
                sim_df_extended['월_sin'] = np.sin(2 * np.pi * (target_month_s - 1) / 12.0)
                sim_df_extended['월_cos'] = np.cos(2 * np.pi * (target_month_s - 1) / 12.0)
                
                hourly_preds_s = []
                
                # 지자체별 독립 LSTM 모델 로드
                m_solar = load_solar_model(sim_region_solar)
                if m_solar is not None:
                    for t in range(24):
                        window_df = sim_df_extended.iloc[t : t+24].copy()
                        scaled_window = sx_solar[sim_region_solar].transform(window_df[features_solar])
                        input_tensor_s = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            pred_scaled_s = m_solar(input_tensor_s).cpu().numpy()
                        pred_actual_s = sy_solar[sim_region_solar].inverse_transform(pred_scaled_s)
                        pred_val = float(np.maximum(pred_actual_s[0][0], 0))
                        
                        # [FIX] 물리 가드: 훈련 코드 create_dataset과 동일하게
                        # window = [t, t+23] → 예측 = t+24 시점
                        # t번째 예측값의 물리 검증 기준은 window 끝 시점(t+23)의 기상
                        guard_idx = min(t + 23, len(sim_df_extended) - 1)
                        insol_val = sim_df_extended.iloc[guard_idx]['일사(MJ/m2)']
                        hour_val  = int(sim_df_extended.iloc[guard_idx]['시간'])
                        if insol_val <= 0.01 or hour_val < 6 or hour_val > 19:
                            pred_val = 0.0
                            
                        hourly_preds_s.append(pred_val)
                    
                    # [FIX] 사후 물리 보정: 일사량↑인데 발전량↓인 구간을 완만한 상승으로 교체
                    for h in range(7, 15):  # 오전 상승 구간(07~14시)
                        insol_h   = sim_df_s.iloc[h]['일사(MJ/m2)']
                        insol_hm1 = sim_df_s.iloc[h - 1]['일사(MJ/m2)']
                        pred_h    = hourly_preds_s[h]
                        pred_hm1  = hourly_preds_s[h - 1]
                        # 일사량이 15% 이상 증가했는데 발전량이 5% 이상 감소한 경우 보정
                        if insol_h > insol_hm1 * 1.15 and pred_h < pred_hm1 * 0.95 and pred_hm1 > 0:
                            hourly_preds_s[h] = pred_hm1 * (1.0 + (insol_h - insol_hm1) / (insol_hm1 + 1e-6) * 0.5)
                    
                    sim_result_s = sum(hourly_preds_s)
                else:
                    st.error("태양광 모델을 찾을 수 없습니다.")
                    sim_result_s = 0.0
                    hourly_preds_s = [0.0] * 24
                
                st.success(f"🗓️ {future_date_s.strftime('%Y년 %m월 %d일')} {sim_region_solar} 태양광 예측 완료!")
                
                # 램핑률 계산
                ramping_rates_s = [abs(hourly_preds_s[i] - hourly_preds_s[i-1]) for i in range(1, len(hourly_preds_s))]
                max_ramping_s = max(ramping_rates_s) if ramping_rates_s else 0.0
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric(label=f"☀️ 예상 일일 총 태양광 발전량", value=f"{sim_result_s:.2f} MWh")
                is_noon_dip_s = False
                if not sim_df_s.empty:
                    insol_10_s = float(sim_df_s[sim_df_s['시간'] == 10]['일사(MJ/m2)'].values[0]) if not sim_df_s[sim_df_s['시간'] == 10].empty else 0.0
                    insol_12_s = float(sim_df_s[sim_df_s['시간'] == 12]['일사(MJ/m2)'].values[0]) if not sim_df_s[sim_df_s['시간'] == 12].empty else 0.0
                    insol_14_s = float(sim_df_s[sim_df_s['시간'] == 14]['일사(MJ/m2)'].values[0]) if not sim_df_s[sim_df_s['시간'] == 14].empty else 0.0
                    if insol_12_s < insol_10_s * 0.9 or insol_12_s < insol_14_s * 0.9:
                        is_noon_dip_s = True
                
                peak_energy_s = max(hourly_preds_s) if hourly_preds_s else 0.0
                avg_insol_val = float(sim_df_s['일사(MJ/m2)'].mean()) if '일사(MJ/m2)' in sim_df_s.columns else 0.0
                if peak_energy_s >= 200.0:
                    if is_noon_dip_s:
                        m_col2.warning("⚠️ 주의: 정오 시간대 구름 유입 및 일시적 광량 급감(Dip) 리스크")
                    else:
                        m_col2.warning("⚠️ 주의: 정오 시간대 태양광 쏠림 및 계통 과전압 리스크")
                elif avg_insol_val < 0.4:
                    m_col2.warning("☁️ 주의: 광량 부족에 따른 태양광 기저 전력 급감 우려")
                else:
                    m_col2.success(f"✅ 안정: 특이 {sim_region_solar} 태양광 계통 불안정 징후 없음")
                
                # 3단계 램핑 알림 연동
                if max_ramping_s < 30.0:
                    m_col3.success(f"✅ 램핑 안정 ({max_ramping_s:.2f} MWh/hr)\n계통 예비력 안정 범위")
                elif 30.0 <= max_ramping_s <= 80.0:
                    m_col3.warning(f"⚠️ 램핑 주의 ({max_ramping_s:.2f} MWh/hr)\nESS 방전 및 가스터빈 백업 대기 권고")
                else: # > 80.0
                    m_col3.error(f"🚨 램핑 위험 ({max_ramping_s:.2f} MWh/hr)\nESS 가동/양수발전기 즉시 기동 필수")
                
                st.divider()
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.markdown("##### 🗺️ 타겟 지역")
                    map_data = []
                    if sim_region_solar in KOR_COORDS: map_data.append({'lat': KOR_COORDS[sim_region_solar][0], 'lon': KOR_COORDS[sim_region_solar][1]})
                    map_df = pd.DataFrame(map_data)
                    if not map_df.empty: st.map(map_df, zoom=6)
                with v_col2:
                    st.markdown("##### 📈 24시간 예상 태양광 발전량 및 일사량 추이")
                    import matplotlib.pyplot as plt
                    import matplotlib.font_manager as fm
                    local_font = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fonts', 'malgun.ttf'))
                    sys_font = r'C:\Windows\Fonts\malgun.ttf'
                    font_name = None
                    font_prop = None
                    if os.path.exists(local_font):
                         try:
                             font_name = fm.FontProperties(fname=local_font).get_name()
                             fm.fontManager.addfont(local_font)
                             font_prop = fm.FontProperties(fname=local_font)
                         except Exception: pass
                    if font_prop is None and os.path.exists(sys_font):
                         try:
                             font_name = fm.FontProperties(fname=sys_font).get_name()
                             fm.fontManager.addfont(sys_font)
                             font_prop = fm.FontProperties(fname=sys_font)
                         except Exception: pass
                     
                    if font_name:
                         plt.rcParams['font.family'] = font_name
                    else:
                         plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['axes.unicode_minus'] = False
                     
                    fig, ax1 = plt.subplots(figsize=(6, 3.5))
                    line1 = ax1.plot(np.arange(24), hourly_preds_s, color='#1f77b4', linewidth=2.0, marker='o', markersize=3, label='예상 발전량')
                    ax1.fill_between(np.arange(24), hourly_preds_s, color='#1f77b4', alpha=0.15)
                    ax1.set_xlabel('시간 (시)', fontsize=8)
                    ax1.set_ylabel('발전량 (MWh)', color='#1f77b4', fontsize=8)
                    ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=8)
                    ax1.set_xticks(np.arange(0, 24, 3))
                     
                    ax2 = ax1.twinx()
                    insol_vals = sim_df_s['일사(MJ/m2)'].values
                    line2 = ax2.plot(np.arange(24), insol_vals, color='#d62728', linewidth=1.5, linestyle='--', marker='s', markersize=3, label='예측 일사량')
                    ax2.set_ylabel('일사량 (MJ/m2)', color='#d62728', fontsize=8)
                    ax2.tick_params(axis='y', labelcolor='#d62728', labelsize=8)
                     
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    if font_prop:
                         ax1.legend(lines, labels, fontsize=8, loc='upper left', prop=font_prop)
                         ax1.set_title("24시간 예상 발전량 및 일사량 시계열 흐름", fontsize=10, fontweight='bold', fontproperties=font_prop)
                    else:
                         ax1.legend(lines, labels, fontsize=8, loc='upper left')
                         ax1.set_title("24시간 예상 발전량 및 일사량 시계열 흐름", fontsize=10, fontweight='bold')
                    ax1.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)

# ==========================================
# 탭 3: AI 성능 분석 및 트러블슈팅
# ==========================================
with tab_performance:
    st.subheader("📊 AI 성능 향상도 비교 및 트러블슈팅 내역")
    st.markdown("전국 통합 단일 모델(Baseline) 대비 **지자체별 독립 적합 및 삼각함수 시간 인코딩(Improved)을** 구현하여 개선된 성과를 보여줍니다.")
    
    perf_col1, perf_col2 = st.columns([3, 2])
    
    with perf_col1:
        # 성능 비교 리포트 마크다운 로드
        comparison_md_path = os.path.join(DOCS_PATH, 'performance_comparison.md')
        if os.path.exists(comparison_md_path):
            with open(comparison_md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
                
            # 대시보드 화면 상 마크다운 표현 실시간 보정 (볼드 기호 내외부 공백 및 렌더링 깨짐 보정)
            import re
            md_content = re.sub(r'\*\*\s*(.*?)\s*\*\*', r'**\1**', md_content)
            # 한국어 조사 결합 시 볼드 깨짐 현상 실시간 치환 보정 (**단어**조사 -> **단어조사**)
            md_content = re.sub(r'\*\*(.*?)\*\*(을|를|과|와|의|은|는|이|가|에|서)', r'**\1\2**', md_content)
                
            # 1. 공통 대제목 및 소개글 분리
            title_parts = md_content.split("## ☀️ 태양광 발전량 예측 모델 성능 비교")
            if len(title_parts) == 2:
                common_header = title_parts[0]
                rest_content = "## ☀️ 태양광 발전량 예측 모델 성능 비교" + title_parts[1]
            else:
                common_header = ""
                rest_content = md_content
                
            # 2. 태양광 표 파트와 풍력 표 파트 분할
            parts = rest_content.split("## 🌪️ 풍력 발전량 예측 모델 성능 비교")
            if len(parts) == 2:
                solar_part = parts[0]
                wind_part = "## 🌪️ 풍력 발전량 예측 모델 성능 비교" + parts[1]
                
                # 3. 주요 기술 요약 파트 분리
                wind_subparts = wind_part.split("## 🧠 주요 성능 개선 기술 요약")
                if len(wind_subparts) == 2:
                    wind_table_part = wind_subparts[0]
                    summary_part = "## 🧠 주요 성능 개선 기술 요약" + wind_subparts[1]
                else:
                    wind_table_part = wind_part
                    summary_part = ""
                    
                # 대제목 및 소개글 렌더링
                if common_header:
                    st.markdown(common_header, unsafe_allow_html=True)
                    
                # 서브 컬럼 가로 배치
                sub_col1, sub_col2 = st.columns(2)
                
                # 동일한 세로 높이(600px)의 스크롤 컨테이너로 두 표의 높이 완벽 정렬
                with sub_col1:
                    st.markdown("### ☀️ 태양광 발전량 예측 모델 성능 비교")
                    with st.container(height=600):
                        st.markdown(solar_part.replace("## ☀️ 태양광 발전량 예측 모델 성능 비교", ""), unsafe_allow_html=True)
                        
                with sub_col2:
                    st.markdown("### 🌪️ 풍력 발전량 예측 모델 성능 비교")
                    with st.container(height=600):
                        st.markdown(wind_table_part.replace("## 🌪️ 풍력 발전량 예측 모델 성능 비교", ""), unsafe_allow_html=True)
                        
                if summary_part:
                    st.markdown(summary_part, unsafe_allow_html=True)
            else:
                st.markdown(md_content)
        else:
            st.info("💡 `compare_results.py`를 실행하면 정량 비교 메트릭 테이블이 여기에 실시간 표시됩니다.")
            
    with perf_col2:
        st.markdown("### 🧠 기술적 학술 설명")
        st.info("""
        * **실제 측정값**: 
          - 테스트 셋(전체 수집 기상 데이터의 최근 20% 기간)에 해당하는 **한국전력거래소(KPX)의 시간별 실제 발전 실측 데이터**입니다.
        * **예측 발전량**:
          - 특정 시간 시점의 **직전 24시간 연속 기상 데이터**(기온, 풍속, 습도, 일사량, 미세먼지 등)를 시퀀스로 묶어 LSTM에 공급하고, 모델 내부의 순환 계층 가중치를 연산하여 도출한 **향후 1시간 후의 발전량 예측치**입니다.
        """)
        
        st.markdown("### 🪵 학습 터미널 로그")
        train_log_path = os.path.join(DOCS_PATH, 'train_log.txt')
        if os.path.exists(train_log_path):
            with open(train_log_path, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            # 최신 50개 라인만 렌더링
            st.code("".join(log_lines[-60:]), language='bash')
        else:
            st.warning("학습 로그 파일(train_log.txt)이 존재하지 않습니다.")
            
    st.divider()
    st.subheader("📈 지역별 모델의 고유 오차 분석 차트 (과적합 회피 결과)")
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        solar_chart_path = os.path.join(DOCS_PATH, 'lstm_result_solar_all.png')
        if os.path.exists(solar_chart_path):
            st.image(solar_chart_path, caption="☀️ 태양광: 지자체별 독립 손실(Loss) 및 실측vs예측 시계열 비교 (x축 날짜 매핑)", use_container_width=True)
    with chart_col2:
        wind_chart_path = os.path.join(DOCS_PATH, 'lstm_result_wind_all.png')
        if os.path.exists(wind_chart_path):
            st.image(wind_chart_path, caption="🌪️ 풍력: 지자체별 독립 손실(Loss) 및 실측vs예측 시계열 비교 (x축 날짜 매핑)", use_container_width=True)

# ==========================================
# 탭 4: 발전 이상 감지 및 출력제어 리스크
# ==========================================
with tab_anomaly:
    render_anomaly_tab(wind_df, solar_df)

# ==========================================
# 탭 5: Seq2Seq 24시간 일괄 예측
# ==========================================
with tab_seq2seq:
    render_seq2seq_tab(wind_df, solar_df)

# ==========================================
# 탭 6: 일일 보고서 생성 (공공기관 서식 준수)
# ==========================================
with tab_report:
    render_report_tab(wind_df, solar_df)