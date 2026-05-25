import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import datetime

st.set_page_config(page_title="재생에너지 미래 발전량 시뮬레이터", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    div.stButton > button[kind="primary"] { background-color: #FFD700; color: #000000; border: none; font-weight: bold; }
    div.stButton > button[kind="primary"]:hover { background-color: #FFC107; color: #000000; }
    </style>
""", unsafe_allow_html=True)

KOR_COORDS = {
    '서울시': [37.5665, 126.9780], '부산시': [35.1796, 129.0756], '대구시': [35.8714, 128.6014], '인천시': [37.4563, 126.7052],
    '광주시': [35.1595, 126.8526], '대전시': [36.3504, 127.3845], '울산시': [35.5384, 129.3114], '세종시': [36.4801, 127.2890],
    '경기도': [37.2636, 127.0286], '강원도': [37.8228, 128.1555], '충청북도': [36.6358, 127.4913], '충청남도': [36.6583, 126.6736],
    '전라북도': [35.8203, 127.1088], '전라남도': [34.8161, 126.4629], '경상북도': [36.5760, 128.5056], '경상남도': [35.2383, 128.6925],
    '제주도': [33.4890, 126.4983], '육지': [36.5, 127.5]
}

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
device = torch.device("cpu")

features_solar = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간', '월', '일사(MJ/m2)']
features_wind  = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간', '월']

@st.cache_resource
def load_models_and_scalers():
    # 1. 태양광 모델 로드
    m_solar = LSTMModel(len(features_solar), 128, 2, 1, 0.3).to(device)
    m_solar.load_state_dict(torch.load(os.path.join(DATASET_PATH, 'best_model.pth'), map_location=device))
    m_solar.eval()
    sx_solar = joblib.load(os.path.join(DATASET_PATH, 'scalers_X_solar.pkl'))
    sy_solar = joblib.load(os.path.join(DATASET_PATH, 'scalers_y_solar.pkl'))
    
    # 2. 육지 풍력 모델(LSTM) 로드
    m_wind = LSTMModel(len(features_wind), 128, 2, 1, 0.3).to(device)
    m_wind.load_state_dict(torch.load(os.path.join(DATASET_PATH, 'best_model_wind.pth'), map_location=device))
    m_wind.eval()
    sx_wind = joblib.load(os.path.join(DATASET_PATH, 'scalers_X_wind.pkl'))
    sy_wind = joblib.load(os.path.join(DATASET_PATH, 'scalers_y_wind.pkl'))
    
    # 3. 제주도 풍력 모델(XGBoost) 로드 및 딕셔너리 병합
    if os.path.exists(os.path.join(DATASET_PATH, 'best_model_wind_jeju_xgb.pkl')):
        m_wind_jeju = joblib.load(os.path.join(DATASET_PATH, 'best_model_wind_jeju_xgb.pkl'))
        sx_wind.update(joblib.load(os.path.join(DATASET_PATH, 'scalers_X_wind_jeju.pkl')))
        sy_wind.update(joblib.load(os.path.join(DATASET_PATH, 'scalers_y_wind_jeju.pkl')))
    else:
        m_wind_jeju = None

    return m_solar, sx_solar, sy_solar, m_wind, m_wind_jeju, sx_wind, sy_wind

@st.cache_data
def load_data():
    s_df = pd.read_csv(os.path.join(DATASET_PATH, 'solar_integrated_dataset.csv'), encoding='utf-8-sig')
    w_df = pd.read_csv(os.path.join(DATASET_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
    s_df['일시'] = pd.to_datetime(s_df['일시'])
    w_df['일시'] = pd.to_datetime(w_df['일시'])
    if '풍속(m/s)' in w_df.columns:
        w_df['풍속_세제곱'] = w_df['풍속(m/s)'] ** 3
    return s_df, w_df

m_solar, sx_solar, sy_solar, m_wind, m_wind_jeju, sx_wind, sy_wind = load_models_and_scalers()
solar_df, wind_df = load_data()

st.title("⚡ AI 기반 신재생 에너지 미래 시뮬레이터")
st.markdown("수집된 과거 데이터를 기반으로 선택하신 날짜의 평균 기상을 자동 추출하여 미래 발전량을 시뮬레이션합니다.")

tab_wind, tab_solar = st.tabs(["🌪️ 풍력 발전량 예측", "☀️ 태양광 발전량 예측"])

# ==========================================
# 탭 1: 풍력 발전량 시뮬레이터 (육지 & 제주도 통합)
# ==========================================
with tab_wind:
    col_w1, col_w2 = st.columns([1, 2])
    
    with col_w1:
        st.subheader("⚙️ 풍력 시뮬레이션 설정")
        future_date_w = st.date_input("🚀 예측할 미래 날짜", value=datetime.date.today() + datetime.timedelta(days=1), min_value=datetime.date.today(), max_value=datetime.date(2027, 12, 31), key="date_wind")
        
        # 목록에 '제주도'가 추가되었습니다!
        sim_region_wind = st.selectbox("🌪️ 예측 지역 (풍력)", list(sx_wind.keys()), key="region_wind")
        
        target_month_w, target_day_w = future_date_w.month, future_date_w.day
        hist_df_w = wind_df[(wind_df['지역'] == sim_region_wind) & (wind_df['일시'].dt.month == target_month_w) & (wind_df['일시'].dt.day == target_day_w)].copy()
        
        if not hist_df_w.empty:
            agg_features_w = [col for col in features_wind if col != '시간']
            base_profile_w = hist_df_w.groupby('시간')[agg_features_w].mean().reset_index()
            default_wind = float(base_profile_w['풍속(m/s)'].mean())
            default_temp_w = float(base_profile_w['기온(°C)'].mean())
        else:
            base_profile_w = wind_df[wind_df['지역'] == sim_region_wind].tail(24).copy()
            default_wind, default_temp_w = 5.0, 15.0

        # 지역과 날짜가 바뀔 때마다 슬라이더가 추천값으로 완벽히 리셋되도록 다이내믹 Key 적용
        sim_wind_speed = st.slider("💨 예상 평균 풍속 (m/s)", 0.0, 30.0, float(round(default_wind, 1)), 0.1, key=f"w_spd_{sim_region_wind}_{future_date_w}")
        sim_temp_w = st.slider("🌡️ 예상 평균 기온 (°C)", -15.0, 40.0, float(round(default_temp_w, 1)), 0.5, key=f"w_tmp_{sim_region_wind}_{future_date_w}")
        
        season_w = "겨울"
        if target_month_w in [3, 4]: season_w = "봄"
        elif 5 <= target_month_w <= 8: season_w = "여름"
        elif target_month_w in [9, 10]: season_w = "가을"
            
        st.caption(f"💡 **[{season_w}철 참고]** 지난 6년간 **{target_month_w}월 {target_day_w}일** 평균 풍속은 **{default_wind:.1f} m/s**, 기온은 **{default_temp_w:.1f} °C** 였습니다.")
        btn_sim_wind = st.button("풍력 시뮬레이션 실행", type="primary", use_container_width=True, key="btn_wind")

    with col_w2:
        if btn_sim_wind:
            with st.spinner(f"AI가 {sim_region_wind} 전용 모델을 사용하여 시뮬레이션 중입니다..."):
                sim_df_w = base_profile_w.copy()
                
                if default_wind > 0: sim_df_w['풍속(m/s)'] = sim_df_w['풍속(m/s)'] * (sim_wind_speed / default_wind)
                else: sim_df_w['풍속(m/s)'] = sim_wind_speed
                    
                sim_df_w['풍속_세제곱'] = sim_df_w['풍속(m/s)'] ** 3
                sim_df_w['기온(°C)'] = sim_df_w['기온(°C)'] + (sim_temp_w - default_temp_w)
                
                scaled_X_w = sx_wind[sim_region_wind].transform(sim_df_w[features_wind])
                
                # 🔥 지역별 모델 자동 분기 로직
                if sim_region_wind == '제주도' and m_wind_jeju is not None:
                    # 제주도 모델 (XGBoost)
                    input_flat = scaled_X_w.flatten().reshape(1, -1)
                    pred_scaled_w = m_wind_jeju.predict(input_flat).reshape(-1, 1)
                else:
                    # 육지 모델 (LSTM)
                    input_tensor_w = torch.tensor(scaled_X_w, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred_scaled_w = m_wind(input_tensor_w).cpu().numpy()
                
                pred_actual_w = sy_wind[sim_region_wind].inverse_transform(pred_scaled_w)
                sim_result_w = float(np.maximum(pred_actual_w[0][0], 0))
                
                st.success(f"🗓️ {future_date_w.strftime('%Y년 %m월 %d일')} {sim_region_wind} 풍력 예측 완료!")
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric(label=f"🌪️ 예상 풍력 발전량", value=f"{sim_result_w:.2f} MWh")
                if sim_wind_speed > 15: m_col2.error("⚠️ 강풍 발생! 터빈 제어(Cut-out) 리스크 발생")
                else: m_col2.success("✅ 안정적인 발전이 예상됩니다.")
                
                st.divider()
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.markdown("##### 🗺️ 타겟 지역")
                    map_data = []
                    if sim_region_wind in KOR_COORDS: map_data.append({'lat': KOR_COORDS[sim_region_wind][0], 'lon': KOR_COORDS[sim_region_wind][1]})
                    map_df = pd.DataFrame(map_data)
                    if not map_df.empty: st.map(map_df, zoom=6)
                with v_col2:
                    st.markdown("##### 📈 24시간 예상 풍속 흐름")
                    st.line_chart(sim_df_w.set_index('시간')['풍속(m/s)'], use_container_width=True)

# ==========================================
# 탭 2: 태양광 발전량 시뮬레이터 (동일 유지)
# ==========================================
with tab_solar:
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.subheader("⚙️ 태양광 시뮬레이션 설정")
        future_date_s = st.date_input("🚀 예측할 미래 날짜", value=datetime.date.today() + datetime.timedelta(days=1), min_value=datetime.date.today(), max_value=datetime.date(2027, 12, 31), key="date_solar")
        sim_region_solar = st.selectbox("☀️ 예측 지역 (태양광)", list(sx_solar.keys()), key="region_solar")
        
        target_month_s, target_day_s = future_date_s.month, future_date_s.day
        hist_df_s = solar_df[(solar_df['지역'] == sim_region_solar) & (solar_df['일시'].dt.month == target_month_s) & (solar_df['일시'].dt.day == target_day_s)].copy()
        
        if not hist_df_s.empty:
            agg_features_s = [col for col in features_solar if col != '시간']
            base_profile_s = hist_df_s.groupby('시간')[agg_features_s].mean().reset_index()
            default_insol = float(base_profile_s['일사(MJ/m2)'].mean())
            default_temp_s = float(base_profile_s['기온(°C)'].mean())
        else:
            base_profile_s = solar_df[solar_df['지역'] == sim_region_solar].tail(24).copy()
            default_insol, default_temp_s = 1.0, 15.0

        # 지역과 날짜가 바뀔 때마다 슬라이더가 추천값으로 완벽히 리셋되도록 다이내믹 Key 적용
        sim_insol = st.slider("☀️ 예상 평균 일사량 (MJ/m2)", 0.0, 5.0, float(round(default_insol, 2)), 0.05, key=f"s_ins_{sim_region_solar}_{future_date_s}")
        sim_temp_s = st.slider("🌡️ 예상 평균 기온 (°C)", -15.0, 40.0, float(round(default_temp_s, 1)), 0.5, key=f"s_tmp_{sim_region_solar}_{future_date_s}")
        
        season_s = "겨울"
        if target_month_s in [3, 4]: season_s = "봄"
        elif 5 <= target_month_s <= 8: season_s = "여름"
        elif target_month_s in [9, 10]: season_s = "가을"
            
        st.caption(f"💡 **[{season_s}철 참고]** 지난 6년간 **{target_month_s}월 {target_day_s}일** 평균 일사량은 **{default_insol:.2f} MJ/m2**, 기온은 **{default_temp_s:.1f} °C** 였습니다.")
        btn_sim_solar = st.button("태양광 시뮬레이션 실행", type="primary", use_container_width=True, key="btn_solar")

    with col_s2:
        if btn_sim_solar:
            with st.spinner('AI가 태양광 시나리오를 시뮬레이션 중입니다...'):
                sim_df_s = base_profile_s.copy()
                
                if default_insol > 0: sim_df_s['일사(MJ/m2)'] = sim_df_s['일사(MJ/m2)'] * (sim_insol / default_insol)
                else: sim_df_s['일사(MJ/m2)'] = sim_insol
                    
                sim_df_s['기온(°C)'] = sim_df_s['기온(°C)'] + (sim_temp_s - default_temp_s)
                
                scaled_X_s = sx_solar[sim_region_solar].transform(sim_df_s[features_solar])
                input_tensor_s = torch.tensor(scaled_X_s, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_scaled_s = m_solar(input_tensor_s).cpu().numpy()
                    pred_actual_s = sy_solar[sim_region_solar].inverse_transform(pred_scaled_s)
                sim_result_s = float(np.maximum(pred_actual_s[0][0], 0))
                
                st.success(f"🗓️ {future_date_s.strftime('%Y년 %m월 %d일')} {sim_region_solar} 태양광 예측 완료!")
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric(label=f"☀️ 예상 태양광 발전량", value=f"{sim_result_s:.2f} MWh")
                if sim_insol < 0.5: m_col2.warning("☁️ 흐린 날씨로 인해 발전량 저하가 예상됩니다.")
                else: m_col2.success("✅ 원활한 태양광 발전이 예상됩니다.")
                
                st.divider()
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.markdown("##### 🗺️ 타겟 지역")
                    map_data = []
                    if sim_region_solar in KOR_COORDS: map_data.append({'lat': KOR_COORDS[sim_region_solar][0], 'lon': KOR_COORDS[sim_region_solar][1]})
                    map_df = pd.DataFrame(map_data)
                    if not map_df.empty: st.map(map_df, zoom=6)
                with v_col2:
                    st.markdown("##### 📈 24시간 예상 일사량 흐름")
                    st.line_chart(sim_df_s.set_index('시간')['일사(MJ/m2)'], use_container_width=True)