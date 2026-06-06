import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import joblib
import datetime
import matplotlib.pyplot as plt
import unicodedata
from fpdf import FPDF
from engine.stochastic_weather import generate_stochastic_weather

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
NATIONAL_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'national')
MICRO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'micro_gps')
DOCS_PATH = os.path.join(PROJECT_ROOT, 'docs')

device = torch.device('cpu')

features_solar = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '시간_sin', '시간_cos', '월_sin', '월_cos', '일사(MJ/m2)']
features_wind  = ['기온(°C)', '풍속(m/s)', '풍속_세제곱', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)', '시간_sin', '시간_cos', '월_sin', '월_cos']

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

@st.cache_resource
def load_generator_scalers():
    sx_solar = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_solar.pkl'))
    sy_solar = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_solar.pkl'))
    sx_wind = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind.pkl'))
    sy_wind = joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind.pkl'))
    if os.path.exists(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind_jeju.pkl')):
        sx_wind.update(joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_X_wind_jeju.pkl')))
    if os.path.exists(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind_jeju.pkl')):
        sy_wind.update(joblib.load(os.path.join(NATIONAL_MODEL_PATH, 'scalers_y_wind_jeju.pkl')))
    if os.path.exists(os.path.join(MICRO_MODEL_PATH, 'scalers_X_wind.pkl')):
        sx_wind_gps = joblib.load(os.path.join(MICRO_MODEL_PATH, 'scalers_X_wind.pkl'))
        sy_wind_gps = joblib.load(os.path.join(MICRO_MODEL_PATH, 'scalers_y_wind.pkl'))
    else:
        sx_wind_gps, sy_wind_gps = {}, {}
    return sx_solar, sy_solar, sx_wind, sy_wind, sx_wind_gps, sy_wind_gps

@st.cache_resource
def load_solar_model(region):
    model_path = os.path.join(NATIONAL_MODEL_PATH, f'best_model_solar_{region}.pth')
    if not os.path.exists(model_path):
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

class PublicSectorPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_malgun = False

    def add_font(self, family, style="", fname=None, **kwargs):
        try:
            super().add_font(family, style, fname, **kwargs)
            if family.lower() == 'malgun':
                self.has_malgun = True
        except Exception as e:
            raise e

    def set_font(self, family, style="", size=0):
        if family.lower() == 'malgun' and not self.has_malgun:
            family = 'Helvetica'
        super().set_font(family, style, size)

    def cell(self, w, h=None, txt="", **kwargs):
        if isinstance(txt, str):
            txt = unicodedata.normalize('NFC', txt)
        return super().cell(w, h, txt, **kwargs)

    def multi_cell(self, w, h=None, txt="", **kwargs):
        if isinstance(txt, str):
            txt = unicodedata.normalize('NFC', txt)
        return super().multi_cell(w, h, txt, **kwargs)

    def header(self):
        if self.page_no() > 1:
            self.set_font('Malgun', '', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, '[대외주의] 신재생에너지 일일 계통 발전 예측 보고서', border=0, ln=1, align='L')
            self.line(15, 20, 195, 20)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Malgun', '', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f'- {self.page_no()} -', border=0, ln=0, align='C')

    def add_bullet(self, indent_width, bullet_str, text_str, h=5):
        orig_margin = self.l_margin
        self.set_x(orig_margin + indent_width)
        if not text_str:
            self.cell(0, h, bullet_str, border=0, ln=1)
            return
        b_w = self.get_string_width(bullet_str) + 1.5
        self.cell(b_w, h, bullet_str, border=0, ln=0)
        self.set_left_margin(orig_margin + indent_width + b_w)
        self.multi_cell(0, h, text_str)
        self.set_left_margin(orig_margin)



def render_report_tab(wind_df, solar_df):
    st.subheader('📋 공문서 및 공기업 규격 일일 전력 발전 예측 보고서 자동 작성 시스템')
    st.markdown('본 모듈은 공문서 표준 기안 양식을 차용하여 3페이지 분량에 최적화된 맞춤형 보고서를 컴파일합니다. (단원 구조별 내용 구성 차별화 및 페이지 밀도 극대화)')
    sx_solar, sy_solar, sx_wind, sy_wind, sx_wind_gps, sy_wind_gps = load_generator_scalers()
    col_rep1, col_rep2 = st.columns([1, 2])
    with col_rep1:
        st.info('⚙️ **보고서 작성 설정 조건**')
        target_energy = st.radio('에너지원 선택', ['태양광 (Solar)', '풍력 (Wind)'], key='rep_energy')
        if target_energy == '태양광 (Solar)':
            regions = list(sx_solar.keys())
            default_reg = '경기도'
        else:
            regions = list(sx_wind.keys())
            default_reg = '제주도'
        sel_region = st.selectbox('분석 대상 지자체 지역', regions, index=regions.index(default_reg) if default_reg in regions else 0, key='rep_region')
        target_date = st.date_input(
            '보고서 기준 일자 (예측일)',
            value=datetime.date.today() + datetime.timedelta(days=1),
            min_value=datetime.date.today(),
            max_value=datetime.date(2027, 12, 31),
            key='rep_date'
        )
        use_stochastic = st.checkbox('🎲 AI 확률론적 기상 시나리오 적용', value=True, key='rep_stochastic')
        use_gps_v2 = False
        if target_energy == '풍력 (Wind)':
            use_gps_v2 = st.checkbox('📡 GPS 초국소 단지별 합산 모델 적용 (v2)', value=False, key='rep_gps')
        st.divider()
        st.markdown('##### ✍️ 기안 정보 기입')
        doc_number = st.text_input('문서 번호', '신재생계통-2026-0604호', key='rep_doc_num')
        drafter_dept = st.text_input('기안 부서', '신재생에너지계통통제원', key='rep_dept')
        drafter_name = st.text_input('기안자 직위/성명', '주임연구원 심온', key='rep_name')
        st.markdown('##### 📄 문서 분량 및 내용 구성')
        st.info('📄 **보고서 분량**: 공문서 규격 3페이지 (고정형)')
        btn_build_report = st.button('공문서 규격 보고서 컴파일', type='primary', use_container_width=True)
    with col_rep2:
        if btn_build_report:
            with st.spinner('AI 시뮬레이션 및 행정 문서 서식 템플릿 컴파일 중...'):
                target_month = target_date.month
                target_day = target_date.day
                if target_energy == '태양광 (Solar)':
                    hist_df = solar_df[(solar_df['지역'] == sel_region) & (solar_df['일시'].dt.month == target_month) & (solar_df['일시'].dt.day == target_day)].copy()
                    if not hist_df.empty:
                        agg_feats = [c for c in features_solar if c not in ['시간', '시간_sin', '시간_cos', '월_sin', '월_cos']]
                        base_profile = hist_df.groupby('시간')[agg_feats].mean().reset_index()
                    else:
                        base_profile = solar_df[solar_df['지역'] == sel_region].tail(24).copy()
                else:
                    hist_df = wind_df[(wind_df['지역'] == sel_region) & (wind_df['일시'].dt.month == target_month) & (wind_df['일시'].dt.day == target_day)].copy()
                    if not hist_df.empty:
                        agg_feats = [c for c in features_wind if c not in ['시간', '시간_sin', '시간_cos', '월_sin', '월_cos']]
                        base_profile = hist_df.groupby('시간')[agg_feats].mean().reset_index()
                    else:
                        base_profile = wind_df[wind_df['지역'] == sel_region].tail(24).copy()
                if use_stochastic:
                    sim_weather_df = generate_stochastic_weather('solar' if target_energy == '태양광 (Solar)' else 'wind', sel_region, target_month)
                else:
                    sim_weather_df = base_profile.copy()
                
                # 태양광 기상 데이터 가드 (기온과 일사량 스왑 오류 방어)
                if target_energy == '태양광 (Solar)' and '기온(°C)' in sim_weather_df.columns and '일사(MJ/m2)' in sim_weather_df.columns:
                    day_df = sim_weather_df[(sim_weather_df['시간'] >= 10) & (sim_weather_df['시간'] <= 16)]
                    if not day_df.empty:
                        avg_temp_day = day_df['기온(°C)'].mean()
                        avg_insol_day = day_df['일사(MJ/m2)'].mean()
                        if avg_insol_day > 10.0:
                            temp_vals = sim_weather_df['기온(°C)'].copy()
                            sim_weather_df['기온(°C)'] = sim_weather_df['일사(MJ/m2)'].copy()
                            sim_weather_df['일사(MJ/m2)'] = temp_vals
                            
                sim_df_extended = pd.concat([sim_weather_df, sim_weather_df], ignore_index=True)
                sim_df_extended['시간'] = np.arange(48) % 24
                if target_energy == '태양광 (Solar)':
                    sim_df_extended['시간_sin'] = np.sin(2 * np.pi * sim_df_extended['시간'] / 24.0)
                    sim_df_extended['시간_cos'] = np.cos(2 * np.pi * sim_df_extended['시간'] / 24.0)
                    sim_df_extended['월_sin'] = np.sin(2 * np.pi * (target_month - 1) / 12.0)
                    sim_df_extended['월_cos'] = np.cos(2 * np.pi * (target_month - 1) / 12.0)
                else:
                    sim_df_extended['풍속_세제곱'] = sim_df_extended['풍속(m/s)'] ** 3
                    sim_df_extended['시간_sin'] = np.sin(2 * np.pi * sim_df_extended['시간'] / 24.0)
                    sim_df_extended['시간_cos'] = np.cos(2 * np.pi * sim_df_extended['시간'] / 24.0)
                    sim_df_extended['월_sin'] = np.sin(2 * np.pi * (target_month - 1) / 12.0)
                    sim_df_extended['월_cos'] = np.cos(2 * np.pi * (target_month - 1) / 12.0)
                hourly_preds = []
                if target_energy == '태양광 (Solar)':
                    m_solar = load_solar_model(sel_region)
                    if m_solar is not None:
                        for t in range(24):
                            window = sim_df_extended.iloc[t : t+24].copy()
                            scaled_window = sx_solar[sel_region].transform(window[features_solar])
                            input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
                            with torch.no_grad():
                                pred_scaled = m_solar(input_tensor).cpu().numpy()
                            pred_actual = sy_solar[sel_region].inverse_transform(pred_scaled)
                            pred_val = float(np.maximum(pred_actual[0][0], 0))
                            insol_val = sim_weather_df.iloc[t]['일사(MJ/m2)']
                            hour_val = int(sim_weather_df.iloc[t]['시간'])
                            if insol_val <= 0.01 or hour_val < 6 or hour_val > 19:
                                pred_val = 0.0
                            hourly_preds.append(pred_val)
                else:
                    if use_gps_v2:
                        stage_preds = {stg: [] for stg in ['한경1', '한경2', '성산1', '성산2']}
                        for stg in ['한경1', '한경2', '성산1', '성산2']:
                            m_wind_gps = load_wind_gps_model(stg)
                            if m_wind_gps is not None:
                                for t in range(24):
                                    window = sim_df_extended.iloc[t : t+24].copy()
                                    scaled_window = sx_wind_gps[stg].transform(window[features_wind])
                                    input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        pred_scaled = m_wind_gps(input_tensor).cpu().numpy()
                                    pred_actual = sy_wind_gps[stg].inverse_transform(pred_scaled)
                                    pred_val = float(np.maximum(pred_actual[0][0], 0))
                                    stage_preds[stg].append(pred_val)
                            else:
                                stage_preds[stg] = [0.0] * 24
                        hourly_preds = [
                            sum(stage_preds[stg][t] for stg in ['한경1', '한경2', '성산1', '성산2'])
                            for t in range(24)
                        ]
                    else:
                        m_wind = load_wind_model(sel_region)
                        if m_wind is not None:
                            for t in range(24):
                                window = sim_df_extended.iloc[t : t+24].copy()
                                scaled_window = sx_wind[sel_region].transform(window[features_wind])
                                input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    pred_scaled = m_wind(input_tensor).cpu().numpy()
                                pred_actual = sy_wind[sel_region].inverse_transform(pred_scaled)
                                pred_val = float(np.maximum(pred_actual[0][0], 0))
                                hourly_preds.append(pred_val)
                if not hourly_preds:
                    st.error('⚠️ AI 예측 데이터 산출 실패.')
                    return
                total_energy = sum(hourly_preds)
                peak_energy = max(hourly_preds)
                peak_hour = hourly_preds.index(peak_energy)
                avg_wind_or_insol = float(sim_weather_df['일사(MJ/m2)'].mean()) if target_energy == '태양광 (Solar)' else float(sim_weather_df['풍속(m/s)'].mean())
                avg_temp = float(sim_weather_df['기온(°C)'].mean())
                ramping_rates = [abs(hourly_preds[i] - hourly_preds[i-1]) for i in range(1, 24)]
                max_ramping = max(ramping_rates) if ramping_rates else 0.0
                max_ramping_hour = ramping_rates.index(max_ramping) + 1 if ramping_rates else 0
                weekday_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
                formatted_target_date = f'{target_date.strftime("%Y. %m. %d.")} ({weekday_map[target_date.weekday()]})'
                current_time = datetime.datetime.now().strftime('%Y. %m. %d. %H:%M')
                
                # Ⅰ~Ⅸ단원 동적 텍스트 변수 정의
                if target_energy == '태양광 (Solar)':
                    bg_need_1 = '지구 온난화 및 기후 변화로 인해 태양광 발전의 일사량 간헐성이 전력망 운영 한계치에 다다르고 있음.'
                    bg_need_2 = '특히 기습적인 구름 유입 및 미세먼지로 인한 급격한 발전량 강하(Drop) 현상으로 송배전 전압 안정이 위협받음.'
                    bg_need_3 = '이에 고정밀 AI LSTM 일사량 감응 모델을 활용한 사전 시뮬레이션으로 안정성을 선제 판정할 필요가 있음.'
                    bg_purpose_1 = f'예측 대상 일자({formatted_target_date})의 예상 날씨에 기초한 {sel_region} 발전 기여 전력량을 사전 연산함.'
                    bg_purpose_2 = '예상되는 주간 과전압 리스크를 최소화하고, ESS(에너지저장장치) 충·방전 스케줄링을 최적화하며 계통 전압을 안정 범위 이내로 통제하는 데 기여함을 목적으로 함.'
                    
                    arch_input_dim = '9차원 기상 피처 (기온, 풍속, 습도, 미세먼지농도, 일사량, 시간/월 주기 인자)'
                    
                    vi_opinion_a = '일출(06시)부터 일몰(20시)까지 일사량 패턴에 감응하여 매끄러운 단봉형(Bell-shape) 발전 곡선을 형성함.'
                    vi_opinion_b = '정오 시간대(12~14시)에 출력이 고도로 밀집되므로 해당 시간대 배전 선로 과전압 유의 바람.'
                    
                    if max_ramping >= 30.0:
                        vii_ramping_a = f'시간당 최대 램핑률이 관리 임계치(30MWh/hr)를 초과한 {max_ramping:.2f} MWh/hr로 관측되어 계통 불안정 리스크가 높음.'
                        vii_ramping_b = '급격한 기상 변화에 대응할 수 있도록 부하 조절용 ESS 방전 및 신속 시동 가스터빈 백업 가동 대기가 필수적임.'
                    else:
                        vii_ramping_a = f'시간당 최대 램핑률이 관리 임계치(30MWh/hr) 미만인 {max_ramping:.2f} MWh/hr로 관측되어 계통 동적 예비력은 안정 범위임.'
                        vii_ramping_b = '일반적인 계통 연계 예비력 범위 내에서 관리 가능하므로 추가적인 비상 백업 가동은 불요함.'
                    
                    road_study_1 = '실제 발전 실측치와 AI 예측 오차를 분석하여 일사 감응 피드백 오차 보정 학습을 진행함.'
                    road_vpp_2 = '스마트 인버터 원격 출력 제어 기술 및 분산 태양광 실시간 수집 연계 소프트웨어를 정비할 예정임.'
                else: # 풍력 (Wind)
                    bg_need_1 = '지구 온난화 및 기후 변화로 인해 풍력 발전의 거대 기압골 변동성이 전력망 운영 한계치에 다다르고 있음.'
                    bg_need_2 = '특히 돌발적인 무풍(Calm) 현상 또는 태풍급 강풍에 따른 급격한 출력 차단(Cut-out)으로 계통 주파수 상실 위협이 증대됨.'
                    bg_need_3 = '이에 고정밀 AI LSTM 풍속/기압 감응 모델을 활용한 사전 시뮬레이션으로 안정성을 선제 판정할 필요가 있음.'
                    bg_purpose_1 = f'예측 대상 일자({formatted_target_date})의 예상 날씨에 기초한 {sel_region} 발전 기여 전력량을 사전 연산함.'
                    bg_purpose_2 = '예상되는 출력제어(Curtailment) 리스크를 최소화하고, 기동이 빠른 백업 전원과의 실시간 출력 연계 가동을 대기시켜 계통 주파수를 안정 범위(60Hz +-0.2) 이내로 통제하는 데 기여함을 목적으로 함.'
                    
                    arch_input_dim = '11차원 기상 피처 (기온, 풍속, 풍속세제곱, 풍향, 습도, 현지기압, 전운량, 시간/월 주기 인자)'
                    
                    vi_opinion_a = '주야간 구분 없이 기압골 및 풍속 추이에 유기적으로 대응하여 불규칙하고 역동적인 예측 곡선을 형성함.'
                    vi_opinion_b = '풍속 변화에 따른 돌발성 램핑(출력 급변) 및 컷인 임계값 경계 구간에서 제어 준비 바람.'
                    
                    if max_ramping >= 30.0:
                        vii_ramping_a = f'바람세기 급변에 따른 시간당 최대 램핑률이 관리 임계치(30MWh/hr)를 초과한 {max_ramping:.2f} MWh/hr로 관측되어 계통 요동이 예상됨.'
                        vii_ramping_b = '풍력 출력 급감 시 즉각적인 전력망 보완을 위해 가동 속도가 빠른 가스터빈 및 양수 발전기의 긴급 예비 시동 대기가 시급함.'
                    else:
                        vii_ramping_a = f'바람세기 변동에 따른 시간당 최대 램핑률이 관리 임계치(30MWh/hr) 미만인 {max_ramping:.2f} MWh/hr로 관측되어 계통 순간 동적 예비력은 안정 범위임.'
                        vii_ramping_b = '일상적인 변동 감시 체계 하에서 통제 가능하므로 추가 백업 가동이나 인위적인 출력제어 준비는 불요함.'
                    
                    road_study_1 = '실제 발전 실측치와 AI 예측 오차를 분석하여 풍속-출력 비선형 곡선(Power Curve) 보정 학습을 진행함.'
                    road_vpp_2 = '개별 터빈 요잉/피치 제어 연동 기술 및 대용량 풍력단지 통합 VPP 스케줄러를 고도화할 예정임.'
                
                if use_gps_v2:
                    road_study_2 = f'{sel_region} 권역 내 4개 GPS 세부 단지의 실시간 수치 조정을 위한 앙상블 가중치 보정을 주간 단위로 실시함.'
                else:
                    road_study_2 = f'{sel_region} 관측소 기상 실측 데이터와의 오차를 피드백하여 피크 오차 보정 학습을 주간 단위로 실시함.'

                # 정오 발전 딥(Dip) 여부 판정 로직
                is_noon_dip = False
                if target_energy == '태양광 (Solar)':
                    insol_10 = float(sim_weather_df[sim_weather_df['시간'] == 10]['일사(MJ/m2)'].values[0]) if not sim_weather_df[sim_weather_df['시간'] == 10].empty else 0.0
                    insol_12 = float(sim_weather_df[sim_weather_df['시간'] == 12]['일사(MJ/m2)'].values[0]) if not sim_weather_df[sim_weather_df['시간'] == 12].empty else 0.0
                    insol_14 = float(sim_weather_df[sim_weather_df['시간'] == 14]['일사(MJ/m2)'].values[0]) if not sim_weather_df[sim_weather_df['시간'] == 14].empty else 0.0
                    if insol_12 < insol_10 * 0.9 or insol_12 < insol_14 * 0.9:
                        is_noon_dip = True

                # 계통 위험도 및 실무 가이드라인 동적 판정 (풍속 물리 조건 정정 반영)
                risk_status = '안정'
                risk_desc = '특이 계통 불안정 징후 없음'
                if target_energy == '풍력 (Wind)':
                    max_wind = float(sim_weather_df['풍속(m/s)'].max()) if '풍속(m/s)' in sim_weather_df.columns else 0.0
                    if max_wind >= 25.0:
                        risk_status = '경고'
                        risk_desc = '태풍급 강풍에 따른 터빈 파손 방지 강제 차단(Cut-out) 확정적 위협'
                        guide_a = '풍속이 안전 한계치인 25m/s를 초과하므로 날개 피치 제어 및 기계식 브레이크를 통한 물리적 터빈 보호 긴급 정지를 실시해야 함.'
                        guide_b = '실시간 발전 출력이 급격히 상실(Zero-out)될 수 있으므로 대용량 기저 화력 발전원 및 양수 발전기의 즉각 시동 가동 대기를 발령함.'
                    elif 15.0 <= max_wind < 25.0:
                        risk_status = '주의'
                        risk_desc = '강풍에 따른 풍력 터빈 일부 자동 차단 및 계통 요동 우려'
                        guide_a = '일부 고고도 터빈에서 순간 풍속 임계치 도달에 따른 국소적 출력제어가 예상되므로 계통 감시를 강화함.'
                        guide_b = '풍력 출력 변동에 맞추어 연계된 ESS 충·방전율을 미세 제어하고 백업 전원의 예비 작동을 유지함.'
                    else:
                        risk_status = '안정'
                        risk_desc = f'특이 {sel_region} 풍력 계통 불안정 징후 없음'
                        guide_a = '예측 풍속이 적정 발전 대역(4~12m/s) 이내에서 안정적으로 분포하므로 주파수 흔들림이나 터빈 과부하 위협은 낮음.'
                        guide_b = f'{sel_region} 송배전 선로의 용량 초과 우려가 없으므로 일상적인 감시 체계를 유지하고 추가적인 백업 전원 대기는 불요함.'
                else: # 태양광 (Solar)
                    avg_insol = float(sim_weather_df['일사(MJ/m2)'].mean()) if '일사(MJ/m2)' in sim_weather_df.columns else 0.0
                    if peak_energy >= 200.0:
                        if is_noon_dip:
                            risk_status = '주의'
                            risk_desc = '정오 시간대 구름 유입 및 일시적 광량 급감(Dip) 리스크'
                            guide_a = '정오 부근(12시~14시)에 구름 유입 또는 광량 급감 현상으로 인한 발전량 강하(Dip) 및 전압 요동 위협이 존재함.'
                            guide_b = '급격한 출력 저하 및 전압 보상을 위해 부하 조절용 ESS 방전 대기 및 전력망 전압 조정 장치를 긴급 정비 바람.'
                        else:
                            risk_status = '주의'
                            risk_desc = '정오 시간대 태양광 쏠림 및 계통 과전압 리스크'
                            guide_a = '맑은 기후로 인해 정오(12시~14시) 전력 공급량이 과잉 적재되어 배전 전압이 임계치를 초과할 위협이 존재함.'
                            guide_b = '계통 안정을 위해 ESS 흡수 충전을 최대화하고 필요시 예비 송전선 차단 등의 출력제어(Curtailment) 준비를 완료함.'
                    elif avg_insol < 0.4:
                        risk_status = '주의'
                        risk_desc = '광량 부족에 따른 태양광 기저 전력 급감 우려'
                        guide_a = '흐린 기후 및 미세먼지로 인해 주간 출력 공급량이 평시 대비 급감하여 주간 전력망 여유량이 부족해질 리스크가 상존함.'
                        guide_b = '부하 밀집 지역을 중심으로 즉각 시동 가동이 가능한 가스터빈 분산 백업 전원의 가동 대기를 요망함.'
                    else:
                        risk_status = '안정'
                        risk_desc = f'특이 {sel_region} 태양광 계통 불안정 징후 없음'
                        guide_a = '금일 발전 예측량 및 주간 일사량 기조가 평년 범위 내에 속하므로 송전 계통 과부하 위험이 낮음.'
                        guide_b = '주파수 및 전압 변동 폭이 허용 임계치 이내이므로 정상적인 자동 연계 운전을 유지함.'
                
                import matplotlib.font_manager as fm
                local_font = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fonts', 'malgun.ttf'))
                sys_font = r'C:\Windows\Fonts\malgun.ttf'
                
                font_name = None
                font_prop = None
                
                # 프로젝트 내장 폰트 파일 탐색 및 등록
                if os.path.exists(local_font):
                    try:
                        font_name = fm.FontProperties(fname=local_font).get_name()
                        fm.fontManager.addfont(local_font)
                        font_prop = fm.FontProperties(fname=local_font)
                    except Exception:
                        pass
                
                # Windows 시스템 폰트 폴백
                if font_prop is None and os.path.exists(sys_font):
                    try:
                        font_name = fm.FontProperties(fname=sys_font).get_name()
                        fm.fontManager.addfont(sys_font)
                        font_prop = fm.FontProperties(fname=sys_font)
                    except Exception:
                        pass
                
                if font_name:
                    plt.rcParams['font.family'] = font_name
                else:
                    plt.rcParams['font.family'] = 'sans-serif'
                
                plt.rcParams['axes.unicode_minus'] = False

                fig, ax1 = plt.subplots(figsize=(6.5, 3.2))
                line1 = ax1.plot(np.arange(24), hourly_preds, color='#1f77b4', linewidth=2.5, marker='o', markersize=4, label='예측 발전량')
                ax1.fill_between(np.arange(24), hourly_preds, color='#1f77b4', alpha=0.15)
                ax1.set_ylabel('발전량 (MWh)', color='#1f77b4', fontsize=8)
                ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=8)
                
                ax2 = ax1.twinx()
                if target_energy == '태양광 (Solar)':
                    weather_vals = sim_weather_df['일사(MJ/m2)'].values
                    weather_lbl = '예측 일사량 (MJ/m2)'
                    weather_color = '#d62728'
                else:
                    weather_vals = sim_weather_df['풍속(m/s)'].values
                    weather_lbl = '예측 풍속 (m/s)'
                    weather_color = '#2ca02c'
                line2 = ax2.plot(np.arange(24), weather_vals, color=weather_color, linewidth=2.0, linestyle='--', marker='s', markersize=3, label=weather_lbl)
                ax2.set_ylabel(weather_lbl, color=weather_color, fontsize=8)
                ax2.tick_params(axis='y', labelcolor=weather_color, labelsize=8)
                
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                
                if font_prop:
                    ax1.set_title(f'{formatted_target_date} {sel_region} {target_energy} 24시간 예측 흐름', fontsize=10, fontweight='bold', pad=10, fontproperties=font_prop)
                    ax1.set_xlabel('시간 (시)', fontsize=8, fontproperties=font_prop)
                    ax1.set_ylabel('발전량 (MWh)', color='#1f77b4', fontsize=8, fontproperties=font_prop)
                    ax2.set_ylabel(weather_lbl, color=weather_color, fontsize=8, fontproperties=font_prop)
                    ax1.legend(lines, labels, fontsize=8, loc='upper left', prop=font_prop)
                else:
                    ax1.set_title(f'{formatted_target_date} {sel_region} {target_energy} 24시간 예측 흐름', fontsize=10, fontweight='bold', pad=10)
                    ax1.set_xlabel('시간 (시)', fontsize=8)
                    ax1.legend(lines, labels, fontsize=8, loc='upper left')
                ax1.set_xticks(np.arange(0, 24, 3))
                ax1.grid(True, linestyle='--', alpha=0.5)

                os.makedirs(DOCS_PATH, exist_ok=True)
                temp_chart_path = os.path.join(DOCS_PATH, 'temp_report_chart.png')
                fig.savefig(temp_chart_path, dpi=200, bbox_inches='tight')
                import io
                import base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                encoded_chart = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close(fig)
                
                pages_num = 3
                report_md = f'# [예측 기안문] 재생에너지 일일 발전계통 영향 평가 및 조치 계획\n\n' \
                            f'| 기안 정보 명세 | | | |\n' \
                            f'| :--- | :--- | :--- | :--- |\n' \
                            f'| **문서 번호** | {doc_number} | **기안 부서** | {drafter_dept} |\n' \
                            f'| **기 안 자** | {drafter_name} | **기안 일시** | {current_time} |\n\n' \
                            f'---\n\n' \
                            f'### Ⅰ. 예측 배경 및 목적\n' \
                            f'  1. **배경 및 필요성** :\n' \
                            f'    가. {bg_need_1}\n' \
                            f'    나. {bg_need_2}\n' \
                            f'    다. {bg_need_3}\n' \
                            f'  2. **추진 목적** :\n' \
                            f'    가. {bg_purpose_1}\n' \
                            f'    나. {bg_purpose_2}\n\n' \
                            f'---\n\n' \
                            f'### Ⅱ. AI 예측 모델 아키텍처 및 하이퍼파라미터 명세\n' \
                            f'  1. **신경망 아키텍처 사양** :\n' \
                            f'    가. 네트워크 종류 : LSTM 순환 신경망 모델  /  입력층 차원 : {arch_input_dim}\n' \
                            f'    나. 모델 은닉층 크기 : 64 차원 (Single LSTM Cell)  /  출력층 크기 : 1차원\n' \
                            f'  2. **최적화 및 학습 하이퍼파라미터** :\n' \
                            f'    가. 학습 최적화 알고리즘 : Adam Optimizer (LR = 0.001)  /  손실함수 : MSE Loss\n' \
                            f'    나. 과적합 방지 규제 : Early Stopping (Patience = 15) 및 Dropout (0.2) 적용 완료\n\n' \
                            f'---\n\n' \
                            f'### Ⅲ. 예보 일자 및 타겟 사양 명세\n' \
                            f'  1. **적용 대상 일자** : {formatted_target_date}\n' \
                            f'  2. **분석 타겟 지역** : {sel_region} 행정구역 전역\n' \
                            f'  3. **적용 예측 모델** : {"LSTM 및 GPS v2 초국소 연합 신경망" if use_gps_v2 else "LSTM 지자체별/지역별 독립 적합 신경망"}\n' \
                            f'  4. **기상 입력 모드** : {"AI 확률 날씨 생성 시퀀스" if use_stochastic else "실시간 예보 연동 날씨 데이터"}\n\n' \
                            f'---\n\n' \
                            f'### Ⅳ. 종합 기상 분석 및 발전 예측 요약\n' \
                            f'  1. **일일 누적 예상 발전량** : **{total_energy:.2f} MWh**\n' \
                            f'  2. **평균 기상 예측 기조** :\n' \
                            f'    가. 평균 기온 : {avg_temp:.1f} °C\n' \
                            f'    나. {"평균 일사량" if target_energy == "태양광 (Solar)" else "평균 풍속"} : {avg_wind_or_insol:.2f} {"MJ/m2" if target_energy == "태양광 (Solar)" else "m/s"}\n' \
                            f'  3. **발전 피크 분석** :\n' \
                            f'    가. 금일 발전량이 집중되는 최빈 피크 시각은 **{peak_hour:02d}:00**로 판정됨.\n' \
                            f'    나. 해당 피크 시각의 순간 최대 발전량은 **{peak_energy:.2f} MWh**에 육박할 것으로 전망됨.\n\n' \
                            f'---\n\n' \
                            f'### Ⅴ. 시간대별 세부 예측 발전량 명세\n' \
                            f'| 시간대 | 기온(°C) | {"일사량(MJ/m2)" if target_energy == "태양광 (Solar)" else "풍속(m/s)"} | 예상 발전량 (MWh) |\n' \
                            f'| :---: | :---: | :---: | :---: |\n'
                for h in range(0, 24, 2):
                    weather_val = sim_weather_df.iloc[h]['일사(MJ/m2)'] if target_energy == '태양광 (Solar)' else sim_weather_df.iloc[h]['풍속(m/s)']
                    temp_val = sim_weather_df.iloc[h]['기온(°C)']
                    report_md += f'| {h:02d}:00 | {temp_val:.1f} | {weather_val:.2f} | {hourly_preds[h]:.2f} |\n'
                report_md += f'\n---\n\n' \
                             f'### Ⅵ. 시간대별 발전 예측 시계열 분석 차트\n' \
                             f'![24시간 예측 그래프](data:image/png;base64,{encoded_chart})\n\n' \
                             f'  1. **시계열 흐름 분석 의견** :\n' \
                             f'    가. {vi_opinion_a}\n' \
                             f'    나. {vi_opinion_b}\n\n' \
                             f'---\n\n' \
                             f'### Ⅶ. 시계열 흐름 세부 해석 및 계통 변동성(Ramping Rate) 분석 의견\n' \
                             f'  1. **발전 램핑률(Ramping Rate) 정량 분석 결과** :\n' \
                             f'    가. 금일 발생 예상되는 최대 시간당 발전량 변동폭 : **{max_ramping:.2f} MWh/hr**\n' \
                             f'    나. 최대 변동성 발생 타겟 시각 : **{max_ramping_hour:02d}:00** 전후 발생 판정\n' \
                             f'  2. **계통 영향성 종합 의견** :\n' \
                             f'    가. {vii_ramping_a}\n' \
                             f'    나. {vii_ramping_b}\n\n' \
                             f'---\n\n' \
                             f'### Ⅷ. 전력 계통 안정성 검토 및 조치 가이드라인\n' \
                             f'  1. **계통 위험도 평가 결과** : **[{risk_status}] {risk_desc}**\n' \
                             f'  2. **실무 운영 조치 가이드** :\n' \
                             f'    가. {guide_a}\n' \
                             f'    나. {guide_b}\n\n' \
                             f'---\n\n' \
                             f'### Ⅸ. 향후 2단계 추진 계획 및 초국소 VPP 최적화 로드맵\n' \
                             f'  1. **예측 모델 보정 및 재학습 일정** :\n' \
                             f'    가. {road_study_1}\n' \
                             f'    나. {road_study_2}\n' \
                             f'  2. **가상발전소(VPP) 연계망 고도화** :\n' \
                             f'    가. {road_vpp_2}\n\n' \
                             f'---\n' \
                             f'※ 본 보고서는 전력 계통 사전 통제용 분석 결과로, 실제 전력거래소 실측 실적치와는 차이가 발생할 수 있으며 센서 결함 및 이변 환경에 따라 상시 오차가 감지될 수 있음을 고지함.'

                pdf = PublicSectorPDF(orientation='P', unit='mm', format='A4')
                pdf.set_margins(15, 15, 15)
                pdf.set_auto_page_break(auto=False)
                
                local_font = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fonts', 'malgun.ttf'))
                local_font_bold = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fonts', 'malgunbd.ttf'))
                if os.path.exists(local_font):
                    font_path = local_font
                    font_bold_path = local_font_bold
                else:
                    font_path = r'C:\Windows\Fonts\malgun.ttf'
                    font_bold_path = r'C:\Windows\Fonts\malgunbd.ttf'
                try:
                    pdf.add_font('Malgun', style='', fname=font_path)
                except Exception: pass
                try:
                    pdf.add_font('Malgun', style='B', fname=font_bold_path)
                except Exception:
                    try: pdf.add_font('Malgun', style='B', fname=font_path)
                    except Exception: pass

                # ==================== 1페이지 작성 (Ⅰ, Ⅱ, Ⅲ, Ⅳ) ====================
                pdf.add_page()
                pdf.set_font('Malgun', 'B', 15)
                pdf.cell(0, 12, '[예측 기안문] 재생에너지 일일 계통 영향 평가 및 조치 계획', border=0, ln=1, align='C')
                pdf.ln(3)
                pdf.set_fill_color(240, 244, 248)
                pdf.set_text_color(50, 50, 50)
                pdf.set_draw_color(180, 180, 180)
                w_lbl, w_val = 25, 60
                pdf.set_font('Malgun', 'B', 9)
                pdf.cell(w_lbl, 7, '문서 번호', border=1, fill=True, align='C')
                pdf.set_font('Malgun', '', 9)
                pdf.cell(w_val, 7, doc_number, border=1, align='L')
                pdf.set_font('Malgun', 'B', 9)
                pdf.cell(w_lbl, 7, '기안 부서', border=1, fill=True, align='C')
                pdf.set_font('Malgun', '', 9)
                pdf.cell(w_val, 7, drafter_dept, border=1, ln=1, align='L')
                pdf.set_font('Malgun', 'B', 9)
                pdf.cell(w_lbl, 7, '기 안 자', border=1, fill=True, align='C')
                pdf.set_font('Malgun', '', 9)
                pdf.cell(w_val, 7, drafter_name, border=1, align='L')
                pdf.set_font('Malgun', 'B', 9)
                pdf.cell(w_lbl, 7, '기안 일시', border=1, fill=True, align='C')
                pdf.set_font('Malgun', '', 9)
                pdf.cell(w_val, 7, current_time, border=1, ln=1, align='L')
                pdf.ln(4)
                pdf.line(15, 43, 195, 43)
                pdf.ln(4)

                title_i = '\u2160. \uc608\uce21 \ubc30\uacbd \ubc0f \ubaa9\uc801'
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, title_i, border=0, ln=1)
                pdf.set_font('Malgun', '', 9)
                pdf.add_bullet(2, '1. 배경 및 필요성 :', '')
                pdf.add_bullet(6, '가. ', bg_need_1)
                pdf.add_bullet(6, '나. ', bg_need_2)
                pdf.add_bullet(6, '다. ', bg_need_3)
                pdf.ln(2)
                pdf.add_bullet(2, '2. 추진 목적 :', '')
                pdf.add_bullet(6, '가. ', bg_purpose_1)
                pdf.add_bullet(6, '나. ', bg_purpose_2)
                pdf.ln(4)

                title_ii = '\u2161. AI \uc608\uce21 \ubaa8\ub378 \uc544\ud0a4\ud14d\ucc98 \ubc0f \ud558\uc774\ud37c\ud30c\ub77c\ubbf8\ud130 \uba85\uc138'
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, title_ii, border=0, ln=1)
                pdf.set_font('Malgun', '', 9)
                pdf.add_bullet(2, '1. 신경망 아키텍처 사양 :', '')
                pdf.add_bullet(6, '가. ', '네트워크 종류 : LSTM 순환 신경망 모델  /  입력층 차원 : ' + arch_input_dim)
                pdf.add_bullet(6, '나. ', '모델 은닉층 크기 : 64 차원 (Single LSTM Cell)  /  출력층 크기 : 1차원')
                pdf.ln(2)
                pdf.add_bullet(2, '2. 최적화 및 학습 하이퍼파라미터 :', '')
                pdf.add_bullet(6, '가. ', '학습 최적화 알고리즘 : Adam Optimizer (LR = 0.001)  /  손실함수 : MSE Loss')
                pdf.add_bullet(6, '나. ', '과적합 방지 규제 : Early Stopping (Patience = 15) 및 Dropout (0.2) 적용 완료')
                pdf.ln(4)

                title_iii = '\u2162. \uc608\ubcf4 \uc77c\uc790 \ubc0f \ud0c0\uac9f \uc0ac\uc591 \uba85\uc138'
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, title_iii, border=0, ln=1)
                pdf.set_font('Malgun', '', 9)
                pdf.add_bullet(2, '1. 적용 대상 일자 : ', formatted_target_date, h=5.2)
                pdf.add_bullet(2, '2. 분석 타겟 지역 : ', f'{sel_region} 행정구역 전역', h=5.2)
                pdf.add_bullet(2, '3. 적용 예측 모델 : ', 'LSTM 및 GPS v2 초국소 연합 신경망' if use_gps_v2 else 'LSTM 지자체별/지역별 독립 적합 신경망', h=5.2)
                pdf.add_bullet(2, '4. 기상 입력 모드 : ', 'AI 확률 날씨 생성 시퀀스' if use_stochastic else '실시간 예보 연동 날씨 데이터', h=5.2)
                pdf.ln(4)

                title_iv = '\u2163. \uc885\ud569 \uae30\uc0c1 \ubd84\uc11d \ubc0f \ubc1c\uc804 \uc608\uce21 \uc694\uc57d'
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, title_iv, border=0, ln=1)
                pdf.set_font('Malgun', '', 9)
                pdf.add_bullet(2, '1. 일일 누적 예상 발전량 : ', f'{total_energy:.2f} MWh')
                pdf.add_bullet(2, '2. 평균 기상 예측 기조 : ', f'평균 기온 {avg_temp:.1f} C, 평균 일사량 {avg_wind_or_insol:.2f}' if target_energy == '태양광 (Solar)' else f'평균 기온 {avg_temp:.1f} C, 평균 풍속 {avg_wind_or_insol:.2f}')
                pdf.add_bullet(2, '3. 발전 피크 분석 : ', f'피크 시각 {peak_hour:02d}:00 (순간 최대 {peak_energy:.2f} MWh 예상)')

                # ==================== 2페이지 작성 (Ⅴ, Ⅵ) ====================
                pdf.add_page()
                title_v = '\u2164. \uc2dc\uac04\ub300\ubcc4 \uc138\ubd80 \uc608\uce21 \ubc1c\uc804\ub7c9 \uba85\uc138'
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, title_v, border=0, ln=1)
                pdf.set_fill_color(70, 110, 140)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font('Malgun', 'B', 8)
                col_w = [45, 45, 45, 45]
                pdf.cell(col_w[0], 5, '시간대', border=1, fill=True, align='C')
                pdf.cell(col_w[1], 5, '기온(°C)', border=1, fill=True, align='C')
                pdf.cell(col_w[2], 5, '일사량(MJ/m2)' if target_energy=='태양광 (Solar)' else '풍속(m/s)', border=1, fill=True, align='C')
                pdf.cell(col_w[3], 5, '예상 발전량(MWh)', border=1, fill=True, ln=1, align='C')
                
                pdf.set_text_color(50, 50, 50)
                pdf.set_font('Malgun', '', 8)
                for h in range(0, 24, 2):
                    w_v = sim_weather_df.iloc[h]['일사(MJ/m2)'] if target_energy=='태양광 (Solar)' else sim_weather_df.iloc[h]['풍속(m/s)']
                    t_v = sim_weather_df.iloc[h]['기온(°C)']
                    pdf.cell(col_w[0], 6, f'{h:02d}:00', border=1, align='C')
                    pdf.cell(col_w[1], 6, f'{t_v:.1f}', border=1, align='C')
                    pdf.cell(col_w[2], 6, f'{w_v:.2f}', border=1, align='C')
                    pdf.cell(col_w[3], 6, f'{hourly_preds[h]:.2f}', border=1, ln=1, align='C')
                pdf.ln(5)

                title_vi = '\u2165. \uc2dc\uac04\ub300\ubcc4 \ubc1c\uc804 \uc608\uce21 \uc2dc\uacc4\uc5f4 \ubd84\uc11d \ucc28\ud2b8'
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, title_vi, border=0, ln=1)
                pdf.ln(1)
                if os.path.exists(temp_chart_path):
                    pdf.image(temp_chart_path, x=22, y=pdf.get_y(), w=166, h=78)
                    pdf.ln(80)
                else:
                    pdf.ln(5)

                pdf.set_font('Malgun', 'B', 10)
                pdf.add_bullet(2, '1. 시계열 흐름 분석 의견 :', '', h=6)
                pdf.set_font('Malgun', '', 9)
                pdf.add_bullet(6, '가. ', vi_opinion_a)
                pdf.add_bullet(6, '나. ', vi_opinion_b)

                # ==================== 3페이지 작성 (Ⅶ, Ⅷ, Ⅸ) ====================
                pdf.add_page()
                title_vii = '\u2166. \uc2dc\uacc4\uc5f4 \ud750\ub984 \uc138\ubd80 \ud574\uc11d \ubc0f \uacc4\ud1b5 \ubcc0\ub3d9\uc131(Ramping Rate) \ubd84\uc11d \uc758\uacac'
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, title_vii, border=0, ln=1)
                pdf.set_font('Malgun', '', 9)
                pdf.add_bullet(2, '1. 발전 램핑률(Ramping Rate) 정량 분석 결과 :', '', h=5.2)
                pdf.add_bullet(6, '가. ', f'금일 발생 예상되는 최대 시간당 발전량 변동폭 : {max_ramping:.2f} MWh/hr', h=5.2)
                pdf.add_bullet(6, '나. ', f'최대 변동성 발생 타겟 시각 : {max_ramping_hour:02d}:00 전후 발생 판정', h=5.2)
                pdf.ln(2)
                pdf.add_bullet(2, '2. 계통 영향성 종합 의견 :', '', h=5.2)
                pdf.add_bullet(6, '가. ', vii_ramping_a, h=5.2)
                pdf.add_bullet(6, '나. ', vii_ramping_b, h=5.2)
                pdf.ln(5)

                title_viii = '\u2167. \uc804\ub825 \uacc4\ud1b5 \uc548\uc815\uc131 \uac80\ud1a0 \ubc0f \uc870\uce58 \uac00\uc774\ub4dc\ub77c\uc778'
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, title_viii, border=0, ln=1)
                pdf.set_font('Malgun', '', 9)
                pdf.add_bullet(2, '1. 계통 위험도 평가 결과 : ', f'[{risk_status}] {risk_desc}', h=5.5)
                pdf.ln(2)
                pdf.add_bullet(2, '2. 실무 운영 조치 가이드 :', '', h=5.5)
                pdf.add_bullet(6, '가. ', guide_a, h=5.5)
                pdf.add_bullet(6, '나. ', guide_b, h=5.5)
                pdf.ln(5)

                title_ix = '\u2168. \ud5a5\ud6c4 2\ub2e8\uacc4 \ucd94\uc9c4 \uacc4\ud68d \ubc0f \ucd08\uad6d\uc18c VPP \ucd5c\uc801\ud654 \ub85c\ub4dc\ub9f5'
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, title_ix, border=0, ln=1)
                pdf.set_font('Malgun', '', 9)
                pdf.add_bullet(2, '1. 예측 모델 보정 및 재학습 일정 :', '', h=5.5)
                pdf.add_bullet(6, '가. ', road_study_1, h=5.5)
                pdf.add_bullet(6, '나. ', road_study_2, h=5.5)
                pdf.ln(2)
                pdf.add_bullet(2, '2. 가상발전소(VPP) 연계망 고도화 :', '', h=5.5)
                pdf.add_bullet(6, '가. ', road_vpp_2, h=5.5)
                pdf.ln(8)
                
                pdf.set_font('Malgun', '', 8.5)
                pdf.set_text_color(100, 100, 100)
                pdf.multi_cell(0, 4.5, '※ 본 보고서는 전력 계통 사전 통제용 분석 결과로, 실제 전력거래소 실측 실적치와는 차이가 발생할 수 있으며 센서 결함 및 이변 환경에 따라 상시 오차가 감지될 수 있음을 고지함.')
                
                pdf.ln(2)

                pdf_bytes = bytes(pdf.output())
                if os.path.exists(temp_chart_path):
                    os.remove(temp_chart_path)
                st.markdown(f'#### 📄 기안문 규격 문서 미리보기 (총 {pages_num}페이지)')
                st.markdown(report_md)
                st.divider()
                st.markdown('##### 📥 문서 내보내기')
                md_bytes = report_md.encode('utf-8-sig')
                filename_base = f'전력발전예측보고서_{sel_region}_{target_date.strftime("%Y%m%d")}'
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(
                        label='📄 Markdown 형식 다운로드 (.md)',
                        data=md_bytes,
                        file_name=f'{filename_base}.md',
                        mime='text/markdown',
                        use_container_width=True
                    )
                with col_btn2:
                    st.download_button(
                        label='📕 공문서 규격 PDF 다운로드 (.pdf)',
                        data=pdf_bytes,
                        file_name=f'{filename_base}.pdf',
                        mime='application/pdf',
                        use_container_width=True
                    )
                st.info(f'💡 **실무 가이드**: 다운로드한 PDF는 A4 규격 총 {pages_num}페이지로 완벽하게 포맷팅되어 있으며, 실제 24시간 시계열 그래프가 인라인으로 자동 인쇄되어 있어 상부 보고용으로 즉시 편철 가능합니다.')
        else:
            st.info('👈 왼쪽 기안 정보와 시뮬레이션 설정을 완료하고 \'공문서 규격 보고서 컴파일\' 단추를 누르십시오.')
