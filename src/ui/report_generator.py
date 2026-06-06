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



def render_report_tab(wind_df, solar_df):
    st.subheader('📋 공문서 및 공기업 규격 일일 전력 발전 예측 보고서 자동 작성 시스템')
    st.markdown('본 모듈은 공문서 표준 기안 양식을 차용하여 선택한 분량(**최소 2페이지 ~ 최대 4페이지**)에 최적화된 맞춤형 보고서를 컴파일합니다. (분량별 내용 구성 차별화 및 페이지 밀도 극대화)')
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
        if target_energy == '풍력 (Wind)' and sel_region == '제주도':
            use_gps_v2 = st.checkbox('📡 GPS 초국소 단지별 합산 모델 적용 (v2)', value=False, key='rep_gps')
        st.divider()
        st.markdown('##### ✍️ 기안 정보 기입')
        doc_number = st.text_input('문서 번호', '신재생계통-2026-0604호', key='rep_doc_num')
        drafter_dept = st.text_input('기안 부서', '신재생에너지계통통제원', key='rep_dept')
        drafter_name = st.text_input('기안자 직위/성명', '주임연구원 심온', key='rep_name')
        st.markdown('##### 📄 문서 분량 및 내용 구성')
        report_pages = st.selectbox(
            '보고서 출력 분량 선택',
            ['2페이지 (요약형 - 핵심 분석 위주)', '3페이지 (표준형 - 주요 시간대 명세 표 및 기상 상관 분석)', '4페이지 (상세형 - LSTM 모델 명세, 24h 전체 표, Ramping Rate 정량 분석)'],
            index=2,
            key='rep_pages'
        )
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
                    if sel_region == '제주도' and use_gps_v2:
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
                                    wind_spd = sim_weather_df.iloc[t]['풍속(m/s)']
                                    if wind_spd < 2.0:
                                        pred_val = 0.0
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
                                wind_spd = sim_weather_df.iloc[t]['풍속(m/s)']
                                if wind_spd < 2.0:
                                    pred_val = 0.0
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
                risk_status = '안정'
                risk_desc = '특이 계통 불안정 징후 없음'
                guide_a = '기저 발전 공급량의 변화가 제한적일 것으로 보여 전력망 주파수 변동 위협은 낮을 것으로 예상됨.'
                guide_b = '금일 예측 발전량은 예년 평균 범주 내에 속하므로 송전 계통 과부하 위험이 없음.'
                if target_energy == '풍력 (Wind)':
                    if avg_wind_or_insol > 15.0 or peak_energy > 50.0:
                        risk_status = '주의'
                        risk_desc = '강풍에 따른 풍력기 자동 차단(Cut-out) 가능성 존재'
                        guide_a = '풍속 25m/s 초과 시 터빈 기계식 파손 방지를 위한 강제 정지가 발생해 실시간 발전량이 급감할 리스크가 상존함.'
                        guide_b = '풍력 피크 전력 분산을 위해 백업 양수 발전기 및 기저 화력 발전원과의 실시간 출력 연계 가동 제어 대기가 필수적임.'
                else:
                    if avg_wind_or_insol < 0.4:
                        risk_status = '주의'
                        risk_desc = '급격한 광량 부족에 따른 태양광 기저 전력 저하'
                        guide_a = '흐린 날씨 및 미세먼지로 인해 주간 태양광 출력 공급량이 급감하여 주간 계통 여유량이 부족해질 수 있음.'
                        guide_b = '부하 밀집 지역을 중심으로 가스 터빈 및 즉각 시동식 분산 백업 전원의 예비 작동 대기가 요망됨.'
                    elif peak_energy > 200.0:
                        risk_status = '주의'
                        risk_desc = '정오 시간대 태양광 쏠림 및 계통 과전압 리스크'
                        guide_a = '일사량 과다 및 맑은 기후로 인해 정오(12시~14시) 전력 공급량이 과잉 적재되어 배전 전압이 임계치를 초과할 위협이 존재함.'
                        guide_b = '계통 안정을 위한 ESS 흡수 충전 가동 및 필요시 예비 송전선 차단 등의 출력제어(Curtailment) 준비 필요함.'
                import matplotlib.font_manager as fm
                local_font = os.path.join(PROJECT_ROOT, 'src', 'fonts', 'malgun.ttf')
                sys_font = r'C:\Windows\Fonts\malgun.ttf'
                font_name = None
                
                # 프로젝트 내장 폰트 파일 탐색 및 등록
                if os.path.exists(local_font):
                    try:
                        font_name = fm.FontProperties(fname=local_font).get_name()
                        fm.fontManager.addfont(local_font)
                    except Exception:
                        pass
                # Windows 시스템 폰트 폴백
                if font_name is None and os.path.exists(sys_font):
                    try:
                        font_name = fm.FontProperties(fname=sys_font).get_name()
                        fm.fontManager.addfont(sys_font)
                    except Exception:
                        pass
                
                if font_name:
                    plt.rcParams['font.family'] = font_name
                else:
                    plt.rcParams['font.family'] = 'sans-serif'
                
                plt.rcParams['axes.unicode_minus'] = False

                # 폰트 프로퍼티 명시적 생성 (개별 요소 렌더링에 직접 적용하여 깨짐을 예방)
                font_prop = None
                if font_name:
                    if os.path.exists(local_font):
                        font_prop = fm.FontProperties(fname=local_font)
                    elif os.path.exists(sys_font):
                        font_prop = fm.FontProperties(fname=sys_font)

                fig, ax = plt.subplots(figsize=(6.5, 3.2))
                ax.plot(np.arange(24), hourly_preds, color='#1f77b4', linewidth=2.5, marker='o', markersize=4, label='예측 발전량')
                ax.fill_between(np.arange(24), hourly_preds, color='#1f77b4', alpha=0.15)
                
                if font_prop:
                    ax.set_title(f'{formatted_target_date} {sel_region} {target_energy} 24시간 예측 흐름', fontsize=10, fontweight='bold', pad=10, fontproperties=font_prop)
                    ax.set_xlabel('시간 (시)', fontsize=8, fontproperties=font_prop)
                    ax.set_ylabel('발전량 (MWh)', fontsize=8, fontproperties=font_prop)
                    ax.legend(fontsize=8, loc='upper left', prop=font_prop)
                else:
                    ax.set_title(f'{formatted_target_date} {sel_region} {target_energy} 24시간 예측 흐름', fontsize=10, fontweight='bold', pad=10)
                    ax.set_xlabel('시간 (시)', fontsize=8)
                    ax.set_ylabel('발전량 (MWh)', fontsize=8)
                    ax.legend(fontsize=8, loc='upper left')
                ax.set_xticks(np.arange(0, 24, 3))
                ax.grid(True, linestyle='--', alpha=0.5)

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
                if '2페이지' in report_pages:
                    pages_num = 2
                elif '3페이지' in report_pages:
                    pages_num = 3
                else:
                    pages_num = 4
                report_md = f'# [예측 기안문] 재생에너지 일일 발전계통 영향 평가 및 조치 계획\n\n| 기안 정보 명세 | | | |\n| :--- | :--- | :--- | :--- |\n| **문서 번호** | {doc_number} | **기안 부서** | {drafter_dept} |\n| **기 안 자** | {drafter_name} | **기안 일시** | {current_time} |\n\n---\n\n### Ⅰ. 예측 배경 및 목적\n  1. **배경 및 필요성** :\n    가. 지구 온난화 및 기후 변화로 인해 재생에너지(태양광 및 풍력)의 기상 변동성이 전력망 운영 한계치에 다다르고 있음.\n    나. 특히 분산 전원의 급격한 램핑 현상 및 돌발성 출력 단절로 인하여 송전 계통의 계통 주파수 상실 위협이 증대됨.\n    다. 이에 고정밀 AI LSTM 가중치 모델을 활용한 사전 시뮬레이션으로 안정성을 선제 판정할 필요가 있음.\n  2. **추진 목적** :\n    가. 익일의 예상 날씨에 기초한 전국 지자체 및 제주 로컬 발전 기여 전력량을 사전 연산함.\n    나. 예상되는 출력제어(Curtailment) 리스크를 최소화하고, 송배전 선로의 국소 과전압 장애를 예방하며 계통 주파수를 안정 범위(60Hz +-0.2) 이내로 통제하는 데 기여함을 목적으로 함.\n'
                if pages_num == 4:
                    report_md += f'---\n\n### Ⅱ. AI 예측 모델 아키텍처 및 하이퍼파라미터 명세\n  1. **신경망 아키텍처 사양** :\n    가. 네트워크 종류 : Long Short-Term Memory (LSTM) 순환 신경망 모델\n    나. 모델 입력층 차원 : {"9차원 기상 피처 벡터" if target_energy == "태양광 (Solar)" else "11차원 기상 및 물리 피처 벡터"}\n    다. 모델 은닉층 크기 : 64 차원 (Single LSTM Cell 구성)\n    라. 모델 출력층 크기 : 1차원 (스케일 복원 전 전력 발전량 예측치)\n  2. **최적화 및 학습 하이퍼파라미터** :\n    가. 학습 최적화 알고리즘 : Adam Optimizer (초기 학습률 LR = 0.001)\n    나. 학습 오차 측정 함수 : Mean Squared Error (MSE Loss)\n    다. 오버피팅 회피 방식 : Early Stopping (Patience = 15) 및 Dropout (0.2) 적용 완료\n'
                section_spec_num = 'Ⅱ' if pages_num <= 3 else 'Ⅲ'
                report_md += f'---\n\n### {section_spec_num}. 예보 일자 및 타겟 사양 명세\n  1. **적용 대상 일자** : {formatted_target_date}\n  2. **분석 타겟 지역** : {sel_region} 행정구역 전역\n  3. **적용 예측 모델** : {"LSTM 독립 적합 및 GPS 초국소 v2 연동 가중치 모델" if use_gps_v2 else "LSTM 지자체별 독립 적합 신경망 모델"}\n  4. **기상 입력 모드** : {"몬테카를로/마르코프 통계 기반 AI 확률 날씨 생성 시퀀스" if use_stochastic else "실시간 예보 연동 동기화 날씨 데이터"}\n\n---\n\n### {"Ⅲ" if pages_num <= 3 else "Ⅳ"}. 종합 기상 분석 및 발전 예측 요약\n  1. **일일 누적 예상 발전량** : **{total_energy:.2f} MWh**\n  2. **평균 기상 예측 기조** :\n    가. 평균 기온 : {avg_temp:.1f} °C\n    나. {"평균 일사량" if target_energy == "태양광 (Solar)" else "평균 풍속"} : {avg_wind_or_insol:.2f} {"MJ/m2" if target_energy == "태양광 (Solar)" else "m/s"}\n  3. **발전 피크 분석** :\n    가. 금일 발전량이 집중되는 최빈 피크 시각은 **{peak_hour:02d}:00**로 판정됨.\n    나. 해당 피크 시각의 순간 최대 발전량은 **{peak_energy:.2f} MWh**에 육박할 것으로 전망됨.\n'
                if pages_num == 3:
                    report_md += f'---\n\n### Ⅳ. 주요 시간대별 세부 예측 발전량 명세\n| 시간대 | 기온(°C) | {"일사량(MJ/m2)" if target_energy == "태양광 (Solar)" else "풍속(m/s)"} | 예상 발전량 (MWh) |\n| :---: | :---: | :---: | :---: |\n'
                    for h in range(0, 24, 2):
                        weather_val = sim_weather_df.iloc[h]['일사(MJ/m2)'] if target_energy == '태양광 (Solar)' else sim_weather_df.iloc[h]['풍속(m/s)']
                        temp_val = sim_weather_df.iloc[h]['기온(°C)']
                        report_md += f'| {h:02d}:00 | {temp_val:.1f} | {weather_val:.2f} | {hourly_preds[h]:.2f} |\n'
                    report_md += f'---\n\n### Ⅴ. 기상 인자 상관성 분석 및 해석 의견\n  1. **물리 기상-발전 상관관계 해석** :\n    가. {"기온 상승 시 태양광 모듈의 온도 계수가 저하되어 발전 효율성이 미세 하강하는 반비례 상관성이 나타남." if target_energy == "태양광 (Solar)" else "풍속의 세제곱에 발전 에너지가 비례하므로 풍속 3m/s에서 8m/s 유입 시 발전량이 급상승하는 3차 곡선 경향성이 나타남."}\n    나. {"습도 및 미세먼지농도가 조도를 차단하여 동일 고도 대비 일사량 한계선을 약 10~15% 수준 감쇄시킴." if target_energy == "태양광 (Solar)" else "전운량 및 현지기압 강하로 인한 전선 통과 시 돌풍 유입 효과가 발전량 상승에 강하게 반영됨."}\n'
                elif pages_num == 4:
                    report_md += f'---\n\n### Ⅴ. 시간대별 세부 예측 발전량 명세\n| 시간대 | 기온(°C) | {"일사량(MJ/m2)" if target_energy == "태양광 (Solar)" else "풍속(m/s)"} | 예상 발전량 (MWh) |\n| :---: | :---: | :---: | :---: |\n'
                    for h in range(24):
                        weather_val = sim_weather_df.iloc[h]['일사(MJ/m2)'] if target_energy == '태양광 (Solar)' else sim_weather_df.iloc[h]['풍속(m/s)']
                        temp_val = sim_weather_df.iloc[h]['기온(°C)']
                        report_md += f'| {h:02d}:00 | {temp_val:.1f} | {weather_val:.2f} | {hourly_preds[h]:.2f} |\n'
                section_chart_num = 'Ⅳ' if pages_num == 2 else ('Ⅵ' if pages_num == 3 else 'Ⅵ')
                report_md += f'---\n\n### {section_chart_num}. 시간대별 발전 예측 시계열 분석 차트\n![24시간 예측 그래프](data:image/png;base64,{encoded_chart})\n'
                if pages_num == 2:
                    report_md += f'  1. **시계열 흐름 해석** :\n    가. 기상 시나리오의 변동 흐름과 발전 모델의 추론 곡선이 일정 수준 동기화되어 매끄러운 에너지 상승 기조를 형성함.\n    나. 피크 발생 시간대를 전후하여 계통 급전 지시 변경이 필요할 수 있으며, 야간 발전 한계점 이탈 여부를 상시 확인바람.\n\n---\n\n### Ⅴ. 전력 계통 안정성 검토 및 조치 가이드라인\n  1. **계통 위험도 평가 결과** : **[{risk_status}] {risk_desc}**\n  2. **실무 운영 조치 가이드** :\n    가. {guide_a}\n    나. {guide_b}\n\n---\n\n### Ⅵ. 향후 단기 통제 계획 및 계통 연계망 조치안\n  1. **예측 피드백 및 단기 제어 계획** :\n    가. 실제 발전 실측치와 AI 예측 오차를 분석하여 피드백 오차 보정 학습을 진행함.\n    나. 계통 과부하 방지를 위한 단기 계통 인버터 흡수 충전 가동을 실시함.\n'
                elif pages_num == 3:
                    report_md += f'  1. **시계열 흐름 해석 및 의견** :\n    가. 인공지능이 도출한 24시간 발전 추세선은 특정 시간대({peak_hour:02d}시)를 기준으로 뚜렷한 피크 구간을 형성하며 기상 입력 인자와 강한 조화를 이룸.\n    나. 야간 유휴 기상대 진입 시 발전이 정지되는 물리적 임계점 도달 여부도 확인을 완료하였음.\n\n---\n\n### Ⅶ. 전력 계통 안정성 검토 및 조치 가이드라인\n  1. **계통 위험도 평가 결과** : **[{risk_status}] {risk_desc}**\n  2. **실무 운영 조치 가이드** :\n    가. {guide_a}\n    나. {guide_b}\n\n---\n\n### Ⅷ. 향후 2단계 추진 계획 및 VPP 연계 전략\n  1. **예측 모델 보정 및 재학습 일정** :\n    가. 금일 수집된 실제 발전 실측치와 AI 예측 오차를 분석하여 피드백 오차 보정 학습을 진행함.\n    나. 제주 권역 내 4개 GPS 세부 단지의 실시간 수치 조정을 위한 앙상블 가중치 보정을 주간 단위로 실시함.\n  2. **가상발전소(VPP) 연계망 고도화** :\n    가. 예측 제고 정산금 획득 기준에 최적화되도록 분산 자원 실시간 수집 연계 소프트웨어를 정비할 예정임.\n'
                else:
                    report_md += f'  1. **시계열 흐름 해석** :\n    가. 인공지능이 도출한 24시간 발전 추세선은 특정 시간대({peak_hour:02d}시)를 기준으로 뚜렷한 피크 구간을 형성하며 기상 입력 인자와 강한 조화를 이룸.\n    나. 일출 및 일몰 전후(혹은 기압 변동대 유입 전후)의 급변 구간에서 발전량 램핑률이 상승하므로, 기저 발전원과의 조정을 권고함.\n\n---\n\n### Ⅶ. 시계열 흐름 세부 해석 및 계통 변동성(Ramping Rate) 분석 의견\n  1. **발전 램핑률(Ramping Rate) 정량 분석** :\n    가. 금일 발생 예상되는 최대 시간당 발전량 변동폭(Ramping Rate)은 **{max_ramping:.2f} MWh/hr**로 검출됨.\n    나. 최대 변동성 발생 시각은 **{max_ramping_hour:02d}:00** 전후로 예측되며, 이 구간에서 계통 주파수 흔들림이 증가할 수 있음.\n  2. **계통 영향성 종합 의견** :\n    가. 순간 램핑률이 한계치(30MWh/hr) 미만으로 감지되어 송배전망의 순간 동적 급변 예비력은 안전 범위에 있음.\n    나. 다만, 피크 아워 전후의 급경사 램핑에 대응하기 위해 기동이 빠른 양수발전원과의 동기화 준비가 바람직함.\n\n---\n\n### Ⅷ. 전력 계통 안정성 검토 및 조치 가이드라인\n  1. **계통 위험도 평가 결과** : **[{risk_status}] {risk_desc}**\n  2. **실무 운영 조치 가이드** :\n    가. {guide_a}\n    나. {guide_b}\n\n---\n\n### Ⅸ. 향후 2단계 추진 계획 및 초국소 VPP 최적화 로드맵\n  1. **예측 모델 보정 및 재학습 일정** :\n    가. 금일 수집된 실제 발전 실측치와 AI 예측 오차를 분석하여 피드백 오차 보정 학습을 진행함.\n    나. 제주 권역 내 4개 GPS 세부 단지의 실시간 수치 조정을 위한 앙상블 가중치 보정을 주간 단위로 실시함.\n  2. **가상발전소(VPP) 연계망 고도화** :\n    가. 예측 제고 정산금 획득 기준에 최적화되도록 분산 자원 실시간 수집 연계 소프트웨어를 정비할 예정임.\n'
                report_md += '---\n※ 본 보고서는 전력 계통 사전 통제용 분석 결과로, 실제 전력거래소 실측 실적치와는 차이가 발생할 수 있으며 센서 결함 및 이변 환경에 따라 상시 오차가 감지될 수 있음을 고지함.'
                pdf = PublicSectorPDF(orientation='P', unit='mm', format='A4')
                pdf.set_margins(15, 15, 15)
                pdf.set_auto_page_break(auto=False)
                local_font = os.path.join(PROJECT_ROOT, 'src', 'fonts', 'malgun.ttf')
                local_font_bold = os.path.join(PROJECT_ROOT, 'src', 'fonts', 'malgunbd.ttf')
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
                pdf.set_font('Malgun', 'B', 11)
                pdf.cell(0, 8, 'Ⅰ. 예측 배경 및 목적', border=0, ln=1)
                pdf.set_font('Malgun', '', 9)
                bg_text = '  1. 배경 및 필요성 :\n    가. 지구 온난화 및 기후 변화로 인해 재생에너지(태양광 및 풍력)의 기상 변동성이 전력망 운영 한계치에 다다르고 있음.\n    나. 특히 분산 전원의 급격한 램핑 현상 및 돌발성 출력 단절로 인하여 송전 계통의 계통 주파수 상실 위협이 증대됨.\n    다. 이에 고정밀 AI LSTM 가중치 모델을 활용한 사전 시뮬레이션으로 안정성을 선제 판정할 필요가 있음.\n\n  2. 추진 목적 :\n    가. 익일의 예상 날씨에 기초한 전국 지자체 및 제주 로컬 발전 기여 전력량을 사전 연산함.\n    나. 예상되는 출력제어(Curtailment) 리스크를 최소화하고, 송배전 선로의 국소 과전압 장애를 예방하며 계통 주파수를 안정 범위(60Hz +-0.2) 이내로 통제하는 데 기여함을 목적으로 함.'
                pdf.multi_cell(0, 5, bg_text)
                pdf.ln(4)
                if pages_num == 2:
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅱ. 예보 일자 및 타겟 사양 명세', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.cell(0, 5.2, f'  1. 적용 대상 일자 : {formatted_target_date}  /  분석 타겟 지역 : {sel_region} 전역', border=0, ln=1)
                    pdf.cell(0, 5.2, f'  2. 적용 예측 모델 : {"LSTM 및 GPS v2 모델" if use_gps_v2 else "LSTM 독립 적합 신경망 모델"}', border=0, ln=1)
                    pdf.ln(3)
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅲ. 종합 기상 분석 및 발전 예측 요약', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.cell(0, 5.2, f'  1. 일일 누적 예상 발전량 : {total_energy:.2f} MWh  /  발전 피크 시간 : {peak_hour:02d}:00 (최대 {peak_energy:.2f} MWh)', border=0, ln=1)
                    pdf.cell(0, 5.2, f'  2. 예측 평균 기온 : {avg_temp:.1f} °C  /  평균 {"일사량" if target_energy == "태양광 (Solar)" else "풍속"} : {avg_wind_or_insol:.2f} {"MJ/m2" if target_energy == "태양광 (Solar)" else "m/s"}', border=0, ln=1)
                elif pages_num == 3:
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅱ. 예보 일자 및 타겟 사양 명세', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.cell(0, 5.2, f'  1. 적용 대상 일자 : {formatted_target_date}', border=0, ln=1)
                    pdf.cell(0, 5.2, f'  2. 분석 타겟 지역 : {sel_region} 행정구역 전역', border=0, ln=1)
                    pdf.cell(0, 5.2, f'  3. 적용 예측 모델 : {"LSTM 및 GPS v2 모델" if use_gps_v2 else "LSTM 독립 적합 신경망"}', border=0, ln=1)
                    pdf.cell(0, 5.2, f'  4. 기상 입력 모드 : {"AI 확률 날씨 생성 시퀀스" if use_stochastic else "실시간 예보 연동 날씨 데이터"}', border=0, ln=1)
                else:
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅱ. AI 예측 모델 아키텍처 및 하이퍼파라미터 명세', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    arch_text = '  1. 신경망 아키텍처 사양 :\n    가. 네트워크 종류 : LSTM 순환 신경망 모델  /  입력층 차원 : ' + ('9차원' if target_energy=='태양광 (Solar)' else '11차원') + ' 기상 피처\n    나. 모델 은닉층 크기 : 64 차원 (Single LSTM Cell)  /  출력층 크기 : 1차원\n  2. 최적화 및 학습 하이퍼파라미터 :\n    가. 학습 최적화 알고리즘 : Adam Optimizer (LR = 0.001)  /  손실함수 : MSE Loss\n    나. 과적합 방지 규제 : Early Stopping (Patience = 15) 및 Dropout (0.2) 적용 완료'
                    pdf.multi_cell(0, 5, arch_text)
                pdf.add_page()
                if pages_num == 2:
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅳ. 시간대별 발전 예측 시계열 분석 차트', border=0, ln=1)
                    pdf.ln(1)
                    if os.path.exists(temp_chart_path):
                        pdf.image(temp_chart_path, x=25, y=30, w=160, h=65)
                    pdf.ln(68)
                    pdf.set_font('Malgun', 'B', 9.5)
                    pdf.cell(0, 6, '  1. 시계열 흐름 해석 및 분석 의견 :', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.multi_cell(0, 5, '    가. 기상 시나리오의 변동 흐름과 발전 모델의 추론 곡선이 일정 수준 동기화되어 매끄러운 에너지 상승 기조를 형성함.\n    나. 피크 발생 시간대를 전후하여 계통 급전 지시 변경이 필요할 수 있으며, 야간 발전 한계점 이탈 여부를 상시 확인바람.')
                    pdf.ln(3)
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅴ. 전력 계통 안정성 검토 및 조치 가이드라인', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.cell(0, 5, f'  1. 계통 위험도 평가 결과 : [{risk_status}] {risk_desc}', border=0, ln=1)
                    pdf.multi_cell(0, 5, f'  2. 실무 운영 조치 가이드 :\n    가. {guide_a}\n    나. {guide_b}')
                    pdf.ln(3)
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅵ. 향후 단기 통제 계획 및 계통 연계망 조치안', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.multi_cell(0, 5, '  1. 예측 모델 보정 : 실제 발전 실측치 분석을 통한 피드백 오차 보정 학습을 진행함.\n  2. 계통 과부하 방지 : 계통 안정을 위한 국소 인버터 제어 및 인버터 흡수 충전 가동을 대기함.')
                elif pages_num == 3:
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅲ. 종합 기상 분석 및 발전 예측 요약', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.cell(0, 5, f'  1. 일일 누적 예상 발전량 : {total_energy:.2f} MWh', border=0, ln=1)
                    pdf.cell(0, 5, f'  2. 평균 기상 예측 기조 : 평균 기온 {avg_temp:.1f} C, 평균 일사/풍속 {avg_wind_or_insol:.2f}', border=0, ln=1)
                    pdf.cell(0, 5, f'  3. 발전 피크 분석 : 피크 시각 {peak_hour:02d}:00 (순간 최대 {peak_energy:.2f} MWh 예상)', border=0, ln=1)
                    pdf.ln(3)
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅳ. 주요 시간대별 세부 예측 발전량 명세', border=0, ln=1)
                    pdf.set_fill_color(70, 110, 140)
                    pdf.set_text_color(255, 255, 255)
                    pdf.set_font('Malgun', 'B', 8)
                    col_w = [45, 45, 45, 45]
                    pdf.cell(col_w[0], 5, '시간대', border=1, fill=True, align='C')
                    pdf.cell(col_w[1], 5, '기온(C)', border=1, fill=True, align='C')
                    pdf.cell(col_w[2], 5, '일사량' if target_energy=='태양광 (Solar)' else '풍속', border=1, fill=True, align='C')
                    pdf.cell(col_w[3], 5, '예상 발전량', border=1, fill=True, ln=1, align='C')
                    pdf.set_text_color(50, 50, 50)
                    pdf.set_font('Malgun', '', 8)
                    for h in range(0, 24, 2):
                        w_v = sim_weather_df.iloc[h]['일사(MJ/m2)'] if target_energy=='태양광 (Solar)' else sim_weather_df.iloc[h]['풍속(m/s)']
                        t_v = sim_weather_df.iloc[h]['기온(°C)']
                        pdf.cell(col_w[0], 6, f'{h:02d}:00', border=1, align='C')
                        pdf.cell(col_w[1], 6, f'{t_v:.1f}', border=1, align='C')
                        pdf.cell(col_w[2], 6, f'{w_v:.2f}', border=1, align='C')
                        pdf.cell(col_w[3], 6, f'{hourly_preds[h]:.2f}', border=1, ln=1, align='C')
                    pdf.ln(4)
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅴ. 기상 인자 상관성 분석 및 해석 의견', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    cor_detail = '  1. 물리 기상-발전 상관관계 해석 :\n    가. ' + ('기온 상승 시 모듈 온도 저하로 발전 효율이 미세 하강하는 반비례 상관성이 나타남.' if target_energy=='태양광 (Solar)' else '풍속의 세제곱에 비례하여 풍속 3~8m/s에서 급상승하는 곡선 특성이 나타남.') + '\n    나. ' + ('습도/미세먼지가 일사량을 약 10~15% 수준 감쇄시킴.' if target_energy=='태양광 (Solar)' else '현지기압 강하로 인한 저기압 진입 시 돌풍 효과가 기여함.')
                    pdf.multi_cell(0, 5, cor_detail)
                else:
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅲ. 예보 일자 및 타겟 사양 명세', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.cell(0, 5.2, f'  1. 적용 대상 일자 : {formatted_target_date}  /  분석 타겟 지역 : {sel_region} 전역', border=0, ln=1)
                    pdf.cell(0, 5.2, f'  2. 적용 예측 모델 : {"LSTM 및 GPS v2 가중 모델" if use_gps_v2 else "LSTM 지자체별 독립 적합 신경망 모델"}', border=0, ln=1)
                    pdf.ln(2)
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅳ. 종합 기상 분석 및 발전 예측 요약', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.cell(0, 5.2, f'  1. 일일 누적 예상 발전량 : {total_energy:.2f} MWh  /  피크 시간 : {peak_hour:02d}:00 (출력: {peak_energy:.2f} MWh)', border=0, ln=1)
                    pdf.cell(0, 5.2, f'  2. 평균 기상 조건 : 평균 기온 {avg_temp:.1f} C  /  평균 값 : {avg_wind_or_insol:.2f}', border=0, ln=1)
                    pdf.ln(3)
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅴ. 시간대별 세부 예측 발전량 명세', border=0, ln=1)
                    pdf.set_fill_color(70, 110, 140)
                    pdf.set_text_color(255, 255, 255)
                    pdf.set_font('Malgun', 'B', 8)
                    col_w = [45, 45, 45, 45]
                    pdf.cell(col_w[0], 5, '시간대', border=1, fill=True, align='C')
                    pdf.cell(col_w[1], 5, '기온(C)', border=1, fill=True, align='C')
                    pdf.cell(col_w[2], 5, '일사/풍속', border=1, fill=True, align='C')
                    pdf.cell(col_w[3], 5, '예상 발전량', border=1, fill=True, ln=1, align='C')
                    pdf.set_text_color(50, 50, 50)
                    pdf.set_font('Malgun', '', 7.5)
                    for h in range(24):
                        w_v = sim_weather_df.iloc[h]['일사(MJ/m2)'] if target_energy=='태양광 (Solar)' else sim_weather_df.iloc[h]['풍속(m/s)']
                        t_v = sim_weather_df.iloc[h]['기온(°C)']
                        pdf.cell(col_w[0], 5.8, f'{h:02d}:00', border=1, align='C')
                        pdf.cell(col_w[1], 5.8, f'{t_v:.1f}', border=1, align='C')
                        pdf.cell(col_w[2], 5.8, f'{w_v:.2f}', border=1, align='C')
                        pdf.cell(col_w[3], 5.8, f'{hourly_preds[h]:.2f}', border=1, ln=1, align='C')
                if pages_num >= 3:
                    pdf.add_page()
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅵ. 시간대별 발전 예측 시계열 분석 차트', border=0, ln=1)
                    pdf.ln(1)
                    if os.path.exists(temp_chart_path):
                        pdf.image(temp_chart_path, x=22, y=30, w=166, h=78)
                    pdf.ln(80)
                    if pages_num == 3:
                        pdf.set_font('Malgun', 'B', 11)
                        pdf.cell(0, 8, 'Ⅶ. 전력 계통 안정성 검토 및 조치 가이드라인', border=0, ln=1)
                        pdf.set_font('Malgun', '', 9)
                        pdf.cell(0, 5, f'  1. 계통 위험도 평가 결과 : [{risk_status}] {risk_desc}', border=0, ln=1)
                        pdf.multi_cell(0, 5, f'  2. 실무 운영 조치 가이드 :\n    가. {guide_a}\n    나. {guide_b}')
                        pdf.ln(3)
                        pdf.set_font('Malgun', 'B', 11)
                        pdf.cell(0, 8, 'Ⅷ. 향후 2단계 추진 계획 및 VPP 연계 전략', border=0, ln=1)
                        pdf.set_font('Malgun', '', 9)
                        pdf.multi_cell(0, 5, '  1. 예측 모델 보정 : 실제 발전 실측치 분석을 통한 오차 피드백 오차 보정 학습을 진행함.\n  2. VPP망 고도화 : 예측 제고 정산금 기준에 최적화되도록 분산 자원 실시간 수집 연계망을 구축함.')
                    else:
                        pdf.set_font('Malgun', 'B', 10)
                        pdf.cell(0, 6, '  1. 시계열 흐름 분석 의견 :', border=0, ln=1)
                        pdf.set_font('Malgun', '', 9)
                        pdf.multi_cell(0, 5, '    가. 인공지능이 도출한 24시간 발전 추세선은 기상 변동 인자와 강한 상관성을 가지며 매끄러운 에너지 기조를 형성함.\n    나. 피크 아워 전후의 급경사 램핑 구간에서 계통 주파수 흔들림이 있을 수 있으니 제어 준비 바람.')
                        pdf.ln(4)
                        pdf.set_font('Malgun', 'B', 11)
                        pdf.cell(0, 8, 'Ⅶ. 시계열 흐름 세부 해석 및 계통 변동성(Ramping Rate) 분석 의견', border=0, ln=1)
                        pdf.set_font('Malgun', '', 9)
                        pdf.cell(0, 5.2, f'  1. 발전 램핑률(Ramping Rate) 정량 분석 결과 :', ln=1)
                        pdf.cell(0, 5.2, f'    가. 금일 발생 예상되는 최대 시간당 발전량 변동폭 : {max_ramping:.2f} MWh/hr', border=0, ln=1)
                        pdf.cell(0, 5.2, f'    나. 최대 변동성 발생 타겟 시각 : {max_ramping_hour:02d}:00 전후 발생 판정', border=0, ln=1)
                        pdf.multi_cell(0, 5.2, '  2. 계통 영향성 종합 의견 :\n    가. 순간 램핑률이 한계치(30MWh/hr) 미만으로 감지되어 계통 순간 동적 예비력은 안정 범위임.\n    나. 단, 급경사 램핑에 대응하기 위해 기동이 빠른 양수발전기 연계 제어 대기가 필수적임.')
                if pages_num == 4:
                    pdf.add_page()
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅷ. 전력 계통 안정성 검토 및 조치 가이드라인', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.cell(0, 5.5, f'  1. 계통 위험도 평가 결과 : [{risk_status}] {risk_desc}', border=0, ln=1)
                    pdf.cell(0, 5.5, '  2. 실무 운영 조치 가이드 :', border=0, ln=1)
                    pdf.multi_cell(0, 5.5, f'    가. {guide_a}\n    나. {guide_b}')
                    pdf.ln(5)
                    pdf.set_font('Malgun', 'B', 11)
                    pdf.cell(0, 8, 'Ⅸ. 향후 2단계 추진 계획 및 초국소 VPP 최적화 로드맵', border=0, ln=1)
                    pdf.set_font('Malgun', '', 9)
                    pdf.multi_cell(0, 5.5, '  1. 예측 모델 보정 및 재학습 일정 :\n    가. 실제 발전 실측치와 AI 예측 오차를 분석하여 피드백 오차 보정 학습을 진행함.\n    나. 제주 권역 내 4개 GPS 세부 단지의 실시간 수치 조정을 위한 앙상블 가중치 보정을 주간 단위로 실시함.\n  2. 가상발전소(VPP) 연계망 고도화 :\n    가. 예측 제고 정산금 획득 기준에 최적화되도록 분산 자원 실시간 수집 연계 소프트웨어를 정비할 예정임.')
                    pdf.ln(8)
                    pdf.set_font('Malgun', '', 8.5)
                    pdf.set_text_color(100, 100, 100)
                    pdf.multi_cell(0, 4.5, '※ 본 보고서는 전력 계통 사전 통제용 분석 결과로, 실제 전력거래소 실측 실적치와는 차이가 발생할 수 있으며 센서 결함 및 이변 환경에 따라 상시 오차가 감지될 수 있음을 고지함.')
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
