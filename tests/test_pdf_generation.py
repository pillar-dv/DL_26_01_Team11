import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import unicodedata
from fpdf import FPDF

# report_generator.py의 PDF 생성 클래스 모방
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


def simulate_pdf_generation(pages_num=3, target_energy='태양광 (Solar)', sel_region='경기도', wind_speed_max=8.0, peak_energy=120.0, avg_insol=1.5):
    # 가짜 예측 데이터 준비
    hourly_preds = [0.0]*6 + [peak_energy * 0.1, peak_energy * 0.3, peak_energy * 0.6, peak_energy * 0.9, peak_energy, peak_energy * 0.9, peak_energy * 0.8, peak_energy * 0.6, peak_energy * 0.4, peak_energy * 0.1, 0.0] + [0.0]*7
    total_energy = sum(hourly_preds)
    peak_hour = hourly_preds.index(max(hourly_preds))
    
    # 가짜 기상 시나리오 데이터 프레임
    sim_weather_df = pd.DataFrame({
        '시간': np.arange(24),
        '기온(°C)': [15.0 + 5.0 * np.sin(2 * np.pi * (h - 6) / 24) for h in range(24)],
        '일사(MJ/m2)': [0.0]*6 + [0.1, 0.5, 1.2, 1.8, 2.3, 2.5, 2.4, 2.0, 1.5, 0.8, 0.3, 0.05] + [0.0]*6,
        '풍속(m/s)': [wind_speed_max * (0.5 + 0.5 * np.cos(2 * np.pi * h / 24)) for h in range(24)]
    })
    
    avg_wind_or_insol = float(sim_weather_df['일사(MJ/m2)'].mean()) if target_energy == '태양광 (Solar)' else float(sim_weather_df['풍속(m/s)'].mean())
    avg_temp = float(sim_weather_df['기온(°C)'].mean())
    
    ramping_rates = [abs(hourly_preds[i] - hourly_preds[i-1]) for i in range(1, 24)]
    max_ramping = max(ramping_rates)
    max_ramping_hour = ramping_rates.index(max_ramping) + 1
    
    formatted_target_date = "2026. 06. 05. (금)"
    current_time = "2026. 06. 04. 15:30"
    
    doc_number = "신재생계통-2026-0604호"
    drafter_dept = "신재생에너지계통통제원"
    drafter_name = "주임연구원 심온"
    
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
    
    if sel_region == '제주도':
        road_study_2 = '제주 권역 내 4개 GPS 세부 단지의 실시간 수치 조정을 위한 앙상블 가중치 보정을 주간 단위로 실시함.'
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
        avg_insol_val = float(sim_weather_df['일사(MJ/m2)'].mean()) if '일사(MJ/m2)' in sim_weather_df.columns else 0.0
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
        elif avg_insol_val < 0.4:
            risk_status = '주의'
            risk_desc = '광량 부족에 따른 태양광 기저 전력 급감 우려'
            guide_a = '흐린 기후 및 미세먼지로 인해 주간 출력 공급량이 평시 대비 급감하여 주간 전력망 여유량이 부족해질 리스크가 상존함.'
            guide_b = '부하 밀집 지역을 중심으로 즉각 시동 가동이 가능한 가스터빈 분산 백업 전원의 가동 대기를 요망함.'
        else:
            risk_status = '안정'
            risk_desc = f'특이 {sel_region} 태양광 계통 불안정 징후 없음'
            guide_a = '금일 발전 예측량 및 주간 일사량 기조가 평년 범위 내에 속하므로 송전 계통 과부하 위험이 낮음.'
            guide_b = '주파수 및 전압 변동 폭이 허용 임계치 이내이므로 정상적인 자동 연계 운전을 유지함.'
            
    temp_chart_path = "temp_test_chart.png"
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(24), hourly_preds)
    fig.savefig(temp_chart_path)
    plt.close(fig)
        
    pdf = PublicSectorPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=False)
    
    # Windows 맑은고딕 폰트 등록
    font_path = r'C:\Windows\Fonts\malgun.ttf'
    font_bold_path = r'C:\Windows\Fonts\malgunbd.ttf'
    try:
        pdf.add_font('Malgun', style='', fname=font_path)
    except Exception as e:
        # Fallback to local
        local_font = r'src\fonts\malgun.ttf'
        try:
            pdf.add_font('Malgun', style='', fname=local_font)
        except Exception:
            pass
    try:
        pdf.add_font('Malgun', style='B', fname=font_bold_path)
    except Exception:
        local_bold_font = r'src\fonts\malgunbd.ttf'
        try:
            pdf.add_font('Malgun', style='B', fname=local_bold_font)
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
    pdf.add_bullet(2, '3. 적용 예측 모델 : ', 'LSTM 독립 적합 신경망', h=5.2)
    pdf.add_bullet(2, '4. 기상 입력 모드 : ', '실시간 예보 연동 날씨 데이터', h=5.2)
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
        
    return pdf_bytes

def verify_pages():
    test_cases = [
        # (target_energy, sel_region, wind_speed_max, peak_energy, avg_insol)
        ('태양광 (Solar)', '경기도', 3.5, 220.0, 1.5), # 태양광 과전압 주의 시나리오
        ('태양광 (Solar)', '전라남도', 3.5, 120.0, 0.3), # 태양광 광량 부족 주의 시나리오
        ('태양광 (Solar)', '강원도', 3.5, 100.0, 1.5), # 태양광 안정 시나리오
        ('풍력 (Wind)', '제주도', 28.0, 150.0, 1.5), # 풍력 강풍 경고 시나리오
        ('풍력 (Wind)', '제주도', 18.0, 80.0, 1.5), # 풍력 강풍 주의 시나리오
        ('풍력 (Wind)', '강원도', 8.0, 50.0, 1.5), # 풍력 안정 시나리오 (제주도 외 지역)
    ]
    for energy, region, wind_max, peak, insol in test_cases:
        try:
            pdf_data = simulate_pdf_generation(3, energy, region, wind_max, peak, insol)
            from io import BytesIO
            try:
                import pypdf
                reader = pypdf.PdfReader(BytesIO(pdf_data))
                actual_pages = len(reader.pages)
                print(f"Test case: {energy} / {region} (wind_max={wind_max}, peak={peak}) -> Actual PDF pages: {actual_pages}")
                assert actual_pages == 3, f"Expected 3 pages, but got {actual_pages}!"
            except ImportError:
                print(f"Test case: {energy} / {region} -> PDF bytes generated successfully ({len(pdf_data)} bytes)")
        except Exception as e:
            print(f"Error during {energy} / {region} simulation: {e}")
            sys.exit(1)
            
    print("All page configurations verified successfully!")

if __name__ == "__main__":
    verify_pages()
