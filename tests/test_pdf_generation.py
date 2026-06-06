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

def simulate_pdf_generation(pages_num=3, target_energy='태양광 (Solar)', sel_region='경기도'):
    # 가짜 예측 데이터 준비
    hourly_preds = [0.0]*6 + [10.5, 25.0, 50.3, 120.0, 180.2, 210.5, 220.0, 195.4, 150.1, 90.2, 30.5, 5.0] + [0.0]*6
    total_energy = sum(hourly_preds)
    peak_energy = max(hourly_preds)
    peak_hour = hourly_preds.index(peak_energy)
    
    # 가짜 기상 시나리오 데이터 프레임
    sim_weather_df = pd.DataFrame({
        '시간': np.arange(24),
        '기온(°C)': [15.0 + 5.0 * np.sin(2 * np.pi * (h - 6) / 24) for h in range(24)],
        '일사(MJ/m2)': [0.0]*6 + [0.1, 0.5, 1.2, 1.8, 2.3, 2.5, 2.4, 2.0, 1.5, 0.8, 0.3, 0.05] + [0.0]*6,
        '풍속(m/s)': [3.5]*24
    })
    
    avg_wind_or_insol = float(sim_weather_df['일사(MJ/m2)'].mean())
    avg_temp = float(sim_weather_df['기온(°C)'].mean())
    
    ramping_rates = [abs(hourly_preds[i] - hourly_preds[i-1]) for i in range(1, 24)]
    max_ramping = max(ramping_rates)
    max_ramping_hour = ramping_rates.index(max_ramping) + 1
    
    formatted_target_date = "2026. 06. 05. (금)"
    current_time = "2026. 06. 04. 15:30"
    
    doc_number = "신재생계통-2026-0604호"
    drafter_dept = "신재생에너지계통통제원"
    drafter_name = "주임연구원 심온"
    
    risk_status = '안정'
    risk_desc = '특이 계통 불안정 징후 없음'
    guide_a = '기저 발전 공급량의 변화가 제한적일 것으로 보여 전력망 주파수 변동 위협은 낮을 것으로 예상됨.'
    guide_b = '금일 예측 발전량은 예년 평균 범주 내에 속하므로 송전 계통 과부하 위험이 없음.'
    
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
    bg_text = '  1. 배경 및 필요성 :\n    가. 지구 온난화 및 기후 변화로 인해 재생에너지(태양광 및 풍력)의 기상 변동성이 전력망 운영 한계치에 다다르고 있음.\n    나. 특히 분산 전원의 급격한 램핑 현상 및 돌발성 출력 단절로 인하여 송전 계통의 계통 주파수 상실 위협이 증대됨.\n    다. 이에 고정밀 AI LSTM 가중치 모델을 활용한 사전 시뮬레이션으로 안정성을 선제 판정할 필요가 있음.\n\n  2. 추진 목적 :\n    가. 익일의 예상 날씨에 기초한 전국 지자체 및 제주 로컬 발전 기여 전력량을 사전 연산함.\n    나. 예상되는 출력제어(Curtailment) 리스크를 최소화하고, 송배전 선로의 국소 과전압 장애를 예방하며 계통 주파수를 안정 범위(60Hz +-0.2) 이내로 통제하는 데 기여함을 목적으로 함.'
    pdf.multi_cell(0, 5, bg_text)
    pdf.ln(4)
    
    title_ii = '\u2161. AI \uc608\uce21 \ubaa8\ub378 \uc544\ud0a4\ud14d\ucc28 \ubc0f \ud558\uc774\ud37c\ud30c\ub77c\uba54\ud130 \uba85\uc138'
    pdf.set_font('Malgun', 'B', 11)
    pdf.cell(0, 8, title_ii, border=0, ln=1)
    pdf.set_font('Malgun', '', 9)
    arch_text = '  1. 신경망 아키텍처 사양 :\n    가. 네트워크 종류 : LSTM 순환 신경망 모델  /  입력층 차원 : ' + ('9차원' if target_energy=='태양광 (Solar)' else '11차원') + ' 기상 피처\n    나. 모델 은닉층 크기 : 64 차원 (Single LSTM Cell)  /  출력층 크기 : 1차원\n  2. 최적화 및 학습 하이퍼파라미터 :\n    가. 학습 최적화 알고리즘 : Adam Optimizer (LR = 0.001)  /  손실함수 : MSE Loss\n    나. 과적합 방지 규제 : Early Stopping (Patience = 15) 및 Dropout (0.2) 적용 완료'
    pdf.multi_cell(0, 5, arch_text)
    pdf.ln(4)
    
    title_iii = '\u2162. \uc608\ubcf4 \uc77c\uc790 \ubc0f \ud0c0\uac9f \uc0ac\uc596 \uba85\uc138'
    pdf.set_font('Malgun', 'B', 11)
    pdf.cell(0, 8, title_iii, border=0, ln=1)
    pdf.set_font('Malgun', '', 9)
    pdf.cell(0, 5.2, f'  1. 적용 대상 일자 : {formatted_target_date}', border=0, ln=1)
    pdf.cell(0, 5.2, f'  2. 분석 타겟 지역 : {sel_region} 행정구역 전역', border=0, ln=1)
    pdf.cell(0, 5.2, f'  3. 적용 예측 모델 : LSTM 독립 적합 신경망', border=0, ln=1)
    pdf.cell(0, 5.2, f'  4. 기상 입력 모드 : 실시간 예보 연동 날씨 데이터', border=0, ln=1)
    pdf.ln(4)

    title_iv = '\u2163. \uc885\ud569 \uae30\uc0c1 \ubd84\uc11d \ubc0f \ubc1c\uc804 \uc608\uce21 \uc694\uc57d'
    pdf.set_font('Malgun', 'B', 11)
    pdf.cell(0, 8, title_iv, border=0, ln=1)
    pdf.set_font('Malgun', '', 9)
    pdf.cell(0, 5, f'  1. 일일 누적 예상 발전량 : {total_energy:.2f} MWh', border=0, ln=1)
    pdf.cell(0, 5, f'  2. 평균 기상 예측 기조 : 평균 기온 {avg_temp:.1f} C, 평균 일사량 {avg_wind_or_insol:.2f}', border=0, ln=1)
    pdf.cell(0, 5, f'  3. 발전 피크 분석 : 피크 시각 {peak_hour:02d}:00 (순간 최대 {peak_energy:.2f} MWh 예상)', border=0, ln=1)
        
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
    pdf.cell(col_w[1], 5, '기온(C)', border=1, fill=True, align='C')
    pdf.cell(col_w[2], 5, '일사량', border=1, fill=True, align='C')
    pdf.cell(col_w[3], 5, '예상 발전량', border=1, fill=True, ln=1, align='C')
    pdf.set_text_color(50, 50, 50)
    pdf.set_font('Malgun', '', 8)
    for h in range(0, 24, 2):
        w_v = sim_weather_df.iloc[h]['일사(MJ/m2)']
        t_v = sim_weather_df.iloc[h]['기온(°C)']
        pdf.cell(col_w[0], 6, f'{h:02d}:00', border=1, align='C')
        pdf.cell(col_w[1], 6, f'{t_v:.1f}', border=1, align='C')
        pdf.cell(col_w[2], 6, f'{w_v:.2f}', border=1, align='C')
        pdf.cell(col_w[3], 6, f'{hourly_preds[h]:.2f}', border=1, ln=1, align='C')
    pdf.ln(5)
    
    title_vi = '\u2165. \uc2dc\uac04\ub300\ubcc4 \ubc1c\uc804 \uc608\uc121 \uc2dc\uacc4\uc5f4 \ubd84\uc11d \ucc28\ud2b8'
    pdf.set_font('Malgun', 'B', 11)
    pdf.cell(0, 8, title_vi, border=0, ln=1)
    pdf.ln(1)
    if os.path.exists(temp_chart_path):
        pdf.image(temp_chart_path, x=22, y=pdf.get_y(), w=166, h=78)
        pdf.ln(80)
    else:
        pdf.ln(5)

    pdf.set_font('Malgun', 'B', 10)
    pdf.cell(0, 6, '  1. 시계열 흐름 분석 의견 :', border=0, ln=1)
    pdf.set_font('Malgun', '', 9)
    pdf.multi_cell(0, 5, '    가. 인공지능이 도출한 24시간 발전 추세선은 기상 변동 인자와 강한 상관성을 가지며 매끄러운 에너지 기조를 형성함.\n    나. 피크 아워 전후의 급경사 램핑 구간에서 계통 주파수 흔들림이 있을 수 있으니 제어 준비 바람.')

    # ==================== 3페이지 작성 (Ⅶ, Ⅷ, Ⅸ) ====================
    pdf.add_page()
    title_vii = '\u2166. \uc2dc\uacc4\uc5f4 \ud750\ub984 \uc138\ubd80 \ud574\uc11d \ubc0f \uacc4\ud1b5 \ubcc0\ub3d9\uc131(Ramping Rate) \ubd84\uc11d \uc62c\uacac'
    pdf.set_font('Malgun', 'B', 11)
    pdf.cell(0, 8, title_vii, border=0, ln=1)
    pdf.set_font('Malgun', '', 9)
    pdf.cell(0, 5.2, '  1. 발전 램핑률(Ramping Rate) 정량 분석 결과 :', ln=1)
    pdf.cell(0, 5.2, f'    가. 금일 발생 예상되는 최대 시간당 발전량 변동폭 : {max_ramping:.2f} MWh/hr', border=0, ln=1)
    pdf.cell(0, 5.2, f'    나. 최대 변동성 발생 타겟 시각 : {max_ramping_hour:02d}:00 전후 발생 판정', border=0, ln=1)
    pdf.multi_cell(0, 5.2, '  2. 계통 영향성 종합 의견 :\n    가. 순간 램핑률이 한계치(30MWh/hr) 미만으로 감지되어 계통 순간 동적 예비력은 안정 범위임.\n    나. 단, 급경사 램핑에 대응하기 위해 기동이 빠른 양수발전기 연계 제어 대기가 필수적임.')
    pdf.ln(5)

    title_viii = '\u2167. \uc804\ub825 \uacc4\ud1b5 \uc548\uc815\uc131 \uac80\ud1a0 \ubc0f \uc870\uce58 \uac00\uc774\ub4dc\ub7bc\uc778'
    pdf.set_font('Malgun', 'B', 11)
    pdf.cell(0, 8, title_viii, border=0, ln=1)
    pdf.set_font('Malgun', '', 9)
    pdf.cell(0, 5.5, f'  1. 계통 위험도 평가 결과 : [{risk_status}] {risk_desc}', border=0, ln=1)
    pdf.cell(0, 5.5, '  2. 실무 운영 조치 가이드 :', border=0, ln=1)
    pdf.multi_cell(0, 5.5, f'    가. {guide_a}\n    나. {guide_b}')
    pdf.ln(5)

    title_ix = '\u2168. \ud5a5\ud6c4 2\ub2e8\uacc4 \ucd94\uc9c4 \uacc4\ud68d \ubc0f \ucd08\uad6d\uc18c VPP \ucd5c\uc801\ud654 \ub85c\ub4dc\uba65'
    pdf.set_font('Malgun', 'B', 11)
    pdf.cell(0, 8, title_ix, border=0, ln=1)
    pdf.set_font('Malgun', '', 9)
    pdf.multi_cell(0, 5.5, '  1. 예측 모델 보정 및 재학습 일정 :\n    가. 실제 발전 실측치와 AI 예측 오차를 분석하여 피드백 오차 보정 학습을 진행함.\n    나. 제주 권역 내 4개 GPS 세부 단지의 실시간 수치 조정을 위한 앙상블 가중치 보정을 주간 단위로 실시함.\n  2. 가상발전소(VPP) 연계망 고도화 :\n    가. 예측 제고 정산금 획득 기준에 최적화되도록 분산 자원 실시간 수집 연계 소프트웨어를 정비할 예정임.')
    pdf.ln(8)
    
    pdf.set_font('Malgun', '', 8.5)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4.5, '※ 본 보고서는 전력 계통 사전 통제용 분석 결과로, 실제 전력거래소 실측 실적치와는 차이가 발생할 수 있으며 센서 결함 및 이변 환경에 따라 상시 오차가 감지될 수 있음을 고지함.')
    
    pdf.ln(2)
    pdf.set_font('Malgun', '', 9)
    pdf.cell(0, 5, '(끝)', border=0, ln=1, align='R')
        
    pdf_bytes = bytes(pdf.output())
    
    if os.path.exists(temp_chart_path):
        os.remove(temp_chart_path)
        
    return pdf_bytes

def verify_pages():
    for p in [3]: # We only verify 3 pages now
        try:
            pdf_data = simulate_pdf_generation(p)
            from io import BytesIO
            try:
                import pypdf
                reader = pypdf.PdfReader(BytesIO(pdf_data))
                actual_pages = len(reader.pages)
                print(f"Option: {p} pages -> Actual Generated PDF pages: {actual_pages}")
                assert actual_pages == p, f"Expected {p} pages, but got {actual_pages}!"
            except ImportError:
                print(f"Option: {p} pages -> PDF bytes generated successfully ({len(pdf_data)} bytes)")
        except Exception as e:
            print(f"Error during {p} pages simulation: {e}")
            sys.exit(1)
            
    print("All page configurations verified successfully!")

if __name__ == "__main__":
    verify_pages()
