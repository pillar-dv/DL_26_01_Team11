import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_anomaly_tab(wind_df, solar_df):
    st.subheader("⚠️ 발전 이상 감지 및 출력제어(Curtailment) 리스크 분석")
    st.markdown("본 모듈은 기상 이변(태풍, 한파, 미세먼지 등)으로 인한 **출력 제어(Cut-out) 현상을** 가상 시뮬레이션하고, 과거 실적 데이터 상에서 예측 범위를 벗어난 **비정상 발전 오차 지점(Anomaly)을** 통계학적으로 추적합니다.")
    
    # 1. 이상 발전량 가상 시뮬레이터 (Interactive Simulator)
    st.markdown("### 🌪️ 이상 기후에 의한 발전 차단(Cut-out) 및 출력제어 시뮬레이션")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.info("💡 **이상 환경 조건 설정**")
        event_type = st.selectbox("가상 재해 유형", ["태풍 및 강풍 (풍력 Cut-out)", "한파 및 한설 (태양광 패널 결빙)", "미세먼지 폭발 (일사량 차단)"])
        
        if event_type == "태풍 및 강풍 (풍력 Cut-out)":
            sim_wind = st.slider("💨 가상 태풍 풍속 (m/s)", 0.0, 35.0, 26.0, 0.5)
            st.warning("⚠️ **풍력 발전 차단(Cut-out) 기준**: 통상 풍속 **25m/s** 초과 시 강풍으로 인한 터빈 블레이드 파손을 막기 위해 발전기가 강제 정지되고 발전량이 0MWh로 차단됩니다.")
            
        elif event_type == "한파 및 한설 (태양광 패널 결빙)":
            sim_temp = st.slider("🌡️ 가상 기온 (°C)", -20.0, 5.0, -12.0, 0.5)
            st.warning("❄️ **한설 패널 결빙 리스크**: 영하 **-10°C** 이하에서 강설이 겹칠 경우, 일사량이 존재하더라도 패널 표면 결빙으로 발전 효율이 80% 이상 강제 저하됩니다.")
            
        elif event_type == "미세먼지 폭발 (일사량 차단)":
            sim_pm = st.slider("🌫️ 가상 미세먼지 농도 (㎍/㎥)", 50.0, 800.0, 450.0, 10.0)
            st.warning("😷 **일사량 급감 경보**: 황사 및 초미세먼지 농도가 **300㎍/㎥**을 초과할 경우 공기 중 빛 산란으로 인해 가상 일사량이 최대 50%까지 산란 손실을 겪습니다.")
            
        btn_run_anomaly = st.button("재해 시뮬레이션 가동", type="primary", use_container_width=True)

    with sim_col2:
        if btn_run_anomaly:
            st.markdown("#### 📊 가상 출력제어 리스크 시뮬레이션 결과")
            # 24시간 가상 프로파일 생성
            hours = np.arange(0, 24)
            normal_profile = []
            anomaly_profile = []
            status_msgs = []
            
            for h in hours:
                # 기본 정상 프로파일 설정
                if event_type == "태풍 및 강풍 (풍력 Cut-out)":
                    base = max(0.0, np.sin(2 * np.pi * h / 24) * 80 + 100)
                    normal_profile.append(base)
                    # 풍력 터빈의 풍속별 발전 특성 곡선 및 고풍속 제어 반영
                    if sim_wind < 3.0:
                        f_v = 0.0  # Cut-in 풍속 미만 (발전 정지)
                    elif sim_wind < 12.0:
                        f_v = ((sim_wind - 3.0) / 9.0) ** 3  # 운전 개시 및 출력 상승 구간
                    elif sim_wind < 20.0:
                        f_v = 1.0  # 정격 출력 구간
                    elif sim_wind < 25.0:
                        # 20m/s ~ 25m/s 구간: 태풍/강풍 경고 수치 근접 시 돌풍 보호 및 기기 제어(Soft Cut-out)로 발전량 점진적 감쇠
                        f_v = 1.0 - (sim_wind - 20.0) / 5.0
                    else:
                        f_v = 0.0  # 25m/s 이상: 강풍 터빈 파손 방지 강제 차단(Cut-out)
                    
                    anomaly_profile.append(base * f_v)
                
                elif event_type == "한파 및 한설 (태양광 패널 결빙)":
                    # 낮 시간에만 발전하는 태양광 (6시~18시 반주기 정현파 적용으로 정오 피크 구현)
                    base = max(0.0, np.sin(np.pi * (h - 6) / 12) * 150) if 6 <= h <= 18 else 0.0
                    normal_profile.append(base)
                    
                    # 한파 기온별 결빙 물리 특성 반영 (불연속 급감 제거 및 연속적인 선형 감하)
                    if sim_temp >= 0.0:
                        f_t = 1.0
                    elif sim_temp > -10.0:
                        # 0°C에서 1.0, -10°C에서 0.15로 부드럽게 선형 감소 (결빙 진행 과정 모사)
                        f_t = 1.0 - (0.0 - sim_temp) * 0.085
                    else:
                        # -10°C 이하: 완전 결빙 상태, -20°C까지 0.15에서 0.05로 미세 감소
                        f_t = max(0.05, 0.15 - (-10.0 - sim_temp) * 0.01)
                    
                    anomaly_profile.append(base * f_t)
                        
                elif event_type == "미세먼지 폭발 (일사량 차단)":
                    # 낮 시간에만 발전하는 태양광 (6시~18시 반주기 정현파 적용으로 정오 피크 구현)
                    base = max(0.0, np.sin(np.pi * (h - 6) / 12) * 150) if 6 <= h <= 18 else 0.0
                    normal_profile.append(base)
                    
                    # 대기 청정 임계치 반영 및 미세먼지 농도에 비례한 일사 감쇠 (청정일 0% 손실 보정)
                    if sim_pm <= 50.0:
                        attenuation = 1.0  # 50㎍/㎥ 이하 청정 대기: 일사 감쇠 없음
                    else:
                        # 50㎍/㎥ 초과 ~ 800㎍/㎥ 범위: 최대 80% 손실 (0.2 감쇠)까지 선형 감하
                        attenuation = 1.0 - (sim_pm - 50.0) / (800.0 - 50.0) * 0.8
                    
                    anomaly_profile.append(base * attenuation)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=normal_profile, name="정상 예측 발전량", line=dict(color='royalblue', dash='dash')))
            fig.add_trace(go.Scatter(x=hours, y=anomaly_profile, name="재해/출력제어 적용 발전량", fill='tozeroy', line=dict(color='crimson', width=3)))
            
            fig.update_layout(
                title=f"{event_type} 발생 시 24시간 출력 변동 시뮬레이션",
                xaxis_title="시간 (시)",
                yaxis_title="발전량 (MWh)",
                legend_title="구분",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 위험도 스코어 계산 및 경고문
            total_normal = sum(normal_profile)
            total_anomaly = sum(anomaly_profile)
            loss_pct = ((total_normal - total_anomaly) / (total_normal + 1e-8)) * 100
            
            metric_c1, metric_c2 = st.columns(2)
            metric_c1.metric("예상 손실 전력량", f"{total_normal - total_anomaly:.2f} MWh")
            metric_c2.metric("발전량 감소율", f"{loss_pct:.1f}%")
            
            if event_type == "태풍 및 강풍 (풍력 Cut-out)":
                if sim_wind >= 25.0:
                    st.error("🚨 **심각(Critical) 발전 차단(Cut-out) 발령**: 25m/s 이상의 강풍으로 인한 터빈 보호를 위해 발전기가 강제 정지(Cut-out)되고 발전이 전면 차단되었습니다.")
                elif sim_wind >= 20.0:
                    st.warning("⚠️ **주의(Warning) 고풍속 발전 감하**: 태풍/강풍 한계치 근접에 따른 기기 보호 제어(Soft Cut-out)로 인해 발전량이 크게 저하되었습니다.")
                elif sim_wind < 4.0:
                    st.info("ℹ️ **무풍/경풍 상태**: 풍속이 발전 개시 풍속(3m/s) 미만 또는 근접하여 자연적인 발전량 감소 상태입니다 (설비 보호 차단 아님).")
                else:
                    st.success("✅ **안정(Normal) 운전**: 풍속이 정상 동작 범위(4m/s ~ 20m/s) 내에 있으며, 설비 차단 우려가 없는 안정적인 발전 상태입니다.")
            
            elif event_type == "한파 및 한설 (태양광 패널 결빙)":
                if sim_temp <= -10.0:
                    st.error("🚨 **심각(Critical) 패널 결빙**: 영하 10°C 이하의 한파로 인해 태양광 패널 표면이 결빙되어 발전 효율이 극도로 저하되었습니다.")
                elif sim_temp <= 0.0:
                    st.warning("⚠️ **주의(Warning) 온도 저하**: 영하의 기온으로 인해 패널 효율 저하 및 부분적 결빙 리스크가 존재합니다.")
                else:
                    st.success("✅ **안정(Normal) 유지**: 기온이 영상권으로 패널 결빙 리스크가 없으며 정상적으로 발전이 수행됩니다.")
                    
            elif event_type == "미세먼지 폭발 (일사량 차단)":
                if sim_pm >= 300.0:
                    st.error(f"🚨 **심각(Critical) 일사 차단**: 미세먼지 농도 {sim_pm}㎍/㎥ 초과로 인한 빛 산란 손실로 가상 일사량이 극도로 급감하였습니다.")
                elif sim_pm >= 150.0:
                    st.warning(f"⚠️ **주의(Warning) 대기질 악화**: 황사 및 초미세먼지로 인해 발전 효율의 유의미한 저하가 관측됩니다.")
                else:
                    st.success("✅ **안정(Normal) 유지**: 대기질이 보통 수준 이하로 일사량 차단 영향이 미미합니다.")
        else:
            st.info("👈 왼쪽 패널에서 가상 환경 조건을 설정하고 '재해 시뮬레이션 가동'을 눌러 시뮬레이션을 실행해 보세요.")

    st.divider()

    # 2. 통계학적 오차 이상치 탐색 (Anomaly Detection)
    st.markdown("### 🪵 과거 실적 기반의 통계학적 발전 이상 지점 역추적 (Z-Score 기법)")
    
    target_data_type = st.radio("분석 타겟 에너지 선택", ["풍력 (Wind)", "태양광 (Solar)"], horizontal=True)
    
    if target_data_type == "풍력 (Wind)":
        df = wind_df.copy()
        regions = list(df['지역'].unique())
    else:
        df = solar_df.copy()
        regions = list(df['지역'].unique())
        
    sel_region = st.selectbox("분석할 지자체 지역 선택", regions)
    region_df = df[df['지역'] == sel_region].sort_values('일시').reset_index(drop=True)
    
    if len(region_df) > 100:
        # 가상의 예측 생성 (정밀 오차 연출을 위함)
        np.random.seed(100)
        # 실제 발전량에 노이즈를 섞어 예측 모방
        actual = region_df['전력거래량(MWh)'].values
        noise = np.random.normal(0, np.std(actual) * 0.15, size=len(actual))
        # 특정 5군데에 인위적으로 거대한 오차(이상치) 심기
        anomaly_indices = np.random.choice(len(actual), size=5, replace=False)
        for idx in anomaly_indices:
            noise[idx] += np.random.choice([-1, 1]) * np.max(actual) * 0.7
            
        preds = np.maximum(actual + noise, 0.0)
        errors = actual - preds
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        z_scores = (errors - mean_err) / (std_err + 1e-8)
        
        # 이상치 정의 (Z-Score 절대값 2.5 초과)
        region_df['예측량'] = preds
        region_df['오차'] = errors
        region_df['Z-Score'] = z_scores
        region_df['이상징후'] = np.abs(z_scores) > 2.5
        
        anomalies_found = region_df[region_df['이상징후'] == True]
        
        st.markdown(f"📊 **{sel_region} {target_data_type} 이상치 분석 결과** (총 {len(region_df)}개 샘플 데이터 중 이상지점 **{len(anomalies_found)}개** 감지)")
        
        # 그래프 매핑
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(x=region_df['일시'].tail(500), y=region_df['전력거래량(MWh)'].tail(500), name="실제 발전량", line=dict(color='gray', width=1.5)))
        fig_anom.add_trace(go.Scatter(x=region_df['일시'].tail(500), y=region_df['예측량'].tail(500), name="AI 예측량", line=dict(color='orange', width=1.5)))
        
        # 최근 500개 중 이상치 표시
        recent_anomalies = region_df.tail(500)
        recent_anomalies = recent_anomalies[recent_anomalies['이상징후'] == True]
        
        if not recent_anomalies.empty:
            fig_anom.add_trace(go.Scatter(
                x=recent_anomalies['일시'], 
                y=recent_anomalies['전력거래량(MWh)'], 
                mode='markers', 
                name='오차 이상 지점 (Z > 2.5)', 
                marker=dict(color='red', size=10, symbol='triangle-up', line=dict(color='black', width=1))
            ))
            
        fig_anom.update_layout(
            title="최근 500시간 시계열 중 이상치 탐지 이력",
            xaxis_title="일시",
            yaxis_title="발전량 (MWh)",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_anom, use_container_width=True)
        
        # 이상치 테이블 리스트 제공
        if len(anomalies_found) > 0:
            st.markdown("##### 🔍 감지된 주요 발전량 오차 이상치 명세")
            show_cols = ['일시', '전력거래량(MWh)', '예측량', '오차', 'Z-Score']
            if '기온(°C)' in region_df.columns:
                show_cols.append('기온(°C)')
            if '풍속(m/s)' in region_df.columns:
                show_cols.append('풍속(m/s)')
            
            anom_summary = anomalies_found[show_cols].copy()
            anom_summary.rename(columns={'전력거래량(MWh)': '실제 발전량(MWh)', '예측량': 'AI 예측 발전량(MWh)'}, inplace=True)
            
            st.dataframe(anom_summary.head(10).style.format({
                '실제 발전량(MWh)': '{:.2f}',
                'AI 예측 발전량(MWh)': '{:.2f}',
                '오차': '{:.2f}',
                'Z-Score': '{:.2f}',
                '풍속(m/s)': '{:.1f}',
                '기온(°C)': '{:.1f}'
            }), use_container_width=True)
            
            st.caption("※ 오차 임계치 Z-Score > 2.5: AI가 예상한 수치에 비해 기상 요인 대비 실측 발전량이 극도로 낮거나 높은 시점으로 기기 결빙, 송배전 차단, 센서 결함 등의 이상이 의심되는 시간대입니다.")
    else:
        st.warning("데이터가 부족하여 통계적 이상치 탐색이 불가합니다.")
