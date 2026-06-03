import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'metadata')
STATS_FILE = os.path.join(METADATA_PATH, 'weather_statistics.pkl')

def build_weather_statistics():
    """
    최초 1회 실행하여 태양광/풍력 데이터셋으로부터 지역별, 월별, 시간별 기상 통계값(mean, std)을 추출 및 저장합니다.
    """
    print("기상 통계 분석 데이터 빌드 중...")
    
    # 1. 태양광 기상 데이터 분석
    solar_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'solar_integrated_dataset.csv'), encoding='utf-8-sig')
    solar_df['일시'] = pd.to_datetime(solar_df['일시'])
    solar_df['월'] = solar_df['일시'].dt.month
    solar_df['시간'] = solar_df['일시'].dt.hour
    
    solar_features = ['기온(°C)', '풍속(m/s)', '습도(%)', '미세먼지농도', '일사(MJ/m2)']
    # 지역별, 월별, 시간별 groupby 평균 및 표준편차
    solar_grouped = solar_df.groupby(['지역', '월', '시간'])[solar_features].agg(['mean', 'std']).reset_index()
    
    # 2. 풍력 기상 데이터 분석
    wind_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'wind_integrated_dataset.csv'), encoding='utf-8-sig')
    wind_df['일시'] = pd.to_datetime(wind_df['일시'])
    wind_df['월'] = wind_df['일시'].dt.month
    wind_df['시간'] = wind_df['일시'].dt.hour
    
    wind_features = ['기온(°C)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '전운량(10분위)']
    wind_grouped = wind_df.groupby(['지역', '월', '시간'])[wind_features].agg(['mean', 'std']).reset_index()
    
    # dict 형태로 정리
    stats = {
        'solar': solar_grouped,
        'wind': wind_grouped
    }
    
    os.makedirs(METADATA_PATH, exist_ok=True)
    joblib.dump(stats, STATS_FILE)
    print(f"기상 통계 분석 데이터 빌드 완료: {STATS_FILE}")
    return stats

def get_stats():
    if not os.path.exists(STATS_FILE):
        return build_weather_statistics()
    return joblib.load(STATS_FILE)

def generate_stochastic_weather(fuel_type, region, target_month):
    """
    몬테카를로 및 마르코프 상관관계를 활용하여 realistic한 24시간 가상 날씨 시퀀스를 생성합니다.
    """
    stats = get_stats()
    df_stats = stats[fuel_type]
    
    # 해당 지역 및 월 필터링
    df_sub = df_stats[(df_stats['지역'] == region) & (df_stats['월'] == target_month)].copy()
    
    if df_sub.empty:
        # 매칭 지역/월이 없을 경우 전체 평균값 폴백
        df_sub = df_stats[df_stats['월'] == target_month].copy()
        if df_sub.empty:
            df_sub = df_stats.copy()
            
    # 시간 순 정렬
    df_sub = df_sub.sort_values('시간').reset_index(drop=True)
    
    hours = np.arange(0, 24)
    simulated_data = {
        '시간': hours,
        '지역': [region] * 24
    }
    
    features = ['기온(°C)', '풍속(m/s)', '습도(%)']
    if fuel_type == 'solar':
        features += ['미세먼지농도', '일사(MJ/m2)']
    else:
        features += ['풍향(16방위)', '현지기압(hPa)', '전운량(10분위)']
        
    for feat in features:
        means = []
        stds = []
        for h in hours:
            row = df_sub[df_sub['시간'] == h]
            if not row.empty:
                m = row[(feat, 'mean')].values[0]
                s = row[(feat, 'std')].values[0]
                # 결측 nan 보정
                m = m if not np.isnan(m) else 0.0
                s = s if not np.isnan(s) else 1.0
            else:
                m, s = 0.0, 1.0
            means.append(m)
            stds.append(s)
            
        # 마르코프 체인 노이즈 생성 (인접 시간대 간의 강한 시계열 상관성 부여)
        raw_noise = np.random.normal(0, 1, size=24)
        correlated_noise = np.zeros(24)
        correlated_noise[0] = raw_noise[0]
        # AR(1) 모형 계수 0.85 적용 (인접 시간대 기온/바람의 급변 억제)
        for i in range(1, 24):
            correlated_noise[i] = 0.85 * correlated_noise[i-1] + 0.5 * raw_noise[i]
            
        # 변수별 물리적 특성에 따른 바인딩 및 역투사
        sim_values = []
        for i in range(24):
            val = means[i] + stds[i] * correlated_noise[i]
            
            # 물리적 바운딩 처리
            if feat == '일사(MJ/m2)':
                # 밤 시간대(19시 ~ 익일 5시) 일사는 무조건 0
                if i < 6 or i > 18:
                    val = 0.0
                else:
                    val = max(0.0, val)
            elif feat in ['풍속(m/s)', '미세먼지농도', '전운량(10분위)', '습도(%)']:
                val = max(0.0, val)
                if feat == '습도(%)':
                    val = min(100.0, val)
                elif feat == '전운량(10분위)':
                    val = min(10.0, val)
            elif feat == '풍향(16방위)':
                val = max(0.0, min(360.0, val))
                
            sim_values.append(val)
            
        simulated_data[feat] = sim_values
        
    sim_df = pd.DataFrame(simulated_data)
    return sim_df

if __name__ == '__main__':
    # 테스트 구동
    print("Stochastic weather module test...")
    st = generate_stochastic_weather('solar', '경기도', 5)
    print(st.head(10))
