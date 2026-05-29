# Release Note: 신재생 에너지 발전량 예측 모델

## **v1.1 (데이터 전처리 구현 성공)**

### 주요 업데이트 및 신규 기능 (Major Updates)
* **태양광 및 풍력 데이터 파이프라인 완전 분리**
  * 태양광과 풍력의 발전 특성 및 입지 조건 차이를 반영하여, 전처리 및 통합 데이터셋 생성 로직을 완전히 분리 설계함. 
  * 최종 산출물이 `solar_integrated_dataset.csv`와 `wind_integrated_dataset.csv` 두 개의 독립적인 파일로 생성되도록 개선.
* **복수 관측소 기반 듀얼 매핑(Spatial Smoothing) 시스템 도입**
  * 광역 지자체 단위의 발전량에 대응하기 위해, 단일 기상 관측소가 아닌 지역 내 여러 관측소의 데이터를 리스트로 묶어 시간대별 평균을 산출하는 로직 추가.
  * 태양광(도심/내륙 위주)과 풍력(해안/고산 위주)의 매핑 딕셔너리를 분리하여 국지적 기상 이상에 따른 예측 노이즈(Noise)를 대폭 감소시킴.
* **학습 프레임워크 PyTorch 전면 마이그레이션**
  * PyTorch 기반의 클래스(`nn.Module`) 객체지향형 LSTM 모델로 아키텍처를 선택함.
  * `TensorDataset` 및 `DataLoader`를 도입하여 배치(Batch) 단위의 시계열 학습 효율성 향상.

### 버그 수정 및 안정화 (Bug Fixes)
* **데이터 분할 병합 시 `KeyError: '연료원'` 발생 현상 수정**
  * 데이터 피벗(Pivot) 작업 이후 사라진 '연료원' 컬럼을 참조하여 발생하던 에러 수정.
  * 태양광/풍력 컬럼을 명시적으로 분리한 뒤, 통일된 타겟 변수명인 `전력거래량(MWh)`으로 이름을 변경하도록 로직 개선.
* **함수 호출 시 `TypeError` 발생 현상 수정**
  * `build_integrated_dataset()` 함수가 요구하는 3번째 인자가 누락되던 문제 수정. 
  * 함수 실행부에 `solar_mapping`과 `wind_mapping` 딕셔너리가 정상적으로 주입되도록 수정.
* **`preprocess_time_series` 정의 누락 문제 해결**
  * 주피터 노트북 셀 분할로 인해 발생할 수 있는 함수 참조 에러 방지를 위해, `generator.py` (및 통합 셀) 최상단에 전처리 함수가 선언되도록 실행 순서 통합.

### 문서화 및 산출물 (Documentation)
* **GitHub Repository 업로드용 리소스 추가**
  * 프로젝트의 개요, 데이터셋 명세, 파이프라인 특징, 실행 방법을 명시한 `README.md` 작성 완료.
  * 용량 초과 및 민감 데이터 유출을 방지하기 위해 `*.csv` 및 데이터 폴더를 제외하는 `.gitignore` 설정 기준 확립.

## v1.2 (generator.py에 LSTM 모델 학습 기능을 통합하고, 학습된 모델로 즉시 예측 가능한 running.py를 신규 추가)

# 🌞💨 기상 데이터 기반 태양광·풍력 발전량 예측 모델

기상 데이터를 활용하여 태양광 및 풍력 발전량을 예측하는 LSTM 딥러닝 모델입니다.

---

## 📁 프로젝트 구조

```
├── generator.py              # 데이터 전처리 및 모델 학습
├── running.py                # 학습된 모델로 발전량 예측
├── dataset/                  # 데이터 폴더 (git 미포함)
│   ├── *.csv                 # 발전량·기상·미세먼지 데이터
│   ├── best_model.pth        # 태양광 최적 모델 가중치
│   ├── best_model_wind.pth   # 풍력 최적 모델 가중치
│   ├── scalers_X_solar.pkl   # 태양광 입력 스케일러
│   ├── scalers_y_solar.pkl   # 태양광 출력 스케일러
│   ├── scalers_X_wind.pkl    # 풍력 입력 스케일러
│   └── scalers_y_wind.pkl    # 풍력 출력 스케일러
└── README.md
```

---

## ⚙️ 설치

```bash
pip install torch pandas numpy scikit-learn matplotlib joblib
```

---

## 🚀 사용 방법

### 1. 학습 (`generator.py`)

`dataset/` 폴더에 CSV 파일을 배치한 후 실행

```bash
python generator.py
```

**출력 파일**
- `best_model.pth` — 태양광 최적 모델
- `best_model_wind.pth` — 풍력 최적 모델
- `scalers_X_solar.pkl`, `scalers_y_solar.pkl` — 태양광 스케일러
- `scalers_X_wind.pkl`, `scalers_y_wind.pkl` — 풍력 스케일러
- `solar_integrated_dataset.csv` — 태양광 통합 데이터셋
- `wind_integrated_dataset.csv` — 풍력 통합 데이터셋
- `lstm_result_solar_all.png` — 태양광 전체 지역 결과 그래프
- `lstm_result_wind_all.png` — 풍력 전체 지역 결과 그래프

### 2. 예측 (`running.py`)

학습 완료 후 실행

```bash
python running.py
```

예측 지역은 `running.py` 하단에서 변경 가능

```python
target_region_solar = '제주도'   # 태양광 예측 지역
target_region_wind  = '전라북도' # 풍력 예측 지역
```

---

## 🧠 모델 구조

| 항목 | 내용 |
|---|---|
| 모델 | LSTM (2층) |
| 옵티마이저 | Adam |
| 손실함수 | MSE |
| 학습률 스케줄러 | ReduceLROnPlateau |
| Early Stopping | patience=20 |
| 입력 시퀀스 | 24시간 |
| 출력 | 다음 1시간 발전량 (MWh) |

---

## 📊 입력 Feature

**태양광 (7개)**
```
기온(°C), 풍속(m/s), 습도(%), 미세먼지농도, 시간, 월, 일사(MJ/m2)
```

**풍력 (9개)**
```
기온(°C), 풍속(m/s), 풍속_세제곱, 풍향(16방위), 습도(%),
현지기압(hPa), 전운량(10분위), 시간, 월
```

---

## 📈 업데이트 내역

> 데이터 전처리 전용이었던 `generator.py`에 LSTM 모델 학습 기능을 통합하고, 학습된 모델로 즉시 예측 가능한 `running.py`를 신규 추가

### generator.py

| 항목 | 이전 | 현재 |
|---|---|---|
| 역할 | 데이터 전처리만 수행 | 전처리 + LSTM 모델 학습 통합 |
| 스케일링 | 전체 데이터에 fit | train 70%에만 fit (데이터 누수 방지) |
| 시퀀스 생성 | 전체 합친 뒤 생성 | 지역별 생성 후 합치기 (경계 오염 제거) |
| 데이터 분리 | train/test 2분할 | train/val/test 3분할 |
| 스케일러 저장 | 저장 없음 | pickle로 저장 |
| 일사량 처리 | 선형 보간 (야간 오류) | `fillna(0)` (야간 일사량 0 처리) |
| 풍력 feature | 태양광과 동일 | 풍향·기압·전운량·풍속세제곱 추가 |
| 학습률 스케줄러 | CosineAnnealingLR | ReduceLROnPlateau |
| 시간 변환 | 거래시간 그대로 사용 | 거래시간 -1 보정 |
| 경로 설정 | 윈도우 전용 (`\\`) | `os.path.join` (클라우드 호환) |
| 시각화 | 전체 처음 300개 | 지역별 개별 시각화 |

### running.py (신규 추가)

학습된 모델로 특정 지역의 발전량을 즉시 예측하는 추론 전용 파일

| 항목 | 내용 |
|---|---|
| 역할 | 저장된 모델·스케일러 로드 후 예측 실행 |
| 스케일러 | pickle 로드 후 `transform`만 수행 (재학습 없음) |
| 입력 | 지역별 최근 24시간 기상 데이터 |
| 출력 | 태양광·풍력 발전량 예측값 (MWh) 및 합계 |
| 경로 | `os.path.join` 사용 (클라우드 호환) |

---

## 🗂️ 데이터 출처

- 태양광·풍력 발전량: 한국전력거래소 공공데이터포털
- 기상 데이터: 기상자료개방포털
- 미세먼지: 환경부 에어코리아
- 풍력기 위치정보: 한국에너지공단

- ## v1.3(태양광 LSTM 지역별 학습 + 풍력 XGBoost 적용)**

> 태양광·풍력 모두 지역별 독립 모델로 전환하고, 풍력은 LSTM에서 XGBoost로 교체

### generator.py

| 항목 | 이전 | 현재 |
|---|---|---|
| 태양광 모델 | 단일 LSTM (전체 지역 통합) | **지역별 독립 LSTM** |
| 풍력 모델 | 단일 LSTM | **지역별 독립 XGBoost** |
| Loss 곡선 | 모든 지역 동일 ❌ | 지역별 각각 다름 ✅ |
| 스케일링 | 전체 데이터 fit | train 70%만 fit (누수 방지) |
| 풍력 피처 | 기상 9개 | 기상 + 래그 + 이동평균 19개 |

### running.py (신규 추가)

| 항목 | 내용 |
|---|---|
| 태양광 | LSTM 지역별 모델 로드 후 예측 |
| 풍력 | XGBoost 지역별 모델 로드 후 예측 |
| 스케일러 | pickle 로드 후 `transform`만 수행 (재학습 없음) |
| 야간 처리 | 태양광 6~19시 외 자동 0 MWh 처리 |
| 출력 | 태양광·풍력 발전량 및 합계 출력 |

---

## 📊 모델 성능

**태양광 LSTM (전체)**

| MAE | RMSE | R² | MAPE |
|---|---|---|---|
| 19.95 MWh | 61.66 MWh | 0.9146 | 25.15% |

**풍력 XGBoost (지역별)**

| 지역 | R² | MAE |
|---|---|---|
| 강원도 | 0.889 | 27.14 MWh |
| 경상북도 | 0.791 | 35.80 MWh |
| 전라북도 | 0.903 | 4.64 MWh |
| 육지 | 0.981 | 29.94 MWh |
