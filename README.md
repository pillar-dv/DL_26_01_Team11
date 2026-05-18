v1.1 (최신 업데이트)
🚀 주요 업데이트 및 신규 기능 (Major Updates)
태양광 및 풍력 데이터 파이프라인 완전 분리

태양광과 풍력의 발전 특성 및 입지 조건 차이를 반영하여, 전처리 및 통합 데이터셋 생성 로직을 완전히 분리 설계함.

최종 산출물이 solar_integrated_dataset.csv와 wind_integrated_dataset.csv 두 개의 독립적인 파일로 생성되도록 개선.

복수 관측소 기반 듀얼 매핑(Spatial Smoothing) 시스템 도입

광역 지자체 단위의 발전량에 대응하기 위해, 단일 기상 관측소가 아닌 지역 내 여러 관측소의 데이터를 리스트로 묶어 시간대별 평균을 산출하는 로직 추가.

태양광(도심/내륙 위주)과 풍력(해안/고산 위주)의 매핑 딕셔너리를 분리하여 국지적 기상 이상에 따른 예측 노이즈(Noise)를 대폭 감소시킴.

학습 프레임워크 PyTorch 전면 마이그레이션

기존 Keras(TensorFlow) 기반 모델에서 PyTorch 기반의 클래스(nn.Module) 객체지향형 LSTM 모델로 아키텍처를 전면 개편함.

TensorDataset 및 DataLoader를 도입하여 배치(Batch) 단위의 시계열 학습 효율성 향상.

범용 GPU 가속 지원 (AMD 라데온 포함)

하드웨어 환경에 구애받지 않도록 동적 디바이스 할당 로직 추가.

NVIDIA(CUDA), Apple Silicon(MPS) 지원은 물론, torch-directml 패키지를 활용한 Windows 환경의 AMD 라데온 GPU 가속 기능을 추가함.

🛠 버그 수정 및 안정화 (Bug Fixes)
데이터 분할 병합 시 KeyError: '연료원' 발생 현상 수정

데이터 피벗(Pivot) 작업 이후 사라진 '연료원' 컬럼을 참조하여 발생하던 에러 수정.

태양광/풍력 컬럼을 명시적으로 분리한 뒤, 통일된 타겟 변수명인 전력거래량(MWh)으로 이름을 변경하도록 로직 개선.

함수 호출 시 TypeError 발생 현상 수정

build_integrated_dataset() 함수가 요구하는 3번째 인자가 누락되던 문제 수정.

함수 실행부에 solar_mapping과 wind_mapping 딕셔너리가 정상적으로 주입되도록 수정.

preprocess_time_series 정의 누락 문제 해결

주피터 노트북 셀 분할로 인해 발생할 수 있는 함수 참조 에러 방지를 위해, generator.py (및 통합 셀) 최상단에 전처리 함수가 선언되도록 실행 순서 통합.

VS Code 및 터미널 환경 실행 충돌 가이드 추가

윈도우 '앱 실행 별칭(App execution aliases)' 충돌로 인해 터미널에서 스크립트 실행이 차단되는 문제 원인 분석 및 해결 프로세스 확립.

📝 문서화 및 산출물 (Documentation)
GitHub Repository 업로드용 리소스 추가

프로젝트의 개요, 데이터셋 명세, 파이프라인 특징, 실행 방법을 명시한 README.md 작성 완료.

용량 초과 및 민감 데이터 유출을 방지하기 위해 *.csv 및 데이터 폴더를 제외하는 .gitignore 설정 기준 확립.

이 패치노트 역시 README.md의 하단에 ## 7. Release Notes 항목으로 추가해 두시면, 프로젝트의 발전 과정(History)을 증명하는 아주 좋은 포트폴리오 자료가 될 것입니다. 팀원들에게 전달하실 때 함께 활용해 보세요.