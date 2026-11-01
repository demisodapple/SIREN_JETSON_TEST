# SIREN JETSON TEST

Jetson Nano에서 **PD·BUS·압축 예측기 실행 → DNN 연산량·시간 측정 → GRU와 ModernTCN 압축 비교 → 전력 로그 결합**을 진행한다. 모든 명령은 `run.py`에서 실행한다.

**원자료 없이 세 DNN 실행과 OPS 측정하기**

`dummy_data/`에 PD·BUS·압축용 ModernTCN 입력을 포함했다. 원자료 다운로드나 전처리 없이 바로 실행한다. 임무는 세 개이며 GRU는 3번 임무의 추가 비교용이다. 기본 `trial`의 세 모델은 PD + BUS + ModernTCN이다.

| 파일 이름 (`dummy_data/` 안) | 형식 | 모양 |
|---|---|---|
| `model1_PD_float.npz` | FP32 이식 경로 입력 | `[4,310,28]` |
| `model2_BUS_float.npz` | FP32 이식 경로 입력 | `[4,64,12]` |
| `model3_COMP_ModernTCN_float.npz` | FP32 이식 경로 입력 | `[4,310,24]`, 마지막 5채널 0 |
| `model1_PD_reference.npz` | 정수 참조 경로 입력 | `[4,310,32]`, 마지막 4채널 0 |
| `model2_BUS_reference.npz` | 정수 참조 경로 입력 | `[4,64,12]` |
| `model3_COMP_ModernTCN_reference.npz` | 정수 참조 경로 입력 | `[4,310,24]` |
| `model3_COMP_GRU_float.npz` | 비교용 GRU 입력 | `[4,1,34]` |

배열은 NumPy NPZ이며 `x`와 모델·가중치 해시·생성 조건이 들어 있다. seed=20261102, 창당 0.6×사인파와 [-0.25,0.25] 균등 잡음을 합친 입력이며 실제 위성 자료가 아니다. FP32/정수 입력은 같은 격자를 사용한다. 파일 SHA-256과 프레임당 연산량은 `dummy_data/manifest.json`에 있다. 결과에는 `synthetic: true`, `calibration_mode: synthetic_dummy`가 남는다.

Jetson SSH 터미널에서 먼저 환경을 확인한다. 기존 설치 패키지는 이 단계에서 재설치하지 않는다.

```bash
cd ~/SIREN_JETSON_TEST
git pull
python3 -c "import torch, numpy; print('PyTorch:',torch.__version__); print('NumPy:',numpy.__version__); print('CUDA:',torch.cuda.is_available())"
```

CPU에서 세 모델을 각각 짧게 실행:

```bash
python3 run.py trial --device cpu --phase single --duration 1 --repeat 1 --warmup 1 --output results/dummy_check01
```

이후 CUDA가 True인 호환 NVIDIA PyTorch 환경에서 GPU 측정:

```bash
python3 run.py trial --device cuda --phase all --duration 60 --repeat 3 --output results/dummy_gpu01
```

각 모델을 60초 이상씩 3회 측정하고, 세 모델 동시 실행을 60초 이상씩 3회 측정한다. 순차 9분 + 동시 3분 = 최소 약 12분에 로딩·예비 실행 시간이 더해진다. 작업 1회 시간이 설정 구간보다 길면 완료까지 기다린다. `--phase single` 또는 `--phase together`로 각 단계를 별도 실행할 수 있다. 다른 시험에는 새 결과 폴더 이름을 사용한다. 실행할 때마다 작업 프로세스와 로그를 만들며 실패 시 다음 단계로 넘어가지 않는다. CPU 스레드는 프로세스당 기본 1개다.

`trial` 기본 backend는 **FP32 이식 경로(float)**다. `--backend reference`는 공개 정수 계산을 INT64/FP64로 재현하는 검증 경로이며 네이티브 INT8 GPU 성능이 아니다. 두 결과를 같은 실행 방식으로 혼동하지 않는다. 실제 Nano GPU 호환성·속도는 보드에서 확인해야 한다. CPU 검증 통과를 Nano 실측 성공으로 표시하지 않는다.

| 결과 | 내용 |
|---|---|
| `pd.json`, `bus.json`, `comp.json` | 단독 실행의 MACs·operations/프레임, 연산/초, GOPS, 지연, 입력 해시와 측정 시각 |
| `together_01/pd.json` 등 | 동시 실행할 때 각 DNN의 지연·처리량 |
| `together_01/system.json` | 세 DNN이 모두 실행된 공통 시간 구간. 전력 로그를 이 구간에 연결 |
| `summary.json` | 생성된 결과 파일 및 측정 구간 요약 |
| 모델별 `.log` | 로딩·예비 실행·실행 실패 등 확인 |

측정은 준비된 입력 → 신경망 출력까지이며 학습·원자료 전처리·잔차 생성·BZ2 압축은 포함하지 않는다. 모델 로딩과 예비 실행도 제외한다. 기본 연산량은 **행렬·합성곱 1 MAC=2 operations** 기준이다. LayerNorm·GELU·잔차 덧셈 등 누락 항목을 결과에 명시하고, 지연은 신경망 전체 호출을 측정한다. 동시 GPU 실행에서는 작업 간 경합과 동기화 대기도 시간에 포함된다.

전력은 외부 측정기 기록을 먼저 시작하고 위 명령을 실행한다. `--tegrastats`를 추가하면 상태 로그도 저장하지만 외부 보드 입력 전력 기록을 대체하지 않는다. 계측기의 모델별 취득 드라이버는 현재 연결하지 않았다. 외부 CSV는 `time_s`(Jetson과 같은 Unix epoch 초), `power_w`(보드 입력 W) 열이 필요하다.

```bash
python3 run.py power --input inputs/power.csv --result results/dummy_gpu01/pd.json --output results/dummy_gpu01/pd_power.json
python3 run.py power --input inputs/power.csv --result results/dummy_gpu01/together_01/system.json --output results/dummy_gpu01/together_01/system_power.json
```

평균 W, 관측 최대 W, 에너지 J와 실제 표본 간격을 계산한다. 100 Hz는 10 ms 간격에서 **관측된 최대 전력**이며 모든 순간 피크를 포착한다는 의미가 아니다. 전체 장치·모델 조합에서 가능한 절대 최대 전력도 보장하지 않는다. 개별 모델 최대 전력을 합산해서 동시 실행 피크로 쓰지 않는다. 정식 데이터의 정확도·운용 전력은 별도 검증 대상이다.

새 더미 입력을 만들려면 기존 prepare 명령을 사용한다.

```bash
python3 run.py prepare --kind dummy --model pd --backend float --frames 4 --seed 20261102 --output inputs/another_dummy.npz
```

**임의 통계·보정값으로 바로 DNN 부하·전력 측정하기**

`demo_calibration.json`을 함께 제공한다. 아래 명령에서 `--demo-calibration demo_calibration.json`을 명시하면 외부 학습 통계·배경표 없이 실제 원자료에서 입력을 만든다. 신경망 가중치는 기존 것을 그대로 사용한다. 준비 단계는 측정 전에 수행하며 DNN 측정 시간에 포함되지 않는다.

| 대상 | 준비할 원자료 | 현재 확인한 예시 |
|---|---|---|
| BUS | EIRSAT-1 Dataset.zip의 Flight Test Data/flighttest_dataset.csv | 실제 CSV의 OBT(초), Channel=15, 지정된 12개 전류·전압·상태 열 |
| PD | NOAA POES/MetOp SEM-2 **L1b** 하루치 netCDF | MetOp-01의 `poes_m01_20220101_proc.nc` |
| 압축 비교도 할 경우 | MetOp SEM-2 **L1a** 하루치 netCDF | `poes_m01_20220101_raw.nc`. PD용 proc 파일과 구분 |

공개 자료 출처: [EIRSAT-1 Zenodo](https://zenodo.org/records/15256385), [NOAA MetOp-01 L1b 2022](https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l1b/v01r00/2022/metop01/). EIRSAT ZIP의 실제 이름은 `EIRSAT-1 Dataset.zip`이다. TVAC CSV도 같은 전처리 명령을 사용할 수 있다. 두 캠페인을 하나의 연속 시계열로 이어 붙이지 않는다.

Jetson의 SSH 터미널에서:

```bash
cd ~/SIREN_JETSON_TEST
git pull
mkdir -p inputs results
wget -O inputs/EIRSAT.zip 'https://zenodo.org/api/records/15256385/files/EIRSAT-1%20Dataset.zip/content'
python3 -m zipfile -e inputs/EIRSAT.zip inputs
wget -O inputs/poes_m01_20220101_proc.nc 'https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l1b/v01r00/2022/metop01/poes_m01_20220101_proc.nc'
```

이미 파일을 갖고 있다면 다운로드를 생략하고 실제 경로를 `--input`에 넣는다. 호환 PyTorch와 `requirements.txt`의 패키지가 설치된 환경에서 입력을 준비한다.

```bash
python3 run.py prepare --kind bus --model bus --demo-calibration demo_calibration.json --time-column OBT --input 'inputs/EIRSAT-1 Dataset/Flight Test Data/flighttest_dataset.csv' --output inputs/model2_BUS_demo.npz
python3 run.py prepare --kind pd --model pd --demo-calibration demo_calibration.json --input inputs/poes_m01_20220101_proc.nc --output inputs/model1_PD_demo.npz
```

CPU에서 작은 실행부터 확인하고, 외부 계측기를 기록 상태로 둔 채 DNN을 반복 실행한다.

```bash
python3 run.py bench --model bus --input inputs/model2_BUS_demo.npz --duration 60 --repeat 3 --output results/model2_BUS_demo.json
python3 run.py bench --model pd --input inputs/model1_PD_demo.npz --duration 60 --repeat 3 --output results/model1_PD_demo.json
```

GPU로 **FP32 이식 경로**를 측정하려면 입력 준비 두 명령에도 `--backend float`를 넣고 다른 출력 이름을 사용한다. 측정 명령에는 `--backend float --device cuda`를 함께 넣는다. 기본 `reference` 경로는 INT64/FP64 정수 계산 재현이며 네이티브 INT8 GPU 성능이 아니다. 각 실행 방식의 상세 조건은 아래 표를 참고한다.

임의값의 의미:

| 항목 | 제공하는 값·처리 |
|---|---|
| BUS 표준화 | 지정된 12채널 순서, 평균 0, 임의 고정 분모: CMC 전류 4096, 전압·버스 전류·스위치 비트맵 1024, 배터리 전류 방향 1. 실제 학습 표준편차가 아니라 원자료 코드값의 과도한 포화를 줄이는 부하 시험용 값 |
| PD | eps=1, 평균 0, 표준편차 1, 잔차 scale=1, 직전 27일 배경 log flux=0, 초기 문맥 코드=0 |
| PD 지리·품질 정보 | 실제 L1b에 있으면 자기위도·L·MLT·IFC 사용. 없는 항목은 JSON의 상수 사용, 사용한 원자료 열·대체 항목 기록 |
| 결과 표시 | `input.calibration_mode = arbitrary_demo`, 보정 JSON SHA-256, 포화 비율, 대체 항목. 탐지 정확도는 검증하지 않음 |

이 측정은 **실제 관측값과 임의 보정으로 만든 입력에서의 DNN 부하·보드 전력**이다. 정식 전처리의 탐지 성능이나 대표 운용 전력을 검증한 결과로 해석하지 않는다. 배경·초기 문맥은 합성 값이고, dtype 범위를 벗어나는 입력은 클리핑되며 그 비율을 기록한다. BUS 모델 내부의 학습된 가중치는 바꾸지 않는다.

외부 전력 CSV를 확보하면 아래처럼 측정 시각과 결합한다. `tegrastats`의 상태 기록만으로 외부 계측이 완료되는 것은 아니다. CSV 시간은 Jetson 측정 결과와 같은 Unix 시각 기준이어야 한다.

```bash
python3 run.py power --input inputs/power.csv --result results/model1_PD_demo.json --output results/model1_PD_demo_power.json
```

원자료 없이 부하 실행만 확인하려면 기존 `prepare --kind smoke`도 사용할 수 있다. 아래의 외부 통계·상태 파일 요구사항은 **정식 전처리 경로**에 해당하며 위 데모 모드는 이를 명시적으로 대체한다.

| 파일 | 역할 |
|---|---|
| `run.py` | 환경 확인, 전처리, 연산량 계산, 성능 측정, 압축·복원, 세 모델 동시 실행, 전력 로그 결합 |
| `models.py` | 네 가중치 로딩, 신경망 구조, CPU/GPU 계산, 압축용 예측기 |
| `preprocess.py` | PD·BUS·압축 입력 준비와 입력 파일 검증 |
| `compression.py` | 공통 raw·delta·신경망 잔차 BZ2 압축 및 정확한 복원 |
| `reference.py` | 원본의 NumPy 정수 계산 기준 구현. 이식한 실행 결과 검증에 사용 |
| `test.py` | 실제 가중치 계산 대조, 압축 복원, 입력 검사, 전력 적분 테스트 |
| `weights/` | 모델별 가중치와 실행 설정을 합친 파일 네 개 |
| `SOURCES.json` | 원본 커밋·파일 경로·SHA-256 및 재구성 파일 SHA-256 |

`inputs/`, `results/`는 명령 실행 중 필요할 때 만든다. 원자료·측정 로그·중간 산출물·FPGA 빌드 파일은 Git에 넣지 않는다.

**모델 이름과 실행 방식**

| 선택 이름 | 파일 | 용도 |
|---|---|---|
| `pd` | `weights/model1_PD_weight.pth` | 입자 환경 위험 탐지 |
| `bus` | `weights/model2_BUS_weight.pth` | 위성 하우스키핑 이상 탐지 |
| `comp` | `weights/model3_COMP_ModernTCN_weight.pth` | ModernTCN 압축 예측기 |
| `gru` | `weights/model3_COMP_GRU_weight.pth` | 같은 압축 임무의 비교용 GRU |

임무는 세 개이며, 3번 임무에 비교용 예측기 두 종류가 있다. `.pth` 파일에는 원본 체크포인트와 필요한 정수 계산 계약을 함께 넣었다. 단순 확장자 변경이나 새 학습이 아니다. 원본 가중치·정규화 값은 보존하고, BUS의 불필요한 학습 PC 경로만 제거했다. 계약은 기본 모델판이며 FPGA 전용 coarse-weight 변형판은 포함하지 않았다.

| 옵션 | 의미 | 결과 해석 |
|---|---|---|
| `--backend reference` 기본 | 공개된 정수 계약을 INT64 및 정확한 FP64 부분곱으로 재현 | 정수 모델 검증·기준 실행. **Jetson의 네이티브 INT8 가속 성능이 아님** |
| `--backend float` | 체크포인트 파라미터를 사용하는 FP32 이식 실행 | QAT 양자화 계산은 제외. 원본 모델 정의가 외부 파일이어서 내보내기 코드와 텐서 구조에 근거해 복원한 실행 경로. 실제 기준 출력·임무 정확도 검증 전에는 별도 실험 결과로 표시 |
| `--model gru` | 원본 GRU와 같은 FP32 계산·상태 갱신 | GRU에는 `reference`를 지정해도 실제 backend는 `float`로 기록 |

가중치가 8 bit로 양자화됐다는 이유로 실행을 INT8 가속이라고 부르지 않는다. 결과 JSON에 실제 실행 방식을 기록한다. `reference`의 검증·동기화 비용도 실행 시간에 포함된다. GPU 실행은 `--device cuda`, CPU 실행은 `--device cpu`다. `--threads 1`은 CPU 계산 스레드 수이며 다른 코어 전원을 끄는 설정은 아니다.

**1. Jetson에서 처음 확인**

Windows는 그대로 사용해도 된다. 아래 명령은 Jetson 터미널 또는 Jetson에 SSH로 접속한 터미널에서 실행한다.

```bash
git clone https://github.com/demisodapple/SIREN_JETSON_TEST.git
cd SIREN_JETSON_TEST
python3 run.py check
```

`results/environment.json`에 운영체제·Jetson Linux·Python·PyTorch·CUDA·메모리·전력 모드 조회 결과가 저장된다. 설치되지 않은 라이브러리와 명령은 표시된다. 이 명령은 시스템 설정을 바꾸지 않는다. 권한 때문에 조회되지 않는 항목은 다음으로 별도 확인한다.

```bash
sudo nvpmodel -q --verbose
sudo jetson_clocks --show
```

Python 3.6 이상 문법과 PyTorch 1.10 계열 API를 대상으로 작성했다. 로컬 검증 환경은 Python 3.12 / PyTorch 2.14 CPU이며 **Nano 및 구버전 환경 실측은 아직 필요**하다. 설치된 JetPack·Python에 맞는 NVIDIA PyTorch를 먼저 선택한다. 일반 PC용 최신 torch나 Orin용 JetPack을 Nano에 무조건 설치하지 않는다. [NVIDIA PyTorch 설치 안내](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)

호환되는 PyTorch 환경을 마련한 뒤:

```bash
python3 -m pip install -r requirements.txt
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -m unittest test -v
```

**2. 합성 입력으로 로딩부터 확인**

다음 입력은 0으로 채운 기능 점검용이다. 전력·압축 성능의 대표 데이터가 아니다. 결과에도 `synthetic: true`가 남는다.

```bash
python3 run.py prepare --kind smoke --model bus --output inputs/model2_BUS_smoke.npz
python3 run.py bench --model bus --input inputs/model2_BUS_smoke.npz --duration 1 --repeat 1 --warmup 1 --output results/model2_BUS_smoke.json
```

`pd`, `comp`, `gru`도 같은 방식으로 확인한다. `--backend float`를 사용할 때는 입력 준비와 실행에 **둘 다** 같은 옵션을 넣는다. 기존 파일은 덮어쓰지 않으므로 반복할 때 결과 이름을 바꾼다.

연산량만 계산하려면:

```bash
python3 run.py ops --model pd --output results/model1_PD_ops.json
python3 run.py ops --model bus --output results/model2_BUS_ops.json
python3 run.py ops --model comp --output results/model3_COMP_ops.json
python3 run.py ops --model gru --output results/model3_GRU_ops.json
```

MAC는 곱셈 후 누적 덧셈이다. **1 MAC = 2 operations**로 계산한다. 편향·활성화·정규화·잔차 덧셈 등 제외 항목은 결과에 적는다. GOPS는 이 연산량을 실제 실행 시간과 10억으로 나눈 값이다. 모든 소프트웨어 명령 수를 세는 지표는 아니다.

**3. 실제 입력 준비**

| 대상 | 현재 지원하는 입력 경로 | 추가로 필요한 자료 |
|---|---|---|
| 전 모델 | 이미 준비된 `.npy`, `x`가 있는 `.npz`, 명시적 자료형의 이진 텐서 → 측정용 NPZ | 모델·정수/실수 입력 규칙에 맞는 텐서, 가능하면 기준 출력 |
| GRU·Comp | MetOp L1a netCDF 또는 계수 NPZ → 정규화·문맥 창 | 해당 28/13 계수 채널과 위치·궤도 보조자료. 정규화 값은 가중치에 포함 |
| BUS | 하우스키핑 CSV → 50초 보간 → 표준화 → 64시점 창, 보폭 16 | 학습 시의 `channels`, `mean`, `std` 통계 NPZ. **원본 저장소에 없음** |
| PD | 5분 요약 `.npy` 또는 원본 도구의 281바이트 레코드 → 배경 잔차·정규화·창 | ASTRAZT1 표, 이전 날짜 상태, 전날 입력 코드. 레코드부터 처리하려면 지자기·태양 표도 필요. **현재 저장소에 없음** |

정식 PD 경로의 일반 netCDF에서 281바이트 레코드를 만드는 어댑터는 제공하지 않는다. 위 데모 모드는 L1b를 직접 읽어 명시적인 대체 통계·배경값으로 입력을 생성한다. 정식 모델 입력을 받으면 원자료 전처리 없이 DNN 측정을 진행할 수 있다.

이미 준비된 정수 입력 가져오기 예시:

```bash
python3 run.py prepare --kind tensor --model bus --input inputs/bus_codes.npy --valid-rows 16 --expected inputs/bus_expected.npy --output inputs/model2_BUS_input.npz
python3 run.py prepare --kind tensor --model pd --input inputs/pd_codes.bin --dtype int16 --valid-rows 288 --output inputs/model1_PD_input.npz
```

입력 모양은 기본 정수 실행에서 PD `[N,310,32]`, BUS `[N,64,12]`, Comp `[N,310,24]`, GRU `[N,1,34]`다. `N`은 창 개수다. `--valid-rows`는 제공자가 확인한 창별 실제 유효 표본 수를 넣는다. 위 PD의 288은 해당 형식의 외부 입력을 받았을 때의 예시이며 패딩·문맥과 별도로 검증해야 한다. 이 저장소의 PD 창 생성은 실제 계약의 문맥 46행을 사용해 최대 264행씩 처리하고 마지막 창의 유효 길이를 따로 기록한다.

압축 예측기 입력 준비:

```bash
python3 run.py prepare --kind counts --model gru --input inputs/metop_day.nc --output inputs/model3_GRU_input.npz
python3 run.py prepare --kind counts --model comp --input inputs/metop_day.nc --output inputs/model3_COMP_input.npz
```

계수 NPZ 대체 입력은 `counts: float32[T,C]`, `auxiliary: float32[T,6]`, `channels: 문자열[C]`다. 보조 입력은 위도 sin/cos, 경도 sin/cos, 진행 방향, 궤도 위상 순서다. 채널 이름으로 정렬하며 누락·결측·손실 양자화를 거부한다. 코드북·정규화 통계를 임의로 다시 학습하지 않는다.

BUS 전처리 예시:

```bash
python3 run.py prepare --kind bus --model bus --input inputs/bus.csv --stats inputs/model2_BUS_stats.npz --time-column time_s --output inputs/model2_BUS_input.npz
```

시간 열은 증가하는 초 단위 숫자다. 첫 시각을 기준으로 50초 격자를 만들며, 실제 학습 파이프라인과의 원점·보간 일치 여부는 기준 출력으로 검증한다. `mean/std`에는 학습 통계를 넣어야 하며 시험 데이터로 새 통계를 만들지 않는다.

PD 전처리 예시:

```bash
python3 run.py prepare --kind pd --model pd --input inputs/pd_summary.npy --date 20220101 --tables inputs/model1_PD_tables.bin --state inputs/model1_PD_history.npz --context inputs/model1_PD_previous_codes.npy --output inputs/model1_PD_input.npz
```

요약은 원본 `SUMMARY_COLUMNS` 순서의 `[288,36]` 배열이다. 상태 NPZ는 `days: 정수[D]`(1970년부터 지난 날짜), `medians: float32[D,18,23]`다. 표는 원본 ASTRAZT1 파일을 그대로 읽는다. 이전 날짜 상태·문맥이 없으면 중단한다. 다음 날짜에 사용할 갱신 상태는 출력 이름의 `.state.npz`로 저장된다. 출력 입력파일에도 `day_codes`가 저장되어 다음 날의 `--context`에 그 NPZ를 사용할 수 있다. 281바이트 레코드 입력은 `--input` 대신 `--records`, 추가로 `--geo`, `--sun`을 지정한다. `--sun`은 해당 날짜의 `[1440,2]` NPY다.

**4. DNN만 측정**

```bash
python3 run.py bench --model comp --device cuda --input inputs/model3_COMP_input.npz --duration 60 --repeat 3 --warmup 3 --tegrastats results/comp_health.txt --output results/model3_COMP_dnn.json
```

가중치·입력·GPU 전송을 측정 전에 준비하고, 예비 실행 뒤 DNN 호출 시간을 측정한다. CUDA는 전후 계산 완료를 기다린다. GRU는 파일 순서대로 상태를 갱신하고 입력 파일을 다시 시작할 때 상태를 초기화한다. 기준 출력이 입력 파일에 있으면 측정 전에 대조한다. 기준 출력이 없으면 검증을 통과했다고 표시하지 않는다.

결과: 프레임당 연산량, 실행 시간, 유효 표본/초, GOPS, 실행 환경, 가중치·입력 해시, 측정 시작·종료 Unix 시각. 전력 결합 구간은 DNN 호출을 반복하는 보드 전체이며 루프·동기화 비용을 포함한다. 본 도구는 신경망 임무 정확도(F1 등)를 산출하지 않는다.

**5. 실제 압축 비교**

네 조건 모두 **같은 28채널 원본을 복원**한다. GRU는 28채널, ModernTCN은 지원하는 13채널을 예측하고 나머지는 raw/delta로 처리한다. 따라서 이 결과는 현재 두 시스템 비교이며 같은 구조·채널 수로 통제한 신경망 연구 실험과 구분한다.

```bash
python3 run.py compress --model gru --mode raw --input inputs/metop_day.nc --output results/A_raw.scbz2
python3 run.py compress --model gru --mode delta --input inputs/metop_day.nc --output results/B_delta.scbz2
python3 run.py compress --model gru --mode neural --input inputs/metop_day.nc --output results/C_GRU.scbz2
python3 run.py compress --model comp --mode neural --input inputs/metop_day.nc --output results/D_ModernTCN.scbz2
```

처음에는 `--limit 16`으로 작은 구간을 확인한다. ModernTCN 압축·복원은 매 시점 같은 모양의 문맥 프레임을 실행하는 정확성 우선 구현이다. 느린 정수 참조의 하루치 실행부터 시작하지 않는다. 이 구현의 압축 시간은 최적화된 프레임 일괄 예측 성능을 의미하지 않는다.

기본 블록은 1,800표본, BZ2 레벨은 5다. raw는 원래 값을 BZ2 압축하는 후보, delta는 직전 값 차이, neural은 예측 오차다. 신경망 조건은 지원 채널에서 세 후보를 전부 계산한다. 선택되지 않은 후보 비용도 포함한다. 비신경망 기준에는 불필요한 보조자료를 저장하지 않는다.

측정 범위는 메모리의 계수·보조자료 → 예측 입력 변환 → 예측 → 후보 압축 → 완성 패킷이다. 파일 읽기·모델 로딩·디스크 저장·복원 검증은 제외한다. 결과 JSON은 패킷 옆에 저장된다. 전체 파일 크기는 헤더·초기값·보조자료를 포함한다. 원본 크기 분모는 int32 계수 바이트다.

모든 압축 명령은 측정 후 원본 없이 패킷과 모델로 복원하고, 입력과 완전히 같은지 검사한 뒤 저장한다. 다른 장치·라이브러리에서의 실수 예측은 반올림이 달라질 수 있다. 패킷의 모델·실행 방식·원본 SHA-256을 검사하며 불일치를 성공으로 처리하지 않는다. 이 저장소 패킷은 자체 실험 포맷으로 기존 선배 코덱과 파일 호환을 주장하지 않는다.

```bash
python3 run.py decode --model gru --input results/C_GRU.scbz2 --output results/restored.npz
```

**6. 세 DNN 동시 실행**

```bash
python3 run.py together --pd-input inputs/model1_PD_input.npz --bus-input inputs/model2_BUS_input.npz --comp-input inputs/model3_COMP_input.npz --comp-model comp --device cuda --duration 60 --tegrastats --output results/together_comp
```

각 작업의 모델 로딩·예비 실행이 끝나야 함께 시작한다. GPU 메모리 부족이나 작업 실패는 로그를 남기고 나머지 작업도 종료한다. `--comp-model gru`와 GRU 입력으로 바꾸면 비교 조건이 된다. 기본은 각 프로세스 CPU 스레드 1개다. 이 명령은 **세 DNN 최대 부하 동시 실행**이며 실제 주기 기반 임무 스케줄러나 압축 전체 동시 실행이 아니다. 공통 시간 구간은 세 측정 구간의 교집합으로 기록한다.

**7. 전력 로그 연결**

계측기는 장비 모델과 연결 방식이 정해지면 드라이버를 붙인다. 현재 구현은 계측기에서 내보낸 CSV 결합이며 장비 제어나 100 Hz 취득 성공을 주장하지 않는다. 형식은 다음과 같다.

```csv
time_s,power_w
1790000000.000,4.20
1790000000.010,4.35
```

`time_s`는 **Jetson 실행 결과와 같은 시계 기준의 Unix epoch 초**, `power_w`는 보드 입력 전력 W다. 상대 시간이나 임의 시차 보정은 자동으로 추정하지 않는다. 다른 PC에서 기록했다면 시계 동기화 또는 검증된 시각 변환을 먼저 한다.

```bash
python3 run.py power --input inputs/power.csv --result results/model3_COMP_dnn.json --output results/model3_COMP_power.json
```

압축 JSON 또는 `together_comp/system.json`도 연결할 수 있다. 전체 측정 구간을 덮지 않는 로그, 비증가 시각, 음수 전력, 허용치보다 큰 기록 공백은 거부한다. 기본 최대 공백은 0.05초이며 계측기 조건에 따라 `--max-gap`을 명시한다. 평균 W, **관측 최대 W**, 적분 에너지 J, 실제 표본 간격을 기록한다. 100 Hz는 10 ms 간격이며 이보다 짧은 순간 피크를 모두 포착하지는 못한다. `tegrastats`는 보조 상태 로그다.

**검증 범위와 다음 준비물**

현재 CPU 자동 검증은 실제 가중치를 사용한 정수 실행의 NumPy 기준 대조, 체크포인트 구조 로딩, 인과성, GRU 예측 계산, 공통 코덱 복원·손상 거부, 입력 식별, 전력 적분을 포함한다. 합성 데이터 기능 검증을 실제 위성 데이터 성능으로 보고하지 않는다.

남은 외부 자료는 ① 모델별 실제 입력과 기준 출력, ② PD 배경·정규화·지자기 표와 상태, ③ BUS 학습 통계, ④ 전력 측정기 사양이다. 실제 Nano CPU/CUDA 결과, 실측 전력, 임무 정확도는 보드에서 별도로 확인한다.

원본: [pd-bus-moderntcn-fpga](https://github.com/juwonde/pd-bus-moderntcn-fpga), [SIREN_MODEL3_COMP](https://github.com/demisodapple/SIREN_MODEL3_COMP). 구체적인 이식 출처와 해시는 `SOURCES.json`에 보존한다.
