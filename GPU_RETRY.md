# Jetson Nano GPU 재진단

| 확인된 사실 | 해석 |
|---|---|
| 이전 커널 로그의 OOM killer가 python3 종료 | 시스템 메모리 부족으로 종료된 사실은 확인 |
| 가중치 로딩과 입력 전송 완료 후 첫 추론 중 종료 | 정확한 층·임시 메모리 원인은 미확인 |
| PD 파라미터 약 46.5 MiB, 압축 TCN 약 28.5 MiB | 가중치 크기만으로 4 GB 부족을 설명할 수 없음 |
| Nano CPU와 GPU가 물리 RAM 공유 | GPU 임시 공간·CUDA 라이브러리·OS 점유를 함께 확인 |

## 1 업데이트 및 환경 기록

Jetson SSH 터미널에서 실행한다. 패키지/OS 재설치나 swap 변경은 하지 않는다.

```bash
cd ~/SIREN_JETSON_TEST
git pull --ff-only
source .venv/bin/activate
export OPENBLAS_CORETYPE=ARMV8
free -h
pgrep -af 'python.*run.py'
sudo nvpmodel -q
```

기존 실험이 동작 중이면 해당 터미널에서 종료한 뒤 진행한다. GPU 진단 중 다른 모델을 동시에 실행하지 않는다. 아래 명령은 서로 다른 결과 폴더를 만든다.

## 2 작은 GPU 작업과 모델 층별 진단

```bash
python3 -u gpu_probe.py --model bus --cudnn-mode off --output "results/gpu_bus_$(date +%Y%m%d_%H%M%S)"
python3 -u gpu_probe.py --model pd --cudnn-mode off --output "results/gpu_pd_off_$(date +%Y%m%d_%H%M%S)"
python3 -u gpu_probe.py --model comp --cudnn-mode off --output "results/gpu_comp_off_$(date +%Y%m%d_%H%M%S)"
```

각 명령의 `success: true`를 확인하고 다음 단계로 진행한다. 실패하면 결과 폴더 전체를 보관한다. 비밀번호가 필요한 커널 로그는 다음 명령으로 별도 저장한다.

```bash
sudo dmesg -T | tail -n 250 > results/gpu_kernel_latest.txt
```

| 파일 | 확인할 내용 |
|---|---|
| stages.jsonl | PyTorch import → 작은 CUDA 연산 → 모델 CPU 로딩 → GPU 전송 → 각 층 시작/완료 → 두 번의 추론 |
| system_memory.jsonl | 진단 중 시스템 가용 RAM·swap, 약 2초 간격 |
| tegrastats.log | 사용 가능한 경우 RAM·GPU·온도, 0.5초 간격 |
| worker.log | 오류 메시지 및 Python traceback |
| diagnosis.json | 종료 코드, timeout 여부, 성공 여부 |
| kernel.log | 권한이 허용될 때 커널 로그. 과거 기록도 포함하므로 시간 비교 필수 |

`layer_begin` 다음에 `layer_done` 없이 종료됐다면 해당 층이 우선 조사 대상이다. CUDA allocated/reserved 수치는 드라이버·외부 라이브러리의 모든 메모리를 포함하지 않는다. 진단 도구는 층마다 동기화하므로 그 실행 시간을 정식 성능으로 사용하지 않는다.

## 3 cuDNN 경로 비교

PD의 off 진단을 끝낸 뒤, 별도 프로세스에서 기존 경로와 비교한다. 성공을 보장하는 설정이 아니라 실패 원인 분리를 위한 시험이다.

```bash
python3 -u gpu_probe.py --model pd --cudnn-mode deterministic --output "results/gpu_pd_original_$(date +%Y%m%d_%H%M%S)"
python3 -u gpu_probe.py --model pd --cudnn-mode heuristic --output "results/gpu_pd_heuristic_$(date +%Y%m%d_%H%M%S)"
```

| 모드 | GPU 실행 | cuDNN | 용도 |
|---|---|---|---|
| deterministic | CUDA | 사용, 결정적 알고리즘 | 기존 코드와 같은 선택 조건 |
| heuristic | CUDA | 사용, 결정성 제한 해제 | 다른 알고리즘 선택 경로 비교 |
| off | CUDA | 사용하지 않음 | PyTorch CUDA 대체 경로 비교; CPU 전환이 아님 |

모든 모드에서 알고리즘 벤치마크 탐색은 꺼져 있다. off만 성공하면 cuDNN 경로 관여를 의심할 수 있으나 정확한 알고리즘·임시 공간이 원인인지 로그로 확인해야 한다. off도 메모리 절약이나 속도 향상을 보장하지 않는다.

## 4 정식 계측 전 짧은 확인

아래는 off가 성공했을 때의 예시다. 다른 모드가 성공했다면 모든 명령에서 모드를 동일하게 바꾼다. 각 단계 성공 후 다음으로 진행한다.

```bash
python3 -u run.py trial --device cuda --cudnn-mode off --phase single --duration 3 --repeat 1 --warmup 1 --output "results/gpu_single_$(date +%Y%m%d_%H%M%S)"
python3 -u run.py trial --device cuda --cudnn-mode off --phase together --duration 5 --repeat 1 --warmup 1 --output "results/gpu_together_check_$(date +%Y%m%d_%H%M%S)"
```

단독은 성공하고 동시 실행만 실패하면 세 프로세스의 CUDA 컨텍스트·모델·임시 메모리 합계를 조사한다. 입력 길이/채널을 임의로 줄여 기존 결과와 비교하지 않는다.

## 5 GPU 전력 및 OPS 측정

짧은 동시 실행까지 성공한 뒤 진행한다. 센서 읽기 권한은 기존 전력 시험과 동일하게 필요하다.

```bash
python3 run.py trial --device cuda --cudnn-mode off --phase together --duration 60 --repeat 3 --record-power --tegrastats --no-progress --output "results/TCN_GPU_power_$(date +%Y%m%d_%H%M%S)"
```

GRU 비교 전에는 `gpu_probe.py --model gru --cudnn-mode off`에 새 `--output`을 지정해 먼저 확인한다. 본 시험은 위 명령에 `--comp-model gru`를 추가하고 결과 이름을 `GRU_GPU_power_...`로 바꾼다.

cuDNN 모드는 trial 설정과 개별 모델 결과에 저장한다. FP32, 입력, 연산량 계산법은 유지한다. GPU 시간에는 CUDA 동기화가 포함된다. 전력은 약 10 Hz 모듈 입력 센서값이며 순간 피크 검증은 아니다. 정확도·원본 모델 동등성은 별도 검증 대상이다.

근거: NVIDIA CUDA for Tegra의 메모리 구조 문서와 PyTorch 1.10 cuDNN backend 소스.
https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html
https://github.com/pytorch/pytorch/blob/v1.10.0/torch/backends/cudnn/__init__.py
