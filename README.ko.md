# 2D 재사용 로켓 자동착륙 유도

[English README](README.md)

2D 강체 시뮬레이션부터 고전제어, 최적제어 교사, 모방학습과 강건성 평가까지 단계적으로 구현하는 재사용 로켓 동력 착륙 프로젝트입니다.

## 프로젝트 현황

| 항목 | 현재 상태 |
|---|---|
| 로드맵 | 2주차 고전제어 단계 |
| 모델 | 가변 질량 평면 3자유도 |
| 상태 | x, z, vx, vz, theta, omega, mass |
| 제어 | throttle, gimbal angle |
| 현재 제어기 | Suicide-burn, 수직속도 PID |
| 실행 환경 | Python 3.12.7 |

현재 저장소에는 검증된 시뮬레이션 환경과 두 가지 비학습 수직 착륙 기준선이 구현되어 있습니다. 수평·자세 제어, 최적제어, Behavior Cloning, DAgger, 외란과 Monte Carlo 평가는 이후 로드맵에서 진행합니다.

## 수직속도 PID 벤치마크

수직속도 제어기는 고도에 따라 달라지는 목표 하강속도를 throttle PID로 추종합니다. 중력과 목표 감속도를 feed-forward throttle로 계산하고, 출력이 0 또는 1에 도달하면 오차 방향을 확인해 적분을 멈추며 적분값 자체도 제한합니다.

선택한 이득은 `Kp=0.08`, `Ki=0.001`, `Kd=0.01`입니다. 고도 80-120 m, 수직속도 -25~-15 m/s, 질량 950-1000 kg 범위에서 고정 seed로 100개 초기조건을 생성했습니다. 수직 전용 벤치마크이므로 수평 위치·속도와 자세·각속도는 0으로 고정했습니다.

| 평가 항목 | 결과 |
|---|---:|
| 안전 착륙 | 100 / 100 |
| 성공률 | 100% |
| 평균 접지 속도 | 1.1010 m/s |
| 접지 속도 95백분위 | 1.1938 m/s |
| 평균 연료 사용량 | 37.4923 kg |

모든 표본이 접지 속도 2 m/s 이하였습니다.

## Suicide-burn 벤치마크

Suicide-burn 제어기는 하강 중인 로켓이 설정된 접지 속도 2 m/s를 만족하도록 점화 시점을 계산합니다. 로드맵에서 요구한 상수 질량 폐형식 계산을 기준식으로 제공하고, 실제 실행 제어기에는 연료 소모에 따른 질량 변화와 0.02초 제어 주기 안에서 발생하는 점화 시점 보정을 적용했습니다.

| 초기 고도 | 초기 수직 속도 | 초기 질량 | 점화 고도 | 접지 속도 | 결과 |
|---:|---:|---:|---:|---:|---|
| 80 m | -15 m/s | 950 kg | 42.184892 m | -1.998971 m/s | success |
| 100 m | -20 m/s | 1000 kg | 58.397961 m | -1.996813 m/s | success |
| 120 m | -25 m/s | 1000 kg | 73.606685 m | -1.995567 m/s | success |

이 결과는 수직, 무풍 조건의 기준선 검증입니다. 구조 안전성이나 수평 오차, 기울기, 바람, 센서 잡음, 구동기 지연이 있는 조건의 성능을 의미하지 않습니다.

## 시스템 흐름

~~~text
초기조건
   |
   v
Gymnasium 환경 <---- 제어기
   |                  ^
   v                  |
가변 질량 동역학 -----+
   |
   +----> 지면 접촉 및 연료 사건
   |
   +----> JSON 또는 NPZ 에피소드 기록
                    |
                    +----> 그래프와 애니메이션
~~~

상태와 제어 규약은 시뮬레이션, 제어기, 이후 데이터셋과 학습 정책이 함께 사용합니다. 저장한 에피소드는 물리를 다시 계산하지 않고 재생할 수 있습니다.

## 설치

프로젝트 루트에서 Python 3.12.7을 사용합니다.

~~~powershell
python --version
python -m pip install -e ".[dev]"
~~~

## 수직 착륙 제어기 실행

수직속도 PID 제어기를 실행합니다.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid
~~~

Suicide-burn 제어기를 실행합니다.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller suicide-burn
~~~

초기조건 순서는 x z vx vz theta omega mass입니다. 모든 값은 SI 단위이며 theta와 omega는 rad와 rad/s를 사용합니다.

에피소드 기록, 진단 그래프와 애니메이션을 함께 저장하려면 다음과 같이 실행합니다.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid --output artifacts/velocity-pid.json --plot artifacts/velocity-pid.png --animation artifacts/velocity-pid.gif
~~~

저장된 기록을 물리 계산 없이 재생할 수 있습니다.

~~~powershell
python scripts/run_episode.py --replay artifacts/velocity-pid.json --animation artifacts/replay.gif
~~~

생성물은 Git에서 제외되며 기존 파일을 자동으로 덮어쓰지 않습니다.

## 검증

전체 회귀 테스트와 코드 검사를 실행합니다.

~~~powershell
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
~~~

1주차 수치 안정성 검증을 다시 실행할 수 있습니다.

~~~powershell
python scripts/verify_week1.py --episodes 100 --seed 20260907
~~~

이 검증은 좌표계와 토크 부호, 연료 유량, 5가지 종료 결과, 상태 유한성, 질량 단조 감소, 지면 침범, 사건 시간과 기록 일치를 확인합니다.

동일한 초기조건에서 PID 이득 조합을 비교합니다.

~~~powershell
python scripts/sweep_velocity_gains.py --episodes 100 --seed 20260910
~~~

성공률, 접지 속도 통계와 평균 연료 사용량이 출력됩니다. 다른 조합은 `--kp`, `--ki`, `--kd`에 각각 여러 값을 전달해 비교할 수 있습니다.

## Suicide-burn 계산

첫 번째 기준 계산에서는 연소 중 질량을 현재값으로 고정합니다.

~~~text
a_net = T_max / mass - gravity
h_stop = (downward_speed^2 - target_speed^2) / (2 * a_net)
~~~

필요 고도보다 위에서 점화하면 early_burn, 아래에서 점화하면 late_burn으로 분류합니다. 실제 실행 기준선은 가변 질량 해석으로 점화 고도를 보정하고, 이상적인 점화가 두 제어 시점 사이에 있으면 첫 연소 스텝에 부분 스로틀을 적용합니다.

## 수직속도 profile

목표값은 고도가 낮아질수록 허용 하강속도가 작아지도록 정의합니다.

~~~text
target_vz = -min(max_descent_speed, sqrt(touchdown_speed^2 + 2 * deceleration * altitude))
throttle = saturate(feed_forward + Kp * error + Ki * integral + Kd * error_rate)
~~~

적분항은 설정 범위로 제한하며, 포화된 출력을 오차가 같은 방향으로 더 밀어내는 동안에는 누적하지 않습니다.

## 저장소 구조

~~~text
README.md / README.ko.md        영문 기본·한국어 프로젝트 소개
configs/                        공용 실험 설정
docs/spec.md / docs/spec.ko.md  영문 기본·한국어 기술 명세
scripts/                        재현 가능한 실험 실행 명령
src/powered_landing_guidance/   물리, 환경, 제어기와 시각화
tests/                          회귀, 경계 조건과 제어기 테스트
.local/experiment_logs/         Git에서 제외되는 로컬 일차 기록
~~~

Git에 포함되는 저장소에는 소스 코드, 재현 가능한 설정, 기술 문서와 테스트만 둡니다. 일차별 작업 기록은 개발 컴퓨터의 .local/experiment_logs/에 보관하며 GitHub에는 올라가지 않습니다.

## 로드맵

- 1주차: 시뮬레이터, 사건 처리, 기록, 재생과 수치 검증 - 완료
- 2주차: suicide-burn과 PID 기준선 - 진행 중
- 3주차: 제약 최적제어 교사
- 4주차: 데이터셋 생성과 Behavior Cloning
- 5주차: DAgger 폐루프 개선
- 6주차: 외란과 Monte Carlo 평가
- 7주차: 보고서, 비교 시각화와 조작 화면

## 현재 한계

- 착륙 직전 동력 하강만 다루며 발사, 상승, 귀환 기동과 재진입은 범위 밖입니다.
- 현재 제어기들은 수직 전용이며 수평 위치와 자세를 보정하지 않습니다.
- 공기저항, 바람, 센서 잡음, 추력 오차와 엔진 지연은 아직 적용하지 않았습니다.
- 지면 접촉은 착륙 다리와 구조 충격이 없는 점 모델입니다.
- 성공 기준 2 m/s는 시뮬레이션 판정값이며 하드웨어 안전을 보증하지 않습니다.

좌표계, 단위, 사건과 에피소드 기록 규약은 [한국어 모델·환경 명세](docs/spec.ko.md)에서 확인할 수 있습니다. [영문 명세](docs/spec.md)도 함께 제공합니다.
