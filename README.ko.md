# 2D 재사용 로켓 자동착륙 유도

[English README](README.md)

2D 강체 시뮬레이션부터 고전제어, 최적제어 교사, 모방학습과 강건성 평가까지 단계적으로 구현하는 재사용 로켓 동력 착륙 프로젝트입니다.

## 프로젝트 현황

| 항목 | 현재 상태 |
|---|---|
| 로드맵 | 3주차 최적제어 문제 정식화(15일차) |
| 모델 | 가변 질량 평면 3자유도 |
| 상태 | x, z, vx, vz, theta, omega, mass |
| 제어 | throttle, gimbal angle |
| 현재 제어기 | Suicide-burn, 수직 PID, 수평·자세 제어, 통합 착륙 제어 |
| 실행 환경 | Python 3.12.7 |

현재 저장소에는 검증된 시뮬레이션 환경, 두 가지 비학습 수직 착륙 기준선, 수평 위치·자세 직렬 제어기, 동결된 통합 PID 기준선, 재현 가능한 튜닝, 분리 외란 평가와 명목 최적제어 문제 정의가 구현되어 있습니다. 최적제어 솔버, Behavior Cloning, DAgger, 복합 불확실성과 더 넓은 Monte Carlo 평가는 이후 로드맵에서 진행합니다.

## 최적제어 Teacher 문제 정식화

15일차에는 7상태·2제어 명목 착륙 문제를 정의했습니다. 제어 구간 100개와 5-25초 자유 종료시간, 시뮬레이터와 일치하는 매끄러운 동역학, 고도·연료·구동기 경로 제한, 착륙 종단 제한, 1단계 가능해 탐색과 2단계 연료·접지·제어 변화 비용이 포함됩니다. 실제 수치해법은 16일차 작업이므로 아직 최적 Teacher 궤적이나 성공률을 주장하지 않습니다. [수식-코드 대응표](docs/spec.ko.md#15일차-최적제어-teacher-문제-정식화)와 [영문 명세](docs/spec.md#21-day-15-optimal-control-teacher-formulation)에서 세부 정의를 확인할 수 있습니다.

## 동결된 2주차 PID 기준선

12일차 grid search에서 선택한 결과를 변경하지 않는 비교 기준 `pid-baseline-v1`로 동결했습니다. 선택된 위치 gain, 속도 gain, 하강 profile 배율 `(1.10, 0.95, 1.10)`은 실행 중 다시 곱하지 않고 `configs/pid-baseline-v1.yaml`에 실제 값으로 기록했습니다. Protocol은 통합 제어기 설정과 생성된 초기조건 행렬의 SHA-256을 모두 저장하며, 둘 중 하나라도 바뀌면 평가를 중단합니다.

고정 평가 집합은 `integrated_uniform_v1`과 seed 20260914로 생성한 무풍 중간 난이도 초기조건 1,000개입니다. 수평 오차 3-15 m, 고도 80-120 m, 수평 속도 -2~2 m/s, 수직 속도 -25~-15 m/s, 자세 -5~5도, 각속도 -2~2 deg/s, 질량 950-1000 kg 범위를 포함합니다. Digest는 `a9593a383acc0017738ed383a123ca743a32bce46c64d249b08ca577a448a952`입니다.

| 평가 항목 | 결과 |
|---|---:|
| 안전 착륙 | 999 / 1,000 (99.9%) |
| 평균 / 최대 절대 접지 위치 | 0.4098 / 1.3494 m |
| 평균 / 최대 수평 접지 속도 | 0.1656 / 0.5915 m/s |
| 평균 / 95백분위 / 최대 수직 접지 속도 | 0.7802 / 0.7810 / 0.7814 m/s |
| 평균 / 최대 접지 자세 | 0.5991 / 3.2498도 |
| 평균 / 최대 접지 각속도 | 1.3178 / 3.6500 deg/s |
| 평균 연료 사용량 | 51.4460 kg |
| 최대 throttle / gimbal 변화율 | 초당 1.8658 / 초당 60.0000도 |

2주차 통과 기준 70%를 충족했습니다. 한 사례는 착륙장 위치 제한 1 m를 벗어나 crash로 판정됐으며 나머지 접지 제한의 최대값은 모두 설정 범위 안이었습니다. 이 결과는 시뮬레이션 기준이며 실제 하드웨어 안전을 의미하지 않습니다. 아래 산출물 명령은 같은 초기상태에서 무풍 성공과 40 m/s 일정 바람 스트레스 실패도 생성합니다. 스트레스 실패는 명목 999/1,000 결과에 포함되지 않습니다.

## 2주차 결과 요약

| 단계 | 고정 평가 | 결과 |
|---|---|---:|
| 수직 suicide-burn | 경계 예시 3개 | 안전 착륙 3 / 3 |
| 수직속도 PID | 수직·무풍 초기조건 100개 | 안전 착륙 100 / 100 |
| 수평·자세 루프 | 8초 고도 유지 조건 100개 | 착륙장 방향 수렴 100 / 100 |
| 통합 PID 튜닝 검증 | 선택에 쓰지 않은 초기조건 100개 | 안전 착륙 100 / 100 |
| 동결 통합 PID | 명목 초기조건 1,000개 | 안전 착륙 999 / 1,000 |

이후 제어기, 최적제어와 학습 정책 비교는 모두 `configs/pid-baseline-v1.yaml`을 `frozen_baseline_initial_states`로 읽어야 합니다. 따라서 초기조건을 동일하게 유지하고 설정 변경도 탐지할 수 있습니다.

## 재현 가능한 기준선 튜닝

통합 제어기는 Cartesian grid search와 seed를 고정한 random search를 모두 지원합니다. 기본 grid는 배율을 적용하지 않은 기준 후보를 포함해 13개 후보를 평가합니다. 후보 선택에는 seed 20260912의 초기조건 16개를 사용하고, 선택된 후보는 seed 20260913의 별도 초기조건 100개에서 한 번만 검증합니다. 검증 조건은 후보 선택에 사용하지 않습니다.

최소화하는 목적함수는 실패율, 전체 250 kg 연료 대비 사용 비율, x·vx·vz·자세·각속도의 평균 정규화 접지 오차를 결합합니다.

~~~text
score = 1000 * failure_rate + 5 * fuel_fraction + 10 * normalized_landing_error
~~~

선택된 배율은 수평 위치 이득 1.10, 수평 속도 이득 0.95, 하강 profile 감속도 1.10입니다. 학습 조건에서 선택 후보와 배율을 적용하지 않은 기준 후보는 모두 16회 중 16회 성공했고, 선택 후보는 점수를 4.179125에서 3.716630으로, 평균 연료 사용량을 52.9862 kg에서 51.1532 kg으로 줄였습니다. 한 번도 선택에 사용하지 않은 검증 조건에서는 100회 모두 성공했고 평균 연료 사용량은 51.9394 kg, 평균 절대 위치 오차는 0.3942 m였습니다.

생성되는 보고서에는 모든 후보, 목적함수 항, seed와 검증 지표가 기록됩니다. 생성 YAML은 그대로 불러올 수 있는 전체 설정입니다. `configs/default.yaml`은 동결 전 튜닝 기준으로 유지하고, `configs/pid-baseline-v1.yaml`은 이후 비교에 쓰는 선택·동결된 2주차 기준선입니다.

## 분리 외란 벤치마크

튜닝된 통합 PID를 seed 20260915의 동일 초기조건 30개에서 외란별로 따로 평가했습니다. 일정 바람과 2초 반사인 돌풍에 대한 이차 공기저항, 서로 독립인 Gaussian 센서 잡음, 실제 추력 scale 오차와 1차 throttle 지연을 다룹니다. 안전 착륙 성공률이 처음으로 95% 아래가 되는 실험 강도를 붕괴점으로 정의했습니다.

| 외란 | 95% 이상인 마지막 실험 강도 | 첫 붕괴 강도 | 붕괴점 성공 |
|---|---:|---:|---:|
| 일정 바람 | 10 m/s | 20 m/s | 26 / 30 |
| 돌풍 진폭 | 40 m/s | 60 m/s | 26 / 30 |
| 센서 잡음 배율 | 1배 | 2배 | 26 / 30 |
| 실제 추력 배율 | 1.00배 | 0.95배 | 28 / 30 |
| Throttle 지연 시정수 | 0.2 s | 0.5 s | 27 / 30 |

무외란 조건은 30회 중 29회 성공했습니다. 앞선 100/100 검증과 다른 seed에서 경계 조건 하나가 드러난 결과이며, 유한한 성공 표본이 모든 조건의 강건성을 보장하지 않는다는 뜻입니다. 초기조건, 바람 방향, 돌풍 시작 시각과 센서 잡음열은 강도 단계 사이에서 짝을 맞췄으므로 각 곡선에서는 하나의 모델 요소만 달라집니다. 이 수치는 설정된 실험 grid의 경계이지 인증된 운용 한계가 아닙니다.

## 수평·자세 제어 벤치마크

수평 제어기는 위치와 속도 오차를 제한된 목표 기울기로 바꾸고, 자세 PD 제어로 gimbal 명령을 계산합니다. 추력 방향이 기울어지면서 줄어드는 수직 추력 성분은 throttle로 보상합니다. 설정 제한은 수평 가속도 2.5 m/s², 목표 기울기 15도, 기체 gimbal 15도입니다.

무풍 초기조건 100개에서 수평 위치 오차 5-20 m, 수평 속도 -2~2 m/s, 자세 -5~5도, 각속도 -2~2 deg/s, 질량 950-1000 kg을 사용했습니다. 수평·자세 동작만 분리해 보기 위해 각 실험은 고도 100 m를 유지하며 8초 동안 실행했습니다.

| 평가 항목 | 결과 |
|---|---:|
| 정상 완료 | 100 / 100 |
| 착륙장 방향 수렴 | 100 / 100 |
| 평균 초기 절대 위치 오차 | 12.4712 m |
| 평균 최종 절대 위치 오차 | 5.4894 m |
| 최대 목표 기울기 | 9.7647도 |
| 최대 gimbal 명령 | 7.0393도 |
| 최대 고도 편차 | 0.0149 m |

이 분리 벤치마크는 통합 착륙 제어기와 별개로 수평 동작만 진단할 때 계속 사용할 수 있습니다.

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

## 착륙 제어기 실행

수평·수직 오차가 있는 상태에서 통합 제어기를 실행합니다.

~~~powershell
python scripts/run_episode.py --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid
~~~

수직속도 PID 제어기를 실행합니다.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid
~~~

Suicide-burn 제어기를 실행합니다.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller suicide-burn
~~~

초기조건 순서는 x z vx vz theta omega mass입니다. 모든 값은 SI 단위이며 theta와 omega는 rad와 rad/s를 사용합니다.

통합 착륙 에피소드 기록, 진단 그래프와 애니메이션을 함께 저장합니다.

~~~powershell
python scripts/run_episode.py --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid --output artifacts/integrated.json --plot artifacts/integrated.png --animation artifacts/integrated.gif
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

수평·자세 제어 벤치마크를 동일한 조건으로 다시 실행합니다.

~~~powershell
python scripts/evaluate_horizontal_control.py --episodes 100 --seed 20260914
~~~

착륙장 방향 수렴률, 위치 오차 감소량, 최대 기울기·gimbal과 수직 추력 보상 중 고도 편차를 출력합니다.

중간 난이도 초기조건 100개의 통합 착륙 벤치마크를 실행합니다.

~~~powershell
python scripts/evaluate_integrated_control.py --episodes 100 --seed 20260914
~~~

모든 접지 상태값, 연료 사용량, phase별 제어 갱신 횟수와 실제 throttle·gimbal 변화율을 출력합니다.

통합 기준선을 튜닝하고 전체 보고서와 최적 설정을 저장합니다.

~~~powershell
python scripts/tune_integrated_controller.py --report artifacts/tuning.json --best-config artifacts/best-integrated.yaml
~~~

Seed가 고정된 random search는 `--method random`을 추가해 실행합니다. 두 출력은 기존 파일을 자동으로 덮어쓰지 않습니다. 선택된 설정은 다른 실행 명령에서 바로 사용할 수 있습니다.

~~~powershell
python scripts/run_episode.py --config artifacts/best-integrated.yaml --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid --output artifacts/tuned-landing.json --plot artifacts/tuned-landing.png --animation artifacts/tuned-landing.gif --fps 20 --max-frames 240
~~~

GIF 렌더링은 시뮬레이션이 끝난 뒤 수행되므로 몇 초 이상 걸릴 수 있습니다. `--max-frames`는 기록된 궤적을 고르게 줄여 렌더링 시간과 파일 크기를 조절하며 물리 시뮬레이션 결과는 바꾸지 않습니다.

동결된 2주차 초기조건 1,000개 평가와 성공·실패 GIF를 재현합니다.

~~~powershell
python scripts/evaluate_frozen_baseline.py --output-dir artifacts/week2-baseline
~~~

실행 전에 제어기와 초기조건 SHA-256을 검증하고 성공률 70% 통과 기준을 적용하며, 다섯 산출물 중 하나라도 있으면 덮어쓰지 않습니다. 두 GIF는 같은 초기조건을 사용합니다. 성공 영상은 명목 조건이고 실패 영상은 명시된 40 m/s 일정 바람 스트레스 조건입니다.

분리 외란 보고서와 성능 곡선을 생성합니다.

~~~powershell
python scripts/evaluate_pid_disturbances.py --report artifacts/disturbances.json --plot artifacts/disturbances.png
~~~

강도별 기본 실험 횟수 30회는 `--episodes`로 바꿀 수 있습니다. 기존 출력은 덮어쓰지 않습니다.

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

## 수평 위치·자세 제어 루프

외부 루프는 수평 위치·속도 오차를 목표 수평 가속도와 제한된 목표 자세로 변환합니다. 내부 루프는 자세 오차와 각속도로 목표 각가속도를 계산하고 추력 토크 식을 역으로 풀어 gimbal 각도를 구합니다.

~~~text
target_ax = clip(Kp_x * (target_x - x) - Kd_x * vx)
target_theta = clip(atan2(target_ax, gravity), max_tilt)
target_alpha = Kp_theta * (target_theta - theta) - Kd_theta * omega
gimbal = clip(asin(-target_alpha * inertia / (lever_arm * thrust)))
~~~

Coupling 보상을 켜면 기본 throttle을 `cos(theta + gimbal)`로 나누어 수직 추력 성분을 유지합니다. 최종 throttle과 gimbal에는 구동기 제한이 적용됩니다. 추력이 0이면 자세 토크를 만들 수 없으므로 gimbal도 안전하게 0을 반환합니다.

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
- 2주차: suicide-burn과 동결 PID 기준선 - 완료
- 3주차: 제약 최적제어 교사 - 문제 정식화 완료, 솔버 예정
- 4주차: 데이터셋 생성과 Behavior Cloning
- 5주차: DAgger 폐루프 개선
- 6주차: 외란과 Monte Carlo 평가
- 7주차: 보고서, 비교 시각화와 조작 화면

## 현재 한계

- 착륙 직전 동력 하강만 다루며 발사, 상승, 귀환 기동과 재진입은 범위 밖입니다.
- 외란은 한 번에 하나씩 평가했으며 바람, 센서와 구동기 고장이 결합된 상황은 아직 다루지 않았습니다.
- 공기저항은 공력 토크, 양력, 고도별 밀도와 난류가 없는 점 힘 모델입니다.
- 센서 오차는 서로 독립인 영평균 Gaussian 표본이며 엔진 지연은 현재 throttle에만 적용합니다.
- 지면 접촉은 착륙 다리와 구조 충격이 없는 점 모델입니다.
- 성공 기준 2 m/s는 시뮬레이션 판정값이며 하드웨어 안전을 보증하지 않습니다.

좌표계, 단위, 사건과 에피소드 기록 규약은 [한국어 모델·환경 명세](docs/spec.ko.md)에서 확인할 수 있습니다. [영문 명세](docs/spec.md)도 함께 제공합니다.
