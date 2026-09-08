# 2D 재사용 로켓 자동착륙 유도

최적제어 해를 학습한 정책이 실시간으로 로켓을 착륙시키는 과정을 구현하는 7주 프로젝트입니다. 현재 1주차인 7일차까지 완료했으며, 2D 강체 물리 모델과 실험 환경, 결과 기록, 시각화, 자동 검증 도구가 준비되어 있습니다. 자동착륙 제어기는 2주차부터 구현합니다.

이 프로젝트는 x, z, theta를 사용하는 평면 3자유도 모델입니다. y축 이동과 6자유도 운동은 포함하지 않습니다.

## 현재 구현 범위

- 상태: [x, z, vx, vz, theta, omega, mass]
- 제어: [throttle, gimbal_angle]
- 중력, 추력 벡터, 회전 토크, 비추력 기반 연료 소모
- Euler 및 RK4 적분
- 지면 최초 접촉 시각과 연료 소진 시각 계산
- 성공, 강한 착륙, 충돌, 연료 고갈, 시간 초과 판정
- seed 기반 초기조건 추출과 Gymnasium 환경
- JSON 및 NPZ 기록 저장과 무계산 재생
- 상태 그래프와 2D GIF 또는 MP4 생성
- 좌표계, 물리 부호, 종료 조건, 무작위 100회 실행 검증

## 실행 환경

Python 3.12.7을 사용합니다. 프로젝트 루트에서 다음 명령을 실행합니다.

~~~powershell
python --version
python -m pip install -e ".[dev]"
~~~

표시된 버전이 Python 3.12.7인지 확인합니다.

## 1주차 전체 검증

다음 명령은 좌표계와 단위, 중력과 짐벌 부호, 연료 유량, 5가지 종료 상황을 확인한 뒤 서로 다른 초기조건에서 무작위 행동 에피소드 100회를 실행합니다.

~~~powershell
python scripts/verify_week1.py
~~~

같은 검증을 재현하거나 실행 수를 바꾸려면 다음과 같이 지정합니다.

~~~powershell
python scripts/verify_week1.py --episodes 100 --seed 20260907
~~~

검증 도중 NaN, Inf, 지면 침범, 건조 질량 침범, 시간 역행, 기록 불일치가 발견되면 명령은 실패합니다.

회귀 테스트와 코드 검사는 다음과 같이 실행합니다.

~~~powershell
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
~~~

관측 공간에는 물리적으로 임의의 위치와 속도 상한을 두지 않았습니다. 이 때문에 Gymnasium 환경 검사에서 무한대 관측 경계에 관한 권고 경고 2개가 표시되지만 검증 실패는 아닙니다.

## 에피소드 실험 방법

### 명목 초기조건 자유낙하

~~~powershell
python scripts/run_episode.py --nominal
~~~

### 초기조건과 고정 제어 명령 지정

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --throttle 0.5 --gimbal-deg 0
~~~

초기조건 순서는 x z vx vz theta omega mass입니다. 모두 SI 단위를 사용하며 theta와 omega는 각각 rad와 rad/s입니다. --gimbal-deg만 degree를 받습니다.

--nominal과 --initial-state를 모두 생략하면 설정 파일의 범위에서 초기조건을 추출합니다. --seed의 기본값은 42입니다. 현재 입력 명령은 에피소드 내내 일정하므로 이 실험은 물리 모델 확인용이며 자동착륙 제어가 아닙니다.

### 기록과 그림 저장

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --throttle 0.5 --gimbal-deg 0 --output artifacts/episode.json --plot artifacts/episode.png --animation artifacts/episode.gif
~~~

출력 폴더는 자동으로 생성됩니다. 실험 결과를 실수로 잃지 않도록 기존 파일은 덮어쓰지 않습니다. GIF는 바로 저장할 수 있고 MP4에는 FFmpeg가 필요합니다.

### 저장 기록 재생

~~~powershell
python scripts/run_episode.py --replay artifacts/episode.json --plot artifacts/replay.png --animation artifacts/replay.gif
python scripts/run_episode.py --nominal --output artifacts/episode.npz
~~~

재생은 저장된 상태를 직접 사용하므로 물리 시뮬레이션을 다시 수행하지 않습니다. 애니메이션에는 궤적, 로켓 자세, 추력 방향, 착륙 목표, 스로틀, 짐벌, 질량과 종료 결과가 표시됩니다.

## Python API 실험

~~~python
from powered_landing_guidance import load_config
from powered_landing_guidance.envs import RocketLandingEnv

env = RocketLandingEnv(load_config("configs/default.yaml"))
state, info = env.reset(seed=42)

while True:
    state, reward, terminated, truncated, info = env.step([0.0, 0.0])
    if terminated or truncated:
        print(info)
        break

env.close()
~~~

초기조건과 행동열, 설정이 같으면 같은 궤적과 결과가 생성됩니다. 제어기를 비교할 때에는 같은 초기조건과 seed를 사용해야 합니다.

## 결과 판정

지면 접촉 직전 상태가 설정된 위치, 속도, 자세, 각속도 한계 안에 있으면 성공입니다.

| 결과 | 의미 |
|---|---|
| success | 모든 착륙 한계 만족 |
| hard_landing | 위치와 자세는 맞지만 수평 또는 수직 속도 초과 |
| crash | 위치, 자세 또는 각속도 한계 초과 |
| fuel_depletion | 공중에서 건조 질량 도달 |
| timeout | 최대 시뮬레이션 시간 도달 |

기본 성공 한계는 수평 위치 1 m, 수평 속도 1 m/s, 수직 속도 2 m/s, 자세 5 deg, 각속도 5 deg/s입니다. 이 기준은 시뮬레이션의 초기 판정값이며 실제 기체의 구조 안전성을 보증하지 않습니다.

## 파일 구성

- src/powered_landing_guidance/: 상태, 설정, 동역학, 환경, 기록과 시각화
- scripts/run_episode.py: 에피소드 실행과 기록 재생
- scripts/verify_week1.py: 1주차 통합 검증
- tests/: 회귀 및 경계 조건 테스트
- configs/default.yaml: 물리, 초기조건과 착륙 기준
- docs/spec.md: 좌표계, 단위, API와 기록 형식 명세
- docs/experiment_log.md: 일차별 구현과 검증 기록

생성한 로그, 그림과 영상은 artifacts/에 저장하며 Git에는 포함하지 않습니다. 일차별로 보존할 실험 결과는 docs/experiment_log.md 한 파일에 이어서 기록해 파일 수를 최소화합니다.

## 이후 로드맵

- 2주차: suicide-burn과 PD/PID 기준 제어기
- 3주차: CasADi와 IPOPT 기반 최적제어 교사
- 4주차: 교사 데이터셋과 Behavior Cloning
- 5주차: DAgger 기반 폐루프 정책 강화
- 6주차: 바람, 센서 오차, 엔진 오차와 Monte Carlo 평가
- 7주차: 결과 보고서, 비교 시각화와 조작 화면

## 현재 한계

현재는 착륙 직전의 동력 하강만 다룹니다. 발사, 상승, 재진입과 귀환 비행은 범위 밖입니다. 공기저항, 바람, 센서 잡음, 엔진 지연은 아직 운동방정식에 적용하지 않았습니다. 관성모멘트와 엔진 레버암은 고정값이고 지면 접촉은 로켓의 점 위치로 판정합니다. 실제 하드웨어 설계나 안전 판단에 바로 사용할 수 있는 모델이 아닙니다.

모델 규약은 [모델과 환경 명세](docs/spec.md), 실제 작업 결과는 [실험 기록](docs/experiment_log.md)에서 확인할 수 있습니다.
