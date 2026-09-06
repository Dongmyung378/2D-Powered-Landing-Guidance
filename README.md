# Powered Landing Guidance

2차원 재사용 로켓의 동력 착륙(powered landing)을 위한 물리 시뮬레이터와 제어·학습 파이프라인을 구축하는 프로젝트입니다. 최종적으로 PID, 최적제어 teacher, Behavior Cloning, DAgger 정책을 동일한 초기조건과 외란 조건에서 비교합니다.

현재 저장소는 **Day 5: Gymnasium 환경과 종료 판정**까지 구현된 상태입니다. 초기조건을 입력하거나 seed로 생성하고, 매 스텝 제어 명령을 적용해 종료 사유와 에피소드 로그를 확인할 수 있습니다.

## 개발 환경

- Python 3.12.7
- NumPy, SciPy, PyYAML
- Gymnasium (5일차부터 필수 의존성)
- pytest, Ruff

현재 Python 3.12.7 환경을 사용하는 경우 다음 명령으로 설치합니다.

```powershell
python --version
python -m pip install -e ".[dev]"
```

`python --version`의 출력이 `Python 3.12.7`인지 확인합니다.

## 검증 명령

```powershell
python -m pytest
python -m ruff check .
python scripts/verify_day1.py
python scripts/verify_day2.py
python scripts/verify_day3.py
python scripts/verify_day4.py
python scripts/verify_day5.py
```

## 디렉터리 구조

```text
configs/                         초기조건과 시뮬레이션 설정
docs/                            모델·좌표계·단위 명세
src/powered_landing_guidance/
  dynamics/                      병진·회전·질량 동역학
  envs/                          Gymnasium 환경
  controllers/                   PD/PID 기준 제어기
  optimal_control/               최적제어 teacher
  policies/                      BC·DAgger 정책
  data/                          데이터셋 처리
  evaluation/                    Monte Carlo 평가
  visualization/                 궤적·상태·애니메이션
tests/                           단위·통합 테스트
scripts/                         실행·검증 스크립트
reports/                         결과 요약
artifacts/                       생성된 모델·그래프·로그
```

## 핵심 규격

- 상태: `x, z, vx, vz, theta, omega, mass`
- 제어: `throttle, gimbal_angle`
- 좌표계: `+x`는 오른쪽, `+z`는 위쪽
- 내부 각도 단위: radian

좌표계와 부호, 단위, 착륙 성공 조건의 상세 정의는 [모델 명세](docs/model_spec.md)를 참조합니다. 초기 수치와 제한값은 [기본 설정](configs/default.yaml)에 모아 두었습니다.

## Day 2 병진 동역학

`src/powered_landing_guidance/dynamics/`에는 다음 기능이 구현되어 있습니다.

- 현재 자세와 짐벌각으로 추력 벡터 계산
- `F = ma`에 현재 질량 반영
- `+z` 반대 방향으로 중력 반영
- Euler 및 RK4 단일 스텝 적분
- 마지막 구간이 짧아도 종료 시각에 정확히 도달하는 고정 간격 시뮬레이션

`theta_dot = omega`와 추력 토크에 의한 `omega_dot`을 반영합니다. Day 4부터 추력을 사용하는 동안 질량도 시간에 따라 감소합니다.

## Day 3 회전 및 액추에이터 제한

- 관성모멘트와 엔진 레버암을 사용한 추력 토크 계산
- `omega_dot = torque / moment_of_inertia` 회전 동역학
- throttle 명령을 설정된 최소·최대 범위로 제한
- gimbal 명령을 `±gimbal_limit` 범위로 제한
- 설정 파일의 degree 값을 로딩 시 radian으로 변환
- 양·음 짐벌의 토크 부호와 회전 방향 검증

현재 좌표계에서 양의 짐벌은 추력 벡터를 `+theta` 방향으로 기울입니다. 엔진 작용점이 질량중심 아래에 있으므로 양의 짐벌은 음의 자세 토크를 만들고, 음의 짐벌은 양의 자세 토크를 만듭니다.

## Day 4 연료 소모와 건조 질량

- `mass_flow = thrust / (specific_impulse × standard_gravity)` 적용
- throttle에 비례하는 연료 소비
- 질량이 건조 질량에 도달하면 추력·토크·질량 유량을 모두 0으로 차단
- 적분 스텝 중간에 연료가 소진되어도 해당 시각에서 powered/coast 구간을 분할
- 수치 오차가 있어도 질량을 건조 질량 아래로 내리지 않음
- 연료 소모 중에는 현재 질량을 사용해 병진 가속도를 계산

기본 설정의 최대 추력 `20,000 N`, 비추력 `250 s`에서는 최대 throttle의 질량 유량이 약 `8.1577 kg/s`입니다. 표준중력은 비추력 단위 변환에만 사용하고, 환경 중력과 별도로 설정합니다.

## Day 5 환경 사용법

프로젝트 루트에서 실행하는 예시입니다. 기존 설치 환경은 `python -m pip install -e ".[dev]"`로 의존성을 갱신합니다.

```python
from powered_landing_guidance import load_config
from powered_landing_guidance.envs import RocketLandingEnv

env = RocketLandingEnv(load_config("configs/default.yaml"))
obs, info = env.reset(seed=42)
while True:
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    if terminated or truncated:
        print(info)
        break
env.save_episode("artifacts/example_episode.json")  # 기존 파일은 덮어쓰지 않음
env.close()
```

이 예시의 무추력 명령은 환경 사용 예제이며 자동착륙 제어기가 아닙니다.

- 관측값: `[x, z, vx, vz, theta, omega, mass]`, float64, SI 단위
- 행동: `[throttle, gimbal_angle]`, 짐벌 단위 radian; 범위를 벗어난 유한값은 clipping
- `reset(seed=42)`: YAML의 `initial_state_sampling` 범위에서 독립 균등 추출
- `reset(options={"randomize": False})`: YAML의 기본 초기조건 사용 (degree를 radian으로 변환)
- `reset(options={"initial_state": [0, 100, 0, -20, 0, 0, 1000]})`: SI/radian 상태 직접 지정
- 같은 seed와 같은 행동열을 입력하면 관측·보상·종료·로그가 재현됨
- `reset(seed=None)`은 기존 난수 흐름을 이어감; 무작위 행동은 `env.action_space.seed()`로 별도 관리

종료 사유는 `info["outcome"]`에 기록됩니다.

| 사유 | 기준 | Gymnasium 반환 |
|---|---|---|
| `success` | 접촉 직전 위치·속도·자세·각속도 모두 허용범위 내 | terminated |
| `hard_landing` | 지점·자세·각속도는 허용범위 내이나 속도 초과 | terminated |
| `crash` | 접촉 시 지점·자세·각속도 중 하나 이상 초과 | terminated |
| `fuel_depletion` | 공중에서 건조 질량에 도달 | terminated |
| `timeout` | 최대 시간 도달 | truncated |

접촉은 점 위치 `z = ground_z_m` 기준이며 착륙 다리 형상은 아직 모델링하지 않습니다. 먼저 발생한 사건에서 멈추고, 동시에 발생하면 접촉 > 연료 고갈 > 시간초과 순서입니다. 속도는 충돌 반응으로 0으로 바꾸지 않고 접촉 직전 값을 보존합니다. 환경은 연료 고갈을 실패로 끝내지만 물리 함수 `simulate_planar`는 기존처럼 소진 후 탄도 운동을 계속 계산할 수 있습니다.

보상은 성공 `+1`, 물리적 실패 `-1`, 진행·시간초과 `0`인 최소 규칙입니다. 학습용 보상 설계는 후속 작업입니다. 공기저항·바람·센서 노이즈는 현재 계산에 적용되지 않습니다. 애니메이션은 예정대로 7주차에 제작합니다.

다섯 종료 시나리오와 seed 재현성을 검증하고 로그를 저장하려면:

```powershell
python scripts/verify_day5.py --output-dir artifacts/day5_demo
```

이미 존재하는 출력 디렉터리는 거부합니다. 로그 필드와 이벤트 계산은 [Day 5 환경 명세](docs/env_spec.md)를 참조하세요.

## 7주차 최종 애니메이션 계획

완성형 애니메이션과 사용자 입력 화면은 물리 모델, 제어기, 학습 정책, Monte Carlo 평가가 모두 끝난 뒤 **Day 47~49**에 제작합니다. 이전 주차의 시각화는 부호와 궤적을 검증하기 위한 최소 그래프와 디버깅 출력으로 제한합니다.

- Day 43~46: 코드 정리, 결과 그래프, 보고서와 README 완성
- Day 47: 초기조건·외란·제어기 선택 입력 화면 제작
- Day 48: 2D 로켓 애니메이션과 실시간 상태·제어 그래프 연결
- Day 49: 전체 시나리오 smoke test, UI 보완, 최종 데모 저장

입력 화면에서는 초기 `x`, `z`, 수평·수직 속도, 자세각, 질량, 바람, seed와 제어기(PID/최적제어/BC/DAgger)를 선택할 수 있게 합니다. 실행 화면에는 로켓 자세, 궤적, 추력 벡터, 착륙 지점, 연료, 속도와 성공·hard landing·crash 판정을 표시합니다.

기본 구현 후보는 Streamlit과 Plotly이며, 실제 프레임 재생 성능을 확인한 뒤 필요하면 애니메이션 부분만 PySide6로 변경합니다. UI 관련 의존성은 핵심 시뮬레이터가 안정화되기 전에는 설치하지 않습니다.

완료 기준은 다음과 같습니다.

- 화면에서 조건을 입력하고 동일한 seed로 결과를 재현할 수 있음
- 제어기를 선택해 동일한 초기조건에서 결과를 비교할 수 있음
- 재생, 일시정지, 초기화가 동작함
- 착륙 결과와 위치 오차, 착륙 속도, 자세, 사용 연료가 표시됨
- 대표 성공·실패 시나리오가 끊김 없이 재생됨
