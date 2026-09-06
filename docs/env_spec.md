# Day 5: Gymnasium 환경·종료·로그 명세

## API

`RocketLandingEnv(config)`에 `load_config`로 읽은 dict를 전달합니다. 생성자는 설정을 복사하고 초기 상태, sampling 범위, 단위, 적분기, 착륙 한계값을 검증합니다. 내부 동역학은 기존 `simulate_planar`를 사용합니다.

- `reset(seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `episode_log`: JSON 직렬화 가능한 전체 기록의 복사본
- `save_episode(path)`: 기록 저장, 기존 파일 덮어쓰기 거부; 부모 폴더는 호출자가 준비

reset 이전 또는 종료 이후의 step은 오류입니다. reset은 새 에피소드를 시작하므로 보관하려는 로그는 그 전에 저장해야 합니다. NaN/Inf, 잘못된 행동 형상, 지면 아래 초기조건, 건조 질량 미만 초기조건은 거부합니다.

관측은 float64 7상태, 행동은 float64 `[throttle, gimbal_angle]`이고 내부 단위는 SI/radian입니다. position/velocity에 임의의 유한 상한을 두지 않아 Gymnasium checker가 관측 공간의 infinity에 관한 권고 경고를 표시할 수 있습니다. 정상 범위에서의 NaN/Inf 관측은 허용하지 않습니다.

## 초기조건

기본 reset은 `initial_state_sampling`의 각 범위에서 독립 균등분포로 값을 추출합니다. YAML 단위는 `initial_state`와 같으며 theta와 omega만 degree에서 radian으로 변환합니다. 범위가 생략된 필드는 기본 초기값을 사용합니다. sampling 섹션 전체가 없으면 기본 초기값을 사용합니다.

- `options={"randomize": False}`: 명목 초기조건
- `options={"initial_state": [...]}`: 정확한 SI/radian 7상태; randomize보다 우선
- `reset(seed=n)`: 난수 발생기를 n으로 초기화
- `reset(seed=None)`: 난수 발생기의 기존 상태를 이어 사용

같은 seed와 동일 행동열은 같은 기록을 생성합니다. action_space의 sample을 사용하는 호출자는 action_space.seed도 별도로 지정합니다. 기록된 초기 상태와 행동열을 사용하면 seed 없이도 같은 환경 설정에서 재계산할 수 있습니다.

## 종료 판정

지면은 현재 모델의 점 위치 z 기준입니다. 첫 접촉의 상태를 판정에 사용하고 지면 이하 관측을 반환하지 않습니다. theta는 판정 시에만 `atan2(sin(theta), cos(theta))`로 한 바퀴 주기를 반영합니다. 내부 상태와 기록은 unwrapped angle을 유지합니다.

| outcome | 판정 |
|---|---|
| running | 진행 중 |
| success | 접촉 위치, vx, vz, theta, omega의 절댓값이 YAML 성공 한계값 이하 |
| hard_landing | 위치, theta, omega는 한계 이내이나 vx 또는 vz 초과 |
| crash | 접촉 시 위치, theta 또는 omega 한계 초과 |
| fuel_depletion | 공중에서 건조 질량 도달 |
| timeout | 최대 시뮬레이션 시간 도달 |

시작 상태가 지면 또는 건조 질량이면 첫 step에서 t=0 종료 판정합니다. 접촉 > 연료 고갈 > 시간초과 순서로 동시 사건을 판정합니다. 순차적으로 발생하는 사건은 먼저 발생한 시각에서 멈춥니다. 연료 고갈 시 환경은 즉시 종료하므로 이후 탄도 착륙 가능성을 성공으로 간주하지 않습니다. 물리 함수의 연료 소진 후 탄도 동작은 그대로 유지됩니다.

성공 및 세 물리적 실패는 terminated=True이고 시간초과만 truncated=True입니다. 성공 보상은 +1, 물리적 실패는 -1, 진행 중과 시간초과는 0입니다. 미래 학습의 최종 reward는 아직 설계하지 않았습니다.

## 스텝 내 사건 계산

한 action은 simulation.dt_s 동안 일정하게 유지됩니다. 내부 적분 간격은 최대 0.02초이며 더 작은 설정 dt는 그대로 사용합니다. 마지막 스텝은 최대 시간에 맞춰 단축합니다. 질량 유량으로 계산한 연료 소진 시점에서도 내부 구간을 단축합니다.

각 내부 구간에서 지면 교차를 검사하고 Brent root search로 접촉 시각을 찾습니다. 양 끝 고도가 모두 양수여도 구간 중간에 하강 후 상승할 수 있으므로 가속도 상한으로 가능한 구간을 선별하고 고도의 최소점을 검색합니다. root search 허용시간 오차는 1e-12초이나 최종 물리 정확도는 선택한 Euler/RK4와 적분 간격에 의해 제한됩니다.

하강 중인 구간 끝점의 고도가 지면보다 부동소수점 정밀도 수준으로 높게 남는 경우에는 접촉으로 취급합니다. 고도 허용오차는 `64 * eps * max(1, |ground_z|, |z_start|, |z_end|)` m이며, 정확히 시간제한과 겹치는 접촉이 반올림 오차로 timeout으로 바뀌는 것을 방지합니다.

## 에피소드 JSON schema_version=1

| 필드 | 내용 |
|---|---|
| config | 실행에 사용한 설정 전체 |
| seed_argument | 해당 reset에 전달한 seed, 생략 시 null |
| rng_state_after_reset | 초기조건 생성 직후 난수 상태 |
| state_names / state_units | 상태 순서 및 단위 |
| action_names / action_units | 행동 순서 및 단위 |
| initial_state | t=0의 SI 7상태 |
| steps | 각 step의 결과 목록 |
| outcome | 최신 종료 사유, 미종료 시 running |

각 steps 원소에는 index, time_s, state, commanded_action, clipped_action, thrust_start_n, thrust_end_n, fuel_used_kg, reward, terminated, truncated, outcome, is_success가 들어갑니다. fuel_used_kg는 초기 질량과 현재 질량의 차이입니다. clipped_action은 액추에이터 제한 명령이며 실제 연료 차단 여부는 thrust_start_n/end_n으로 구분합니다. 상태·반환 info·로그를 외부에서 변경해도 환경 내부 기록이 바뀌지 않도록 복사합니다.

API 참고: [Gymnasium Env](https://gymnasium.farama.org/api/env/), [환경 검사기](https://gymnasium.farama.org/api/utils/).
