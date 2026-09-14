# 모델과 환경 명세

[English specification](spec.md)

이 문서는 시뮬레이터, 제어기, 데이터셋, 학습 정책이 공유하는 상태·제어·좌표계·단위 규격을 정의합니다. 부호와 단위를 변경할 때에는 코드와 이 명세를 함께 수정합니다.

## 1. 모델 범위

- 2D planar rigid body
- 병진 자유도: `x`, `z`
- 회전 자유도: `theta`
- 가변 질량: `mass`
- 제어 입력: throttle과 추력 벡터 gimbal
- 중력, 추력, 연료 소모와 점 위치 기준 지면 접촉을 구현
- 공기저항과 바람은 아직 미적용; 관성모멘트와 레버암은 고정

## 2. 관성 좌표계

```text
                       +z (위)
                        ^
                        |       body axis
                        |      /   theta > 0
                        |     /
                        |    /
                  COM   o----------------> +x (오른쪽)
                        |
                        |
                   ground: z = 0
```

- 원점: 착륙 목표 지점의 지면
- `+x`: 화면 오른쪽
- `+z`: 위쪽
- 중력 가속도 벡터: `(0, -g)`
- 지면: `z = ground_z_m`, 기본값은 `0 m`

## 3. 각도와 부호

- `theta = 0`: 로켓의 길이 방향 축이 `+z`와 일치한 수직 자세
- `theta > 0`: 로켓 기수가 `+x` 쪽으로 기울어지는 시계방향 회전
- `omega = d(theta)/dt`: `theta`와 같은 방향이 양수
- `gimbal_angle = 0`: 추력 벡터가 로켓 길이 방향 축과 일치
- `gimbal_angle > 0`: **추력 벡터**가 로켓 축에서 `+theta` 방향으로 회전
- `gimbal_angle`은 노즐 하드웨어의 기계적 편향각이 아니라 힘 벡터의 편향각

추력 벡터의 관성계 방향각은 `theta + gimbal_angle`입니다. 따라서 병진 동역학에서 사용할 추력 성분은 다음 규약을 따릅니다.

```text
thrust_x = thrust * sin(theta + gimbal_angle)
thrust_z = thrust * cos(theta + gimbal_angle)
```

운동방정식은 다음과 같습니다.

```text
x_dot     = vx
z_dot     = vz
vx_dot    = thrust_x / mass
vz_dot    = thrust_z / mass - g
theta_dot = omega
omega_dot = torque / moment_of_inertia
mass_dot  = -thrust / (specific_impulse * standard_gravity)
```

질량이 건조 질량에 도달하면 적용 추력, 토크와 질량 유량을 모두 0으로 설정합니다.

```text
if mass <= dry_mass:
    applied_thrust = 0
    torque = 0
    mass_dot = 0
```

`standard_gravity`는 비추력 `s`를 유효 배기속도 `m/s`로 변환하기 위한 표준중력이며, 시뮬레이션 환경에 적용하는 `gravity`와 구분합니다.

엔진 작용점은 질량중심 아래의 로켓 길이 방향 축 위에 있다고 가정합니다. 엔진 레버암의 길이를 `l`, 추력을 `T`, 짐벌각을 `delta`라고 하면 현재 부호 규약에서 토크는 다음과 같습니다.

```text
torque    = -l * T * sin(delta)
omega_dot = torque / moment_of_inertia
```

따라서 양의 짐벌은 음의 자세 토크와 각가속도를 만들고, 음의 짐벌은 양의 자세 토크와 각가속도를 만듭니다.

## 4. 상태 벡터

상태 배열의 순서는 변경하지 않습니다.

| 인덱스 | 이름 | 의미 | 내부 단위 |
|---:|---|---|---|
| 0 | `x` | 수평 위치 | m |
| 1 | `z` | 수직 위치 | m |
| 2 | `vx` | 수평 속도 | m/s |
| 3 | `vz` | 수직 속도 | m/s |
| 4 | `theta` | 수직축 기준 자세각 | rad |
| 5 | `omega` | 각속도 | rad/s |
| 6 | `mass` | 현재 질량 | kg |

```text
state = [x, z, vx, vz, theta, omega, mass]
```

## 5. 제어 벡터

제어 배열의 순서는 변경하지 않습니다.

| 인덱스 | 이름 | 의미 | 내부 단위/범위 |
|---:|---|---|---|
| 0 | `throttle` | 최대 추력 대비 명령 비율 | dimensionless, `[0, 1]` |
| 1 | `gimbal_angle` | 로켓 축 기준 추력 벡터 편향 | rad |

```text
control = [throttle, gimbal_angle]
```

설정 파일에서 사람이 읽기 쉬운 각도 제한과 초기값은 `_deg` 접미사를 사용합니다. 시뮬레이터 경계에서 한 번만 radian으로 변환하고, 내부 계산·로그·학습 데이터에는 radian을 사용합니다.

제어기가 생성하는 원시 명령은 물리 모델에 들어가기 전에 다음과 같이 제한합니다.

```text
applied_throttle = clip(commanded_throttle, throttle_min, throttle_max)
applied_gimbal   = clip(commanded_gimbal, -gimbal_limit, +gimbal_limit)
```

## 6. 기본 제한과 불변조건

- `mass >= dry_mass`
- `0 <= throttle <= 1`
- `abs(gimbal_angle) <= gimbal_limit`
- 모든 상태·제어 값은 유한한 실수여야 함
- 설정의 `dt_s`, `max_time_s`, `gravity_m_s2`, 질량, 추력은 양수여야 함
- 초기 질량은 건조 질량보다 크거나 같아야 함

## 7. 착륙 성공 기준

지면 접촉 직전의 상태가 아래 조건을 모두 만족할 때 성공으로 판정합니다.

| 기준 | 초기 한계값 |
|---|---:|
| 수평 위치 오차 | `abs(x) <= 1.0 m` |
| 수평 속도 | `abs(vx) <= 1.0 m/s` |
| 수직 속도 | `abs(vz) <= 2.0 m/s` |
| 자세각 | `abs(theta) <= 5 deg` |
| 각속도 | `abs(omega) <= 5 deg/s` |
| 질량 | `mass >= dry_mass` |

이 값들은 초기 권장값이며 `configs/default.yaml`에서 관리합니다. 이후 평가 시 모든 제어기에 동일한 seed와 초기조건을 적용합니다.


## API

`RocketLandingEnv(config)`에 `load_config`로 읽은 dict를 전달합니다. 생성자는 설정을 복사하고 초기 상태, sampling 범위, 단위, 적분기, 착륙 한계값을 검증합니다. 내부 동역학은 `simulate_planar`를 사용합니다.

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

## 기록 재생과 시각화

`save_episode_data`는 같은 에피소드 기록을 JSON 또는 압축 NPZ로 저장합니다. NPZ에는 원본 JSON과 검증용 `times_s`, `states`, `actions`, `thrust_n` 배열을 함께 저장하며 객체 pickle은 사용하지 않습니다. `actions`는 각 step에 적용한 명령이고 나머지 배열은 초기 프레임을 포함합니다. `load_episode_data`는 배열과 원본 기록의 일치 여부를 확인합니다. 두 형식 모두 저장된 상태를 직접 읽으므로 물리 모델을 다시 실행하지 않습니다.

시계열 그래프는 위치, 속도, 자세·각속도, 질량, throttle, gimbal을 표시합니다. 2D 애니메이션은 x-z 궤적, 로켓 자세, 추력 벡터, 지면과 x=0 착륙 목표를 표시합니다. 로켓 도형은 부호 확인용이며 실제 크기나 착륙 다리 형상을 나타내지 않습니다. GIF는 기본 지원하고 MP4는 FFmpeg가 있을 때 지원합니다. 출력 파일은 기존 파일을 덮어쓰지 않습니다.

## 수직 Suicide burn 기준선

SuicideBurnController는 수평 위치와 자세 오차가 없는 수직 하강만 다룹니다. 첫 번째 점화 고도 계산은 연소 중 질량을 현재 질량으로 고정합니다.

~~~text
net_deceleration = max_thrust / mass - gravity
stopping_distance = (downward_speed^2 - target_speed^2) / (2 * net_deceleration)
~~~

estimate_suicide_burn은 이 상수 질량 계산을 그대로 제공합니다. 실제 SuicideBurnController는 연료 소모에 따라 질량과 가속도가 변하는 수직 운동 해석해로 점화 고도를 보정합니다. 필요한 속도 감소에 비해 연료가 부족하거나 최대 추력이 중력을 이기지 못하면 infeasible로 판정하고 즉시 최대 추력을 명령합니다.

이상적인 점화가 0.02초 제어 주기 안에 있으면 첫 연소 스텝의 스로틀 비율을 조정해 이산 시간 오차를 줄입니다. 점화 이후에는 최대 추력을 유지합니다. 실제 점화 고도가 계산값보다 높으면 early_burn, 낮으면 late_burn, 허용오차 안이면 on_time으로 분류합니다.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller suicide-burn
~~~

이 기준선은 바람, 수평 오차, 기체 기울기와 센서 또는 엔진 지연을 보정하지 않습니다. 접지 속도 2 m/s는 현재 환경의 성공 판정 한계이며 실제 구조 안전 속도가 아닙니다.

## 수직속도 PID 기준선

VerticalVelocityPIDController는 고도에 따른 목표 수직속도를 만들고 throttle PID로 추종합니다. 수평 위치, 수평 속도, 자세와 각속도가 0인 수직 하강 조건을 대상으로 합니다.

~~~text
speed_limit = sqrt(touchdown_speed^2 + 2 * profile_deceleration * altitude)
target_vz = -min(max_descent_speed, speed_limit)
error = target_vz - current_vz
~~~

기본 profile은 접지 목표속도 1 m/s, 최대 하강속도 30 m/s, 감속도 3 m/s²입니다. Profile의 감속도와 중력을 보상하는 feed-forward throttle에 PID 보정값을 더합니다.

~~~text
feed_forward = mass * (gravity + target_acceleration) / max_thrust
raw_throttle = feed_forward + Kp * error + Ki * integral + Kd * error_rate
throttle = clip(raw_throttle, throttle_min, throttle_max)
~~~

기본 이득은 Kp=0.08, Ki=0.001, Kd=0.01입니다. 적분 오차는 ±10 m로 제한합니다. 출력이 상한에서 포화되고 오차가 양수이거나, 하한에서 포화되고 오차가 음수이면 해당 스텝의 적분을 취소합니다. reset은 적분값, 이전 오차와 마지막 명령 정보를 모두 초기화합니다.

평가 초기조건은 고도 80-120 m, 수직속도 -25~-15 m/s, 질량 950-1000 kg의 독립 균등분포입니다. 20260910 seed로 생성한 100개 조건에서 100회 모두 성공했으며 평균 접지 속도는 1.1010 m/s, 95백분위는 1.1938 m/s, 평균 연료 사용량은 37.4923 kg입니다.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid
python scripts/sweep_velocity_gains.py --episodes 100 --seed 20260910
~~~

이 결과는 현재 설정 범위의 결정론적 수직·무풍 시뮬레이션에 한정됩니다. 수평 위치와 자세 제어는 현재 범위 밖이며 바람, 모델 오차, 센서 잡음과 구동기 지연은 이후 강건성 단계에서 평가합니다.

## 수평 위치·자세 제어기

`HorizontalAttitudeController`는 학습을 사용하지 않는 직렬 제어기입니다. 현재 상태와 별도 수직 제어기가 계산한 기본 throttle을 입력받습니다. 이 인터페이스로 수평·자세 로직을 통합 제어기 단계 전까지 독립적으로 유지합니다.

외부 루프는 위치와 속도 오차로 수평 가속도를 계산합니다.

~~~text
raw_ax = Kp_x * (target_x - x) - Kd_x * vx
target_ax = clip(raw_ax, -max_ax, max_ax)
target_theta = clip(atan2(target_ax, gravity), -max_tilt, max_tilt)
~~~

기본 외부 루프 이득은 `Kp_x=0.05 s^-2`, `Kd_x=0.45 s^-1`입니다. 수평 가속도는 2.5 m/s², 목표 기울기는 15도로 제한합니다.

내부 루프는 목표 각가속도를 계산하고 문서의 토크 부호 규약을 역으로 풉니다. 양의 gimbal은 음의 각가속도를 만듭니다.

~~~text
target_alpha = Kp_theta * (target_theta - theta) - Kd_theta * omega
sin(gimbal) = -target_alpha * inertia / (lever_arm * thrust)
~~~

기본 내부 루프 이득은 `Kp_theta=4.0 s^-2`, `Kd_theta=3.0 s^-1`입니다. 역계산 값은 삼각함수 정의역과 기체의 15도 gimbal 제한에 맞춰 자릅니다. 기본 throttle이 0이면 추력 편향으로 토크를 만들 수 없으므로 gimbal은 0입니다.

기울어진 추력의 수직 성분은 `cos(theta + gimbal)`만큼 줄어듭니다. Coupling 보상을 켜면 출력 throttle의 수직 성분이 요청된 기본 throttle과 같아지도록 반복 계산하며, 최종 출력에는 0-1 구동기 제한을 적용합니다.

재현 실험은 고도 100 m의 무풍 조건에서 100개 초기조건을 각각 8초 동안 실행합니다. 초기 수평 오차는 -20~20 m 범위에서 최소 5 m이며 수평 속도, 자세, 각속도와 질량은 설정 범위에서 표본화합니다. Seed 20260914에서 100개 모두 정상 완료했고 절대 수평 오차가 감소했습니다. 평균 오차는 12.4712 m에서 5.4894 m로 줄었고 최대 목표 기울기는 9.7647도, 최대 gimbal은 7.0393도, 최대 고도 편차는 0.0149 m였습니다.

~~~powershell
python scripts/evaluate_horizontal_control.py --episodes 100 --seed 20260914
~~~

이 결과는 무풍에서 착륙장 방향으로 수렴함을 검증합니다. 아직 접지 성공을 의미하지 않으며 수직 throttle과 수평 gimbal 명령은 통합 제어기 단계 전까지 분리되어 있습니다.
