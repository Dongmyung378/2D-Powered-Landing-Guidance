# 2D 로켓 착륙 모델 명세

이 문서는 시뮬레이터, 제어기, 데이터셋, 학습 정책이 공유하는 상태·제어·좌표계·단위 규격을 정의합니다. 구현 중 부호가 애매할 때에는 코드보다 이 문서를 우선 기준으로 사용합니다.

## 1. 모델 범위

- 2D planar rigid body
- 병진 자유도: `x`, `z`
- 회전 자유도: `theta`
- 가변 질량: `mass`
- 제어 입력: throttle과 추력 벡터 gimbal
- 이후 단계에서 중력, 추력, 질량 감소, 공기저항, 수평 바람, 지면 접촉을 구현

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

Day 2 병진 운동방정식은 다음과 같습니다.

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

## 8. Day 2 검증 기준

- 상태와 제어 순서를 코드 상수와 대조
- 자유낙하 수치 결과를 `z(t) = z0 + vz0*t - 0.5*g*t^2`와 비교
- 자유낙하 수치 결과를 `vz(t) = vz0 - g*t`와 비교
- 수직 추력 결과를 일정 가속도 해석해와 비교
- 수직 추력에서 `x`, `vx`, `theta`, `omega`, `mass`가 의도대로 유지되는지 확인
- Euler의 시간 간격을 줄였을 때 해석해 오차가 감소하는지 확인
- RK4가 일정 가속도 해석해와 수치 정밀도 범위에서 일치하는지 확인

## 9. Day 3 검증 기준

- 설정의 `gimbal_limit_deg`가 내부에서 정확히 radian으로 변환됨
- 범위를 벗어난 throttle과 gimbal 명령이 안전하게 제한됨
- 잘못된 형상이나 NaN을 포함한 제어 명령이 거부됨
- 양의 짐벌에서 음의 토크, 음의 짐벌에서 양의 토크가 발생함
- throttle 또는 gimbal이 0이면 추력 토크가 0임
- 각가속도가 `torque / moment_of_inertia`와 일치함
- 양·음 짐벌 회전 결과가 일정 각가속도 해석해와 일치함
- 내부 상태와 동역학 계산에서 각도 단위로 radian만 사용함

## 10. Day 4 검증 기준

- 질량 유량이 `thrust / (specific_impulse * standard_gravity)`와 일치함
- 질량 유량이 throttle에 비례함
- 연료가 남아 있는 동안 시간에 따른 질량이 해석해와 일치함
- throttle이 0이면 질량이 감소하지 않음
- 건조 질량에서 추력, 토크와 질량 유량이 모두 0임
- 초기 질량이 건조 질량보다 작으면 시뮬레이션을 거부함
- 한 적분 스텝 중간에 연료가 소진되어도 질량이 건조 질량 아래로 내려가지 않음
- 소진 이후 운동이 무추력 탄도 운동과 일치함

## 11. Day 5 환경 인터페이스

`RocketLandingEnv`는 이 물리 모델을 사용해 seed 기반 reset, 매 스텝 제어,
지면 접촉·연료 고갈·시간초과의 종료 판정과 JSON 로그를 제공합니다.
환경의 fuel_depletion 종료 규칙은 소진 후에도 운동을 계속 계산하는 물리 함수와 별개입니다.
자세한 정의와 사용법은 [환경 명세](env_spec.md)를 참조합니다.
