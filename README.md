# Powered Landing Guidance

2차원 재사용 로켓의 동력 착륙(powered landing)을 위한 물리 시뮬레이터와 제어·학습 파이프라인을 구축하는 프로젝트입니다. 최종적으로 PID, 최적제어 teacher, Behavior Cloning, DAgger 정책을 동일한 초기조건과 외란 조건에서 비교합니다.

현재 저장소는 **Day 2: 병진 동역학 구현**까지 완료된 상태입니다. 중력, 추력 벡터, 현재 질량을 반영한 운동방정식과 Euler/RK4 고정 간격 적분기를 제공합니다.

## 개발 환경

- Python 3.12.7
- NumPy, SciPy, PyYAML
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

현재 단계에서는 질량, 각속도가 일정합니다. `theta_dot = omega`만 반영하며, 질량 감소와 회전 토크는 각각 Day 4와 Day 3에 추가합니다.

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
