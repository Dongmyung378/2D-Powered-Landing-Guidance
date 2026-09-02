# Powered Landing Guidance

2차원 재사용 로켓의 동력 착륙(powered landing)을 위한 물리 시뮬레이터와 제어·학습 파이프라인을 구축하는 프로젝트입니다. 최종적으로 PID, 최적제어 teacher, Behavior Cloning, DAgger 정책을 동일한 초기조건과 외란 조건에서 비교합니다.

현재 저장소는 **Day 1: 프로젝트 규격과 저장소 구축**까지 완료된 상태입니다. 물리식 구현은 Day 2부터 진행합니다.

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
