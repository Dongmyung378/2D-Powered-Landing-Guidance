# Powered Landing Guidance

2차원 재사용 로켓의 동력 착륙 시뮬레이션 프로젝트입니다. 현재 6일차까지의 물리 모델, Gymnasium 환경, 종료 판정, 기록 저장과 시각화가 구현되어 있습니다. 자동착륙 제어기는 아직 없습니다.

## 설치와 검증

Python 3.12.7 환경에서 프로젝트 루트에서 실행합니다.

```powershell
python --version
python -m pip install -e ".[dev]"
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
```

테스트는 자유낙하·가변 질량 해석해, 회전 부호, 연료 차단, 접촉 시각, 종료 우선순위, seed 재현성과 로그 재생을 확인합니다. Gymnasium 검사기의 무한대 경고 2개는 관측 공간에 임의의 상한을 두지 않아 발생합니다.

## 실험 실행

기본 초기조건에서 무추력 낙하:

```powershell
python scripts/run_episode.py --nominal
```

초기조건과 고정 추력을 지정하고 JSON 기록, 그래프와 애니메이션 저장:

```powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --throttle 0.5 --gimbal-deg 0 --output artifacts/episode.json --plot artifacts/episode.png --animation artifacts/episode.gif
```

초기조건 순서는 `x z vx vz theta omega mass`이며 SI 단위입니다. 자세각과 각속도는 rad, rad/s이고, CLI의 `--gimbal-deg`만 degree를 받습니다. `--nominal` 또는 `--initial-state`를 생략하면 YAML 범위에서 초기조건을 추출합니다. `--seed` 기본값은 42입니다. 저장 폴더는 자동 생성하며 기존 로그는 덮어쓰지 않습니다.

고정 명령을 적용하는 물리 실험이며 착륙 제어기가 아닙니다.

저장된 JSON 또는 NPZ는 물리를 다시 계산하지 않고 그래프와 애니메이션으로 재생할 수 있습니다.

```powershell
python scripts/run_episode.py --replay artifacts/episode.json --plot artifacts/replay.png --animation artifacts/replay.gif
python scripts/run_episode.py --nominal --output artifacts/episode.npz
```

GIF는 별도 프로그램 없이 생성할 수 있습니다. MP4 저장에는 FFmpeg가 필요합니다. 애니메이션에는 x-z 궤적, 로켓 자세, 추력 방향, 착륙 목표, throttle, gimbal, 질량과 종료 원인이 표시됩니다.

## Python API

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
env.close()
```

행동은 `[throttle, gimbal_angle]`이며 짐벌각은 radian입니다. 상세 규칙은 [명세](docs/spec.md), 수치와 제한값은 [설정](configs/default.yaml)에 있습니다.

## 파일 구성

- `src/powered_landing_guidance/`: 상태, 설정, 동역학, 환경, 기록·시각화
- `tests/`: 모델·동역학·환경 테스트
- `scripts/run_episode.py`: 실험 실행
- `configs/default.yaml`, `docs/spec.md`: 설정과 명세

날짜별 검증은 테스트로 통합했습니다. 미구현 기능의 빈 패키지는 두지 않습니다. 생성물은 `artifacts/`에 저장하며 Git에서 제외합니다.

## 이후 계획과 한계

PID, 최적제어 teacher, Behavior Cloning, DAgger를 구현한 뒤 동일한 초기조건과 외란에서 비교할 예정입니다. 현재 관성모멘트와 레버암은 고정이고, 공기저항·바람·센서 노이즈는 적용하지 않습니다. YAML의 항력·바람 항목은 후속 구현용입니다. 접촉은 점 위치 기준이며 실제 하드웨어 실험용 모델이 아닙니다.

완성형 입력 화면과 제어기 비교 애니메이션은 기존 계획대로 7주차 마지막에 제작합니다.

- 43-46일차: 코드 정리, 결과 그래프, 보고서와 README
- 47일차: 초기조건·외란·seed·제어기 선택 화면
- 48일차: 2D 로켓·궤적·추력 벡터와 상태 그래프 연결
- 49일차: 재생·일시정지·초기화, 성공·실패 시나리오 검증과 데모 저장

화면에서 같은 seed로 결과를 재현하고, 제어기별 착륙 오차·속도·자세·사용 연료를 비교하는 것이 완료 기준입니다. UI 의존성은 해당 단계에서 결정합니다.
