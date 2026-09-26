# 평면 BC 데이터셋 카드

[English dataset card](dataset-card.md)

## 상태와 목적

`planar-bc-v1-final`은 프로젝트의 7상태 평면 동력 착륙 문제에서 Behavior Cloning policy를 학습하고 평가하기 위한 최종 offline dataset입니다. Train 1,007개, IID validation 100개와 hard OOD test 191개로 구성됩니다. 24일차에 더 넓은 난이도 편향 train 보강, marginal coverage 복구, train 전용 normalization과 최종 물리 shard 검증을 완료했습니다.

지도학습 mapping은 비종단 상태 하나에서 다음 구간에 적용할 최적제어 action을 예측하는 방식입니다. 종단 상태에는 action target이 없습니다. 이 dataset version은 동결된 `planar-teacher-v1` solver만 label 출처로 인정합니다.

## 사용 목적

- 현재 평면 상태에서 throttle과 gimbal을 예측하는 float32 student policy 학습
- IID validation split에서 architecture와 학습 설정 선택
- 의도적으로 더 어려운 OOD test split에서 일반화 측정
- 동결 Teacher·PID와 학습 policy의 성공률, 연료, 제약 위반과 평가시간 비교

이 dataset은 비행 안전, hardware 준비, 6자유도 성능, 외란 강건성 또는 문서화한 범위 밖 동작을 증명하지 않습니다.

## 상태와 제어 schema

저장하는 모든 물리값은 float64입니다. 학습 코드는 load와 검증이 끝난 뒤 batch만 float32로 변환할 수 있습니다.

| 종류 | 순서가 고정된 field | 단위 |
|---|---|---|
| 상태 | `x, z, vx, vz, theta, omega, mass` | `m, m, m/s, m/s, rad, rad/s, kg` |
| 제어 | `throttle, gimbal_angle` | `1, rad` |

채택 궤적마다 다음 항목을 기록합니다.

- 안정적인 trajectory ID와 split에 독립적인 initial-condition ID
- 원본 case index, 종료시간, 시간 node, 상태 node와 구간 action
- 압축 배열의 상태·action offset
- Solver 성공, A·B단계 반복 수, hard·종단 잔차
- 재적분 결과, 최종·node 오차비, 연료 사용량과 시도 수

실패한 최적화 시도는 학습 shard에 넣지 않습니다. Dataset 수율을 과장하지 않도록 manifest의 명시적 실패 기록에 남깁니다.

## 압축 저장 규약

각 split은 `allow_pickle=False`인 `packed_npz_v1` shard 하나를 사용합니다. 길이가 다른 궤적을 이어 붙이고 별도의 상태·action offset 배열로 복원합니다. 한 궤적의 앞 `T`개 상태는 Behavior Cloning 입력, `T`개 action은 target이며 마지막 상태는 궤적·착륙 검증용으로만 보존합니다.

필수 배열은 `trajectory_ids`, `initial_condition_ids`, `source_case_indices`, `state_offsets`, `action_offsets`, `times_s`, `states`, `actions`, `durations_s`, `solver_success`, `stage_a_iterations`, `stage_b_iterations`, `max_hard_violation`, `max_terminal_violation`, `replay_passed`, `max_final_error_ratio`, `max_node_error_ratio`, `fuel_used_kg`, `attempt_count`입니다.

## Split 설계

각 split은 서로 다른 seed로 독립 생성합니다. Validation은 IID model 선택을 위해 원래 쉬운 범위를 유지합니다. 최종 train은 23일차 easy shard, 더 넓은 challenge 보강과 coverage 표적 보강을 합칩니다. Test는 수평 오차, 고도, 하강속도와 질량 범위가 확장 train과도 겹치지 않고, 수평속도·자세·각속도 범위가 더 넓은 hard OOD 조건입니다.

| Field | Easy train·validation | Challenge train | Hard test |
|---|---:|---:|---:|
| 수평 절대 오차 | 2-10 m | 8-11 m | 12-20 m |
| 고도 | 70-105 m | 90-108 m | 110-140 m |
| 수평속도 | -1.5-1.5 m/s | -3-3 m/s | -4-4 m/s |
| 수직속도 | -22에서 -15 m/s | -23.5에서 -20 m/s | -32에서 -24 m/s |
| 자세 | -3-3 deg | -7-7 deg | -10-10 deg |
| 각속도 | -1.5-1.5 deg/s | -3-3 deg/s | -5-5 deg/s |
| 질량 | 970-1000 kg | 962-980 kg | 900-960 kg |

| Split | 난이도 | 계획 궤적 수 | Seed |
|---|---|---:|---:|
| Train | easy nominal | 800 | 20260922 |
| Train 보강 | 난이도 과표집 challenge | 후보 640개 중 160개 | 20260924 |
| Train coverage | 부족 구간 적응 보강 | 최대 160개 | 20261024 |
| Validation | easy IID | 100 | 20261022 |
| Test | hard OOD | 200 | 20261122 |

생성한 22일차 계획에는 split 내부 initial-condition ID 중복과 split 사이 겹침이 없습니다. 동결된 초기상태 SHA-256은 train `06e0203d6ffb9f26687ce6d734fb9010c969b9cb7d139bbef22a1231311937d1`, validation `ca45656bf8785808716bb522f201bfa72827f45af21d67e98c1485e2652070eb`, test `b244481859f99fd19c3ba44235fbd8f6afb4d31b04e59fa37e9d67c173420725`입니다.

Hard split은 train 분포에서 무작위로 뽑은 꼬리 구간이 아닙니다. Train 범위를 확장한 뒤에도 모든 test case는 모든 train case보다 pad에서 멀고, 높고, 더 빠르게 하강하며, 사용할 수 있는 추진제가 적습니다. 나머지 운동 범위도 더 넓습니다.

## Leakage 방지

Split 단위는 개별 state-action 행이 아니라 완전한 trajectory입니다. 한 trajectory는 물리 shard 하나에만 들어갑니다. Initial-condition ID는 split label을 붙이기 전에 canonical little-endian float64 상태 byte로 계산합니다. 같은 ID가 split 안에서 중복되거나 split 사이에 나타나면 생성을 실패 처리합니다. Normalization 통계와 이후 sampling weight는 train에서만 계산합니다.

Validation과 test 궤적은 normalization, checkpoint 선택 gradient, replay-buffer 보강, DAgger 수집 또는 hyperparameter fitting에 사용할 수 없습니다. Validation은 checkpoint 선택에 사용할 수 있습니다. Test는 학습 절차를 고정한 뒤 한 번 평가합니다.

## Normalization 규약

Normalization은 채택된 train trajectory의 서로 정렬된 비종단 상태·action pair에서 standard score를 계산합니다.

~~~text
normalized = (value - train_mean) / max(train_population_std, 1e-6)
~~~

통계는 float64로 누적하고 저장합니다. 상태와 action의 각 field는 count, mean, population standard deviation, 실제 적용 scale, minimum과 maximum을 기록합니다. 각도는 rad를 유지합니다. 추론할 때 상태를 normalize하고, 예측 action을 denormalize한 다음 물리 구동기 제한을 적용합니다. 종단 상태는 action label이 없으므로 통계에서 제외합니다.

최종 train shard에는 정렬된 비종단 pair 100,700개가 있습니다. `normalization.json`에 저장한 float64 통계가 `planar-bc-v1-final`의 최종 normalization 값입니다.

## 품질 기준

두 최적화 단계가 설정 허용오차를 충족하고 새 simulator 재적분이 최종 상태 일치, 착륙 제약, 고도, 추진제 여유, 기울기, 각속도, throttle과 gimbal 검사를 통과해야 학습 shard에 들어갈 수 있습니다. NaN·무한값, 잘못된 offset, 중복 ID, 감소하는 시간, 상태·action 개수 불일치와 재적분 실패 기록은 거부합니다.

Dataset manifest는 생성, 채택, 재시도, timeout, 실패와 필터 제외 수를 각각 보고합니다. 많은 채택 건수로 최적화 실패를 숨기지 않습니다.

23일차 실행은 쉬운 train 조건 800개를 모두 처리하고 trajectory 797개를 채택했습니다. 4개 case가 재시도됐고 1개는 복구됐으며, 3개는 세밀한 재적분에서 기울기 제한을 넘어 제외됐습니다. Solver를 통과한 결과 중 중복 ID, 비유한 값, 잘못된 구조 또는 다른 제약 문제로 추가 제외된 궤적은 없습니다. Packed shard는 dtype, offset, 유한값, 고유성, 시간축, ID 재구성과 accepted-only 검사를 모두 통과했습니다.

24일차에는 후보 640개 중 난이도 상위 120개를 포함한 challenge 160개를 선택했고 모두 채택했습니다. Coverage는 feature마다 8구간, 구간당 최소 30개를 사용했습니다. 한 번의 표적 보강에서 51개 중 50개가 채택되어 전체 marginal deficit이 51에서 0으로 줄었습니다. Validation은 100개 중 100개, hard OOD test는 200개 중 191개를 채택했습니다. Hard test 실패 9개와 coverage 실패 1개는 manifest에 보존하고 최종 shard에는 채택 궤적만 넣었습니다. Split 사이 initial-condition 중복, timeout과 solver 성공 후 filter 탈락은 없습니다.

## 첫 모델 사용 결과(25일차)

첫 `7-64-64-2` Behavior Cloning 정책은 train 상태-action pair 100,700개와 IID validation 10,000개를 사용했습니다. 작은 부분집합 과적합 검사를 통과했고, 최고 79 epoch checkpoint의 정규화 validation MSE는 `0.030977`, throttle MAE는 `0.032989`, gimbal MAE는 `0.272693도`였습니다. 학습과 checkpoint 선택 중 hard OOD test shard는 읽지 않았습니다. 이 수치는 action 예측 오차이며 착륙 결과는 이후 폐루프 평가에서 측정합니다.

## 알려진 한계

- 모델은 평면 3자유도이며 3D 또는 6자유도 기체가 아닙니다.
- 이후 version에서 별도로 밝히지 않는 한 label은 무풍 명목 조건입니다.
- 제어 변화율 제한은 soft 목적함수 항이며 hard actuator slew 제한이 아닙니다.
- 지면 접촉에 landing leg와 구조 충격 모델이 없습니다.
- Hard test는 초기조건 기준으로 더 어렵지만 sensor noise, engine lag, 난류나 model mismatch는 포함하지 않습니다.
- 구동기 또는 상태 경계에 가까운 Teacher 궤적은 student에게 충분한 강건성 margin을 주지 못할 수 있습니다.

## 재현 방법

~~~bash
python scripts/design_offline_dataset.py
python scripts/generate_offline_dataset.py
python scripts/finalize_offline_dataset.py
~~~

마지막 명령은 23일차 원본을 검증하고 `artifacts/day24-final-dataset/`에 version manifest, normalization 통계, 분포 PNG와 물리 shard 3개를 생성합니다. 생성 폴더는 Git에서 제외되며 명령은 기존 결과를 덮어쓰지 않습니다. 최종 shard SHA-256은 train `4161bc93b6753bdcc84cfb6eec6afe1c45bbf814b34525817a7ec24835691e0d`, validation `e77b561f63f5738ec6e09e24b9dbffe2e4712ff33e7d186d12b4945e2aaa618c`, test `6b67c7866e202b6719f07c733cd2cd82fdd3e2f1638c5bb33ec10ea3f3f5ba17`입니다.
