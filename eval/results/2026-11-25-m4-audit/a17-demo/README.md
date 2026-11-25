# A17 지식 반영 시연 (M4 점검, 2026-11-25 KST)

껍데기 방지 G3(추출 → 검수 → 반영 → 그래프)를 실제 dev 추출 후보로 확인한 기록이다 (설계 4단계 7장).

- 후보: 추출 실행 `2026-11-25-m4-extract-dev-r1`(claude-haiku-5-5, extraction-v1)의 후보 4개
  - CH-05:004 별칭 거래소 → 경매장 (검수 TM03)
  - CH-05:005 별칭 게임 DB → DB (검수 TM03, TM10)
  - CH-01:001 의존 캐시 상점 → 우편함, K에 이미 있음 (검수 TM03, TM06)
  - PM-01:008 C02 → P6 조건 미지급, conflict (검수 TM03) → 반려
- 누른 곳: 운영 화면 코드(apps/review_ui/app.py)를 Streamlit AppTest로 실행해 버튼을 눌렀다. 화면 캡처는 같은 개발용 DB로 띄운
  실제 화면이다 (Playwright)
- 순서: 검수(1-review.json) → 실행 팀 TM05가 반영 요청(1-request-list.png, 2-request.json) → 승인권자 TM03, TM06, TM10
  승인(2-approve-pending.png, 3-approve.json) → TM05가 반영 실행(3-execute.json, 4-history.png)
- 결과: 실행 기록 AR-ca09cbd91a04 완료. 시드 커밋 `1a9af8b`(4-commit.patch)가 systems.yaml에 별칭 2개를 넣고
  provenance.yaml에 출처 3건을 남겼다. 개발용 Neo4j에 다시 적재해 SY14 별칭과 의존 관계의 source_doc_ids를 확인했다
- 커밋은 로컬 브랜치 `kb-demo-m4`에만 했고 푸시하지 않았다. A17로 바뀐 시드는 M8 전까지 main에 넣지 않는다 (M4 PR 5 결정 5).
  시연 뒤 개발용 Neo4j는 main 시드로 다시 적재했다. 후보 판단과 A17 실행 기록은 개발용 DB에 남아 있다
- 시연 중 발견: 문서에 적힌 실행 명령 `streamlit run apps/review_ui/app.py`가 저장소 모듈을 찾지 못했다. 화면이 저장소 루트를
  모듈 경로에 넣게 고쳤다. 그 전 캡처는 PYTHONPATH를 주고 띄웠다
- 별칭 후보의 근거 문장(CH-05)은 그 이름이 나온 문장이라 별칭 관계를 말하지 않는다. 별칭 후보는 코드가 이름 정규화 결과로
  만들기 때문이다 (5단계 5장). 근거 표본 확인에서 따로 본다
