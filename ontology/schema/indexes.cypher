// 지식 추출 계층 조회용 인덱스 (설계 3단계 2-3, M4 PR 3).
// 검수 화면은 실행(run_id)과 상태(status)로 후보를 고른다
CREATE INDEX candidate_run IF NOT EXISTS FOR (n:Candidate) ON (n.run_id);
CREATE INDEX candidate_status IF NOT EXISTS FOR (n:Candidate) ON (n.status);
