// 지식 추출 계층 조회용 인덱스 (설계 3단계 2-3, M4 PR 3).
// 검수 화면은 실행(run_id)과 상태(status)로 후보를 고른다
CREATE INDEX candidate_run IF NOT EXISTS FOR (n:Candidate) ON (n.run_id);
CREATE INDEX candidate_status IF NOT EXISTS FOR (n:Candidate) ON (n.status);

// 문서 검색 조각의 벡터 인덱스 (설계 4단계 2-5, 5단계 2장, M5 PR 1).
// Voyage voyage-4의 기본 차원(1024)과 코사인 유사도. 조각 방식(scheme)으로 걸러 찾는다
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding WITH [c.scheme]
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};
