#!/usr/bin/env bash
# 개발용 Docker와 Neo4j를 띄운다 (CLAUDE.md 10장).
#
# 클라우드 작업 환경은 머신이 다시 시작될 때 dockerd가 함께 내려가고, 자동으로 다시 뜨지 않는다.
# 이때 /var/run/docker.pid가 남아 있으면 새 dockerd가 "아직 실행 중"이라며 시작을 거부한다.
# 재시작 뒤 다른 프로세스가 같은 PID를 받는 경우다 (2026-11-24 확인).
#
# 실행: scripts/dev_up.sh              # neo4j-test(7688)와 neo4j(7687)
#       scripts/dev_up.sh neo4j-test   # 테스트용 DB만
#       scripts/dev_up.sh neo4j-ext    # 구성 C-ext 그래프 DB(7689). C-ext 실행 전에 띄운다
# 이미 떠 있으면 아무것도 바꾸지 않고 상태만 확인한다. 떠 있는 컨테이너는 다시 만들지 않으므로
# compose 설정이나 이미지를 바꾼 뒤에는 `docker compose up -d <서비스>`로 따로 다시 만든다.
set -euo pipefail

cd "$(dirname "$0")/.."
SERVICES=("$@")
[ ${#SERVICES[@]} -eq 0 ] && SERVICES=(neo4j-test neo4j)
PIDFILE=/var/run/docker.pid
LOG=/tmp/dockerd.log

# 확인 명령 한 번에 쓰는 최대 초. dockerd가 연결만 받고 답하지 않아도 대기 제한이 지켜지게 한다
CHECK_TIMEOUT=10

docker_ready() { timeout "$CHECK_TIMEOUT" docker info >/dev/null 2>&1; }
healthy() {  # 서비스 이름
    timeout "$CHECK_TIMEOUT" docker compose ps "$1" 2>/dev/null | grep -q "(healthy)"
}

wait_for() {  # 최대 초, 확인 명령 (명령 한 번은 CHECK_TIMEOUT 안에 끝난다)
    local limit=$1; shift
    local deadline=$((SECONDS + limit))
    while [ "$SECONDS" -lt "$deadline" ]; do
        "$@" && return 0
        sleep 1
    done
    return 1
}

if ! docker_ready; then
    if pgrep -x dockerd >/dev/null; then
        echo "dockerd가 실행 중이다. 준비될 때까지 기다린다"
    else
        # dockerd가 없는데 pid 파일이 남아 있으면 지난 머신의 흔적이다
        if [ -f "$PIDFILE" ]; then
            echo "dockerd 없이 남은 $PIDFILE (PID $(cat "$PIDFILE"))를 지운다"
            rm -f "$PIDFILE"
        fi
        echo "dockerd를 시작한다 (로그: $LOG)"
        nohup dockerd >"$LOG" 2>&1 &
    fi
    if ! wait_for 60 docker_ready; then
        echo "dockerd가 60초 안에 준비되지 않았다. $LOG를 확인한다" >&2
        tail -5 "$LOG" >&2 || true
        exit 1
    fi
fi
echo "docker 준비됨"

# 떠 있는 컨테이너는 그대로 두고, 없으면 만들고, 멈춰 있으면 시작한다
docker compose up -d --no-recreate "${SERVICES[@]}"
for service in "${SERVICES[@]}"; do
    if ! wait_for 180 healthy "$service"; then
        echo "$service가 180초 안에 healthy가 되지 않았다" >&2
        timeout "$CHECK_TIMEOUT" docker compose ps "$service" >&2 || true
        exit 1
    fi
    echo "$service healthy"
done
