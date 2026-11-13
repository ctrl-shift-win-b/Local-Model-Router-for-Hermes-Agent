#!/bin/bash
# Shared start/stop holds for the GPU router and the manual engine scripts.
# Hermes holds only the router. The router starts a GPU for a request and
# stops it after the idle timeout. A manual `start` holds that engine so the
# router will not idle-stop it or unload it for another model. An explicit
# `stop` clears the hold and stops the process. A remote text server is not
# a unit here: the router on that machine starts it.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/hermes-gpus"
HOLD="$STATE/holds"
LOCK="$STATE/lock"
ROUTER_URL="${NINFER_ROUTER_URL:-http://127.0.0.1:8092}"
# Hermes starts the router only. Managed engines stay unloaded until a
# request needs them. Manual start/stop of those engines still uses their names.
ROUTER_UNIT=ninfer-router.service
ACQUIRE_NAMES=(router)

mkdir -p "$HOLD"

hold_dir() {
  printf '%s\n' "$HOLD/$1"
}

holds_left() {
  local dir
  dir="$(hold_dir "$1")"
  [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

add_hold() {
  mkdir -p "$(hold_dir "$1")"
  : > "$(hold_dir "$1")/$2"
}

clear_holds() {
  rm -rf "$(hold_dir "$1")"
}

with_lock() {
  local fd
  exec {fd}>"$LOCK"
  flock "$fd"
  "$@"
  local status=$?
  flock -u "$fd"
  return "$status"
}

unit_for() {
  printf 'ninfer-%s.service\n' "$1"
}

wait_http() {
  local url=$1 timeout=$2 deadline
  deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "timed out waiting for $url" >&2
  return 1
}

start_router() {
  if ! systemctl --user start "$ROUTER_UNIT"; then
    echo "warning: systemctl --user start $ROUTER_UNIT failed" >&2
  fi
}

supervisor_pid_file() {
  printf '%s/supervisors/%s.pid\n' "$STATE" "$1"
}

stop_supervised() {
  local name=$1 file pid i
  file="$(supervisor_pid_file "$name")"
  [[ -f "$file" ]] || return 0
  pid="$(tr -cd '0-9' <"$file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$file"
}

stop_unit_if_idle() {
  local name=$1
  if holds_left "$name"; then
    return 0
  fi
  systemctl --user stop "$(unit_for "$name")" || true
}

acquire_pid() {
  local pid=$1
  with_lock bash -c '
    set -euo pipefail
    hold=$1 pid=$2
    shift 2
    for name in "$@"; do
      mkdir -p "$hold/$name"
      : > "$hold/$name/hermes-$pid"
    done
  ' _ "$HOLD" "$pid" "${ACQUIRE_NAMES[@]}"
  start_router
}

release_pid() {
  local pid=$1 name
  with_lock bash -c '
    set -euo pipefail
    hold=$1 pid=$2
    shift 2
    for name in "$@"; do
      rm -f "$hold/$name/hermes-$pid"
    done
  ' _ "$HOLD" "$pid" "${ACQUIRE_NAMES[@]}"
  for name in "${ACQUIRE_NAMES[@]}"; do
    stop_unit_if_idle "$name"
  done
}

wait_ready() {
  local deadline
  deadline=$((SECONDS + ${NINFER_READY_TIMEOUT:-60}))
  echo "waiting for the gpu router"
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "$ROUTER_URL/health" >/dev/null 2>&1; then
      echo "router is listening; a GPU loads on the first request and unloads after it is idle"
      return 0
    fi
    sleep 1
  done
  echo "router did not listen on $ROUTER_URL within ${NINFER_READY_TIMEOUT:-60}s" >&2
  return 1
}

manual_start() {
  local name=$1 port=$2
  local unit deadline state
  unit="$(unit_for "$name")"
  with_lock add_hold "$name" manual
  if ! systemctl --user start "$unit"; then
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "$name is already serving on 127.0.0.1:${port}; it stays up until ninfer-$name stop"
      return 0
    fi
    with_lock clear_holds "$name"
    echo "systemctl --user start $unit failed" >&2
    return 1
  fi
  echo "loading $name"
  deadline=$((SECONDS + ${NINFER_READY_TIMEOUT:-300}))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "$name is serving on 127.0.0.1:${port}"
      return 0
    fi
    state="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
    if [[ "$state" == "failed" || "$state" == "inactive" ]]; then
      echo "$unit is $state" >&2
      systemctl --user status "$unit" --no-pager -l >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "timed out waiting for $name on 127.0.0.1:${port}" >&2
  return 1
}

manual_stop() {
  local name=$1
  with_lock clear_holds "$name"
  systemctl --user stop "$(unit_for "$name")" || true
  stop_supervised "$name"
}

show_status() {
  local name=$1 port=$2
  local unit active
  unit="$(unit_for "$name")"
  active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
  printf '%s service: %s\n' "$name" "${active:-unknown}"
  if [[ -n "$port" ]]; then
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print("models:", " ".join(str(m.get("id","")) for m in d.get("data",[])))'; then
      :
    else
      echo "models: not responding on 127.0.0.1:${port}"
    fi
  fi
}

engine_exec() {
  local cuda=$1
  shift
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$cuda"
  exec "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  command=${1:-}
  shift || true
  case "$command" in
    acquire) acquire_pid "${1:?pid}" ;;
    release) release_pid "${1:?pid}" ;;
    wait-ready) wait_ready ;;
    *)
      echo "usage: engine-common.sh acquire|release PID | wait-ready" >&2
      exit 2
      ;;
  esac
fi
