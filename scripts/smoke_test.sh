#!/usr/bin/env bash
# Smoke test for auth + admin gating against a running local API server.
#
# Assumes the server is already up on $API (see .claude/skills/run-api).
# Registers a fresh throwaway user, promotes it to admin directly in
# SQLite, then checks that admin-gated endpoints return the right status
# code with no token / a non-admin token / the admin token.
#
# Usage: ./scripts/smoke_test.sh
# Safe to re-run — each run registers a new unique admin email.

set -euo pipefail
cd "$(dirname "$0")/.."

API="http://localhost:8000"
DB="data/chatbot.db"
STAMP=$(date +%s)
EMAIL="smoketest-admin-${STAMP}@example.com"
PASSWORD="SmokeTest12345"
REGULAR_EMAIL="smoketest-regular-${STAMP}@example.com"

pass=0
fail=0

check() {
  local desc="$1" expected="$2" actual="$3"
  if [ "$actual" = "$expected" ]; then
    echo "  OK   $desc (HTTP $actual)"
    pass=$((pass + 1))
  else
    echo "  FAIL $desc (expected $expected, got $actual)"
    fail=$((fail + 1))
  fi
}

echo "== Registering throwaway admin + regular users =="
ADMIN_JSON=$(curl -s -X POST "$API/auth/register" -H "Content-Type: application/json" \
  -d "{\"name\":\"Smoke Admin\",\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")
REGULAR_JSON=$(curl -s -X POST "$API/auth/register" -H "Content-Type: application/json" \
  -d "{\"name\":\"Smoke Regular\",\"email\":\"$REGULAR_EMAIL\",\"password\":\"$PASSWORD\"}")

ADMIN_TOKEN=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['access_token'])" "$ADMIN_JSON")
ADMIN_ID=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['user_id'])" "$ADMIN_JSON")
REGULAR_TOKEN=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['access_token'])" "$REGULAR_JSON")

sqlite3 "$DB" "UPDATE users SET is_admin = 1 WHERE email = '$EMAIL';"

echo "== Checking gated endpoints =="

code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/seed-universities")
check "POST /seed-universities, no token -> 401" 401 "$code"

code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/seed-universities" -H "Authorization: Bearer $REGULAR_TOKEN")
check "POST /seed-universities, regular token -> 403" 403 "$code"

code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/seed-universities" -H "Authorization: Bearer $ADMIN_TOKEN")
check "POST /seed-universities, admin token -> 200" 200 "$code"

code=$(curl -s -o /dev/null -w "%{http_code}" "$API/users")
check "GET /users, no token -> 401" 401 "$code"

code=$(curl -s -o /dev/null -w "%{http_code}" "$API/users" -H "Authorization: Bearer $REGULAR_TOKEN")
check "GET /users, regular token -> 403" 403 "$code"

code=$(curl -s -o /dev/null -w "%{http_code}" "$API/users" -H "Authorization: Bearer $ADMIN_TOKEN")
check "GET /users, admin token -> 200" 200 "$code"

code=$(curl -s -o /dev/null -w "%{http_code}" "$API/users/$ADMIN_ID" -H "Authorization: Bearer $REGULAR_TOKEN")
check "GET /users/{other_id}, regular token -> 403" 403 "$code"

echo ""
echo "Passed: $pass  Failed: $fail"
[ "$fail" -eq 0 ]
