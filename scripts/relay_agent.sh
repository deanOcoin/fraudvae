#!/bin/bash
# fraudvae relay agent. Runs on the rented box when the laptop that normally drives it over
# ssh is not going to stay awake: the box polls a private git repo for command scripts,
# runs each one once, and pushes the output back. The other side is `fraudvae remote relay`.
#
#   repo layout   cmd/<id>.sh   a command to run (pushed by the laptop side)
#                 out/<id>.log  its output, out/<id>.rc the exit code, out/<id>.done when finished
#                 files/        anything the laptop side wants the box to have (code tarballs)
#
# usage: nohup scripts/relay_agent.sh ~/fraudvae-relay > ~/relay_agent.out 2>&1 &
# Long jobs must detach themselves inside their command (nohup ... &); the agent runs
# commands one at a time and a command blocks the queue until it returns.
REPO_DIR=${1:-$HOME/fraudvae-relay}
POLL=${POLL:-15}
MAX_LOG=${MAX_LOG:-3000000}
cd "$REPO_DIR" || { echo "no repo at $REPO_DIR"; exit 1; }
git config user.email "relay@fraudvae.local" >/dev/null 2>&1
git config user.name "fraudvae relay agent" >/dev/null 2>&1
git config pull.rebase true >/dev/null 2>&1
mkdir -p cmd out files
echo "$(date -Is) agent up in $REPO_DIR, pid $$" >> agent.log
echo $$ > agent.pid
while true; do
  git pull -q 2>>agent.err || { git rebase --abort >/dev/null 2>&1; git reset -q --hard origin/main 2>>agent.err; }
  for f in $(ls cmd/*.sh 2>/dev/null | sort); do
    n=$(basename "$f" .sh)
    [ -f "out/$n.done" ] && continue
    echo "$(date -Is) run $n" >> agent.log
    bash "$f" 2>&1 | head -c "$MAX_LOG" > "out/$n.log"
    echo "${PIPESTATUS[0]}" > "out/$n.rc"
    date -Is > "out/$n.done"
    git add out >/dev/null 2>&1
    git commit -qm "$n" >/dev/null 2>&1 || true
    for try in 1 2 3; do
      git push -q 2>>agent.err && break
      git pull -q 2>>agent.err || true
      sleep 3
    done
  done
  sleep "$POLL"
done
