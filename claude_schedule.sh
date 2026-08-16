#!/usr/bin/env bash

# ===== EDIT THESE =====
PROJECT_DIR="/afs/cern.ch/work/j/jmgray/private/tmom-recon"
CONTEXT_FILE="/afs/cern.ch/work/j/jmgray/private/tmom-recon/src/tmom_recon/acd/reconstruction.py"
PROMPT_FILE="/afs/cern.ch/work/j/jmgray/private/tmom-recon/claude_prompt.txt"
LOG_FILE="/tmp/claude_2am.log"
# ======================

cat > /tmp/run_claude_2am.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_DIR"

claude \
  --model opus \
  --effort medium \
  --permission-mode auto \
  -p "
Please read the following context file:

$CONTEXT_FILE

Then carry out these instructions:

\$(cat "$PROMPT_FILE")
" \
  > "$LOG_FILE" 2>&1
EOF

chmod +x /tmp/run_claude_2am.sh

echo "/tmp/run_claude_2am.sh" | at 5:00am

echo "Scheduled."
echo
echo "Queue:"
atq
