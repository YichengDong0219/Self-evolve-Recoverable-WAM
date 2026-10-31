# MR8 branch environment (Claude 2026-11-01): the machine's env.sh, then code_mr8 first on PYTHONPATH so the branch's
# modules are used and code/ (MR7 mainline) stays untouched. Usage: source code_mr8/env_mr8.sh <gpu>
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" "$1"
export PYTHONPATH=$L/code_mr8:$PYTHONPATH
