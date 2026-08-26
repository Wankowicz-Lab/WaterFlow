# Keep the baked virtualenv active while a synced checkout overrides the baked
# source tree for interactive ACTL work.
if [ -d /app/.venv/bin ]; then
  VIRTUAL_ENV=/app/.venv
  export VIRTUAL_ENV
  case ":${PATH}:" in
    *:/app/.venv/bin:*) ;;
    *) PATH="/app/.venv/bin:${PATH}" ;;
  esac
  export PATH
fi

if [ -d /home/dev/workspace ]; then
  case ":${PYTHONPATH:-}:" in
    *:/home/dev/workspace:*) ;;
    *) PYTHONPATH="/home/dev/workspace${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
fi
case ":${PYTHONPATH:-}:" in
  *:/app:*) ;;
  *) PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/app" ;;
esac
export PYTHONPATH

if [ -d /mnt/diffuse-shared ]; then
  if ! mkdir -p /mnt/diffuse-shared/waterflow/{pdb,cache,checkpoints,outputs,logs,splits}; then
    printf 'waterflow: warning: could not initialize shared storage under /mnt/diffuse-shared/waterflow\n' >&2
  fi
fi

case "$-" in
  *i*) [ -d /home/dev/workspace ] && cd /home/dev/workspace || true ;;
esac
