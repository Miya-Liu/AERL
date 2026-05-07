#!/bin/bash
set -e

AREAL_DIR="/dfs/share-groups/letrain/zhoujie/AReaL-main"
BACKEND_DIR="/dfs/share-groups/letrain/zhoujie/le-agent-dev/backend"

echo "Starting jobs in background..."

# 1. Tree search training job
cd "$AREAL_DIR"
nohup bash -c 'uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml 2>&1 | tee training.log' > /dev/null 2>&1 &
TRAIN_PID=$!
echo "tree_search  started (PID: $TRAIN_PID), log: $AREAL_DIR/training.log"

# 2. API server
cd "$BACKEND_DIR"
nohup bash -c 'uv run api.py 2>&1 | tee api.log' > /dev/null 2>&1 &
API_PID=$!
echo "api_backend  started (PID: $API_PID), log: $BACKEND_DIR/api.log"

# 3. Dramatiq workers
cd "$BACKEND_DIR"
nohup bash -c 'uv run -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 8 --threads 8 2>&1 | tee worker.log' > /dev/null 2>&1 &
WORKER_PID=$!
echo "worker_backend started (PID: $WORKER_PID), log: $BACKEND_DIR/worker.log"

echo ""
echo "Monitor live logs with:"
echo "  tail -f $AREAL_DIR/training.log"
echo "  tail -f $BACKEND_DIR/api.log"
echo "  tail -f $BACKEND_DIR/worker.log"
echo ""
echo "Stop jobs with:"
echo "  kill $TRAIN_PID"
echo "  kill $API_PID"
echo "  kill $WORKER_PID"
