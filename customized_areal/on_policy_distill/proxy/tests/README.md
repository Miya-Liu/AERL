# Proxy History Consistency Tests

These tests verify that the customized `proxy_rollout_server` correctly stores and
exports conversation history with token-level rewards.

## Test Overview

### 1. `test_proxy_history_consistency.py`

Tests history consistency when using an external proxy server.

**Usage:**

Terminal 1 - Start proxy server:

```bash
python -m customized_areal.on_policy_distill.proxy.proxy_rollout_server \
    --host 0.0.0.0 \
    --port 8000 \
    --admin-api-key "test-admin-key"
```

Terminal 2 - Run test:

```bash
python customized_areal/on_policy_distill/proxy/tests/test_proxy_history_consistency.py \
    --proxy-addr http://localhost:8000 \
    --admin-api-key "test-admin-key"
```

### 2. `test_proxy_history_integration.py`

Integrated test that automatically launches the proxy server, runs the test, and
verifies history consistency.

**Usage:**

Run with auto-managed proxy server:

```bash
python customized_areal/on_policy_distill/proxy/tests/test_proxy_history_integration.py
```

Or specify a custom port:

```bash
python customized_areal/on_policy_distill/proxy/tests/test_proxy_history_integration.py --port 9999
```

Or use an existing proxy server:

```bash
# First, start proxy manually
python -m customized_areal.on_policy_distill.proxy.proxy_rollout_server --port 8000

# Then run test against it
python customized_areal/on_policy_distill/proxy/tests/test_proxy_history_integration.py \
    --use-existing \
    --proxy-addr http://localhost:8000
```

## What the Tests Verify

1. **Agent History Saving**: The agent saves its own copy of the conversation history
   during execution
1. **Proxy Storage**: The proxy server stores interactions via the HTTP API
1. **History Export**: After the session ends, the proxy exports all interactions
1. **Consistency Check**: The test verifies:
   - Number of completions match between agent and proxy
   - Completion IDs match exactly
   - Token-level rewards are properly stored
   - Scalar rewards (sum of token rewards) are consistent

## Expected Output (Success)

```
======================================================================
PROXY HISTORY CONSISTENCY TEST
======================================================================

1. Starting workflow with proxy at http://localhost:8000

2. Workflow completed, exported 2 interactions

3. Agent saved history:
   - Session ID: abc123...
   - Conversation turns: 4
   - Completion IDs: ['comp-1', 'comp-2']

4. Proxy exported interactions:
   Interaction 1: comp-1
   - Messages: 2
   - Reward: 1.5
   - Token rewards: [0.1, 0.2, 0.3, 0.4, 0.5]
   Interaction 2: comp-2
   - Messages: 2
   - Reward: 2.7
   - Token rewards: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

5. COMPARING HISTORIES:
----------------------------------------------------------------------
Agent recorded 2 completions
Proxy exported 2 interactions
Agent completion IDs: {'comp-1', 'comp-2'}
Proxy interaction IDs: {'comp-1', 'comp-2'}

6. CHECKING TOKEN REWARDS:
----------------------------------------------------------------------
comp-1:
  Agent rewards: [0.1, 0.2, 0.3, 0.4, 0.5]
  Proxy rewards: [0.1, 0.2, 0.3, 0.4, 0.5]
  ✓ Token rewards match

7. CHECKING REWARD VALUES:
----------------------------------------------------------------------
comp-1:
  Expected (sum of token rewards): 1.5
  Actual (proxy scalar): 1.5
  ✓ Rewards match

======================================================================
✓ TEST PASSED: Agent history and proxy export are consistent!
======================================================================
```

## Troubleshooting

### Connection Refused

If you see connection errors, ensure the proxy server is running:

```bash
curl http://localhost:8000/docs
```

### Import Errors

Make sure you're running from the project root:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
python customized_areal/on_policy_distill/proxy/tests/test_proxy_history_integration.py
```

### Port Already in Use

If port 8000 (or your chosen port) is already in use:

```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or use a different port
python customized_areal/on_policy_distill/proxy/tests/test_proxy_history_integration.py --port 9999
```
