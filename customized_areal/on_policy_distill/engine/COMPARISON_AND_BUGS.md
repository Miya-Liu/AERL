# FSDPEngine Comparison Analysis

## Overview

This document compares two FSDP engine implementations:

1. **Original**: `areal/engine/fsdp_engine.py` (1,759 lines)
1. **Customized**: `customized_areal/on_policy_distill/engine/fsdp_engine.py` (333
   lines)

## Key Differences Summary

| Aspect                      | Original FSDPEngine           | MultiCandidateFSDPEngine                                                       |
| --------------------------- | ----------------------------- | ------------------------------------------------------------------------------ |
| **Primary Purpose**         | General-purpose FSDP training | Multi-candidate logprob gathering for on-policy distillation                   |
| **Inheritance**             | `TrainEngine`                 | `FSDPEngine`                                                                   |
| **Lines of Code**           | 1,759                         | 333 (extends parent)                                                           |
| **Key Method Overrides**    | Full implementation           | `_compute_logprobs_entropy`, `_compute_logprobs`, `_compute_logprobs_and_loss` |
| **Multi-Candidate Support** | No                            | Yes (core feature)                                                             |
| **Tree Training**           | Yes                           | Inherits from parent                                                           |

## Detailed Method Comparison

### 1. `_compute_logprobs_entropy`

#### Original (lines 1488-1516)

```python
def _compute_logprobs_entropy(self, logits, inputs, ulysses_pad_size=0):
    labels = inputs.get("rolled_input_ids", torch.roll(inputs["input_ids"], shifts=-1, dims=-1))
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels.squeeze(0)
    logprobs, entropy = gather_logprobs_entropy(
        logits, labels, temperature=self.config.temperature,
        tp_group=self.parallel_helper.tp_group if self.parallel_helper.tp_size > 1 else None
    )
    if self.parallel_helper.sp_size > 1:
        logprobs = self._sp_all_gather(logprobs)
        entropy = self._sp_all_gather(entropy)
        if ulysses_pad_size > 0:
            logprobs = logprobs[:-ulysses_pad_size]
            entropy = entropy[:-ulysses_pad_size]
    return logprobs, entropy
```

**Characteristics:**

- Simple, straightforward single-candidate logprob gathering
- Uses `gather_logprobs_entropy` from `areal.utils.functional`
- Standard Ulysses sequence parallelism handling

#### Customized (lines 42-145)

**Key differences:**

1. **Multi-candidate labels preparation** (lines 161-220)

   - New method `_prepare_multi_candidate_labels` extracts candidate token IDs from
     `position_rewards`
   - Creates 2D labels tensor `[seq_len, max_num_candidates]`
   - Handles variable number of candidates per position

1. **Complex shape handling** (lines 81-96)

   - Handles multiple dimension combinations:
     - `[1, seq_len]` labels with `[seq_len, vocab]` logits → squeeze
     - `[1, seq_len, num_candidates]` labels with `[seq_len, vocab]` logits → squeeze
     - `[seq_len, num_candidates]` labels with `[seq_len, vocab]` logits → add batch dim
     - `[seq_len]` labels with `[seq_len, vocab]` logits → add batch dim

1. **Sequence parallelism with multi-candidate** (lines 117-144)

   - **Potential Bug:** Transpose before all_gather (lines 121-136)
     ```python
     if logprobs.ndim == 2 and logprobs.shape[0] > 1:
         logprobs = logprobs.transpose(0, 1)  # [seq_len, num_candidates] -> [num_candidates, seq_len]
         # ... all_gather ...
         logprobs = logprobs.transpose(0, 1)  # transpose back
     ```
   - **Issue:** The transpose assumes all_gather concatenates along `dim=-1`. For
     sequence parallelism, this should concatenate along the sequence dimension. The
     transpose operation may be incorrect if `logprobs.shape[0]` is the sequence length.

1. **Uses `gather_logprobs_entropy_multi_candidates`** (lines 99-106)

   - Custom function from `..training.logprobs`
   - Handles both single and multi-candidate cases

______________________________________________________________________

### 2. `_compute_logprobs`

#### Original (lines 1518-1544)

```python
def _compute_logprobs(self, logits, inputs, ulysses_pad_size=0):
    labels = inputs.get("rolled_input_ids", torch.roll(...))
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels.squeeze(0)
    logprobs = gather_logprobs(logits, labels, ...)
    if self.parallel_helper.sp_size > 1:
        logprobs = self._sp_all_gather(logprobs)
        if ulysses_pad_size > 0:
            logprobs = logprobs[:-ulysses_pad_size]
    return logprobs
```

#### Customized (lines 147-159)

```python
def _compute_logprobs(self, logits, inputs, ulysses_pad_size=0):
    """Compute logprobs with multi-candidate support (entropy discarded)."""
    logprobs, _ = self._compute_logprobs_entropy(logits, inputs, ulysses_pad_size)
    return logprobs
```

**Analysis:** The customized version simply delegates to `_compute_logprobs_entropy`,
which is correct and reduces code duplication.

______________________________________________________________________

### 3. `_compute_logprobs_and_loss`

#### Original (lines 1557-1622)

- Standard single-candidate path
- Tree training support with `gather_packed_tree_logprobs_entropy`
- Standard vocab stats gathering

#### Customized (lines 222-332)

**Key additions:**

1. **Multi-candidate label preparation** (lines 271-304)

   - Extracts `position_rewards` from `ctx.mb_input`
   - Calls `_prepare_multi_candidate_labels` to create 2D labels
   - Temporarily overrides `rolled_input_ids` with multi-candidate labels
   - Restores original labels after computation

1. **Potential Bug - Missing pad handling for multi-candidate** (lines 310-315)

   ```python
   if ctx.pad_length > 0:
       logprobs = logprobs[: -ctx.pad_length]
       entropy = entropy[: -ctx.pad_length]
       logits = logits[: -ctx.pad_length]
   ```

   - **Issue:** When `logprobs` has shape `[seq_len, num_candidates]`, slicing with
     `[:-pad_length]` correctly removes padding from the sequence dimension. However, if
     `pad_length` is applied to a 2D tensor, the semantics should be verified.

______________________________________________________________________

## Potential Bugs Analysis

### Critical Bugs

#### 1. **Sequence Parallelism Multi-Candidate Transpose Logic** (Line 117-144)

**Location:** `customized_areal/on_policy_distill/engine/fsdp_engine.py:117-144`

**Problem:**

```python
if logprobs.ndim == 2 and logprobs.shape[0] > 1:
    # logprobs: [seq_len, num_candidates] -> transpose -> [num_candidates, seq_len]
    logprobs = logprobs.transpose(0, 1)
    # ... all_gather concatenates along dim=-1 (which is now seq_len)
    logprobs = self._sp_all_gather(logprobs)
    # ... transpose back
    logprobs = logprobs.transpose(0, 1)
```

**Issue Details:**

1. The `_sp_all_gather` method concatenates along `dim=-1` (the last dimension)
1. After transpose from `[seq_len, num_candidates]` to `[num_candidates, seq_len]`,
   `dim=-1` is the sequence dimension
1. This means all_gather concatenates along the sequence dimension, which is correct
1. **However:** The condition `logprobs.shape[0] > 1` is ambiguous. If `shape[0]` is the
   sequence length, this check may not correctly identify multi-candidate cases.

**Recommendation:**

```python
# More explicit check for multi-candidate
if logprobs.ndim == 2 and logprobs.shape[1] > 1:  # Check num_candidates > 1
    # Multi-candidate handling
    logprobs = logprobs.transpose(0, 1)
    # ... all_gather ...
    logprobs = logprobs.transpose(0, 1)
```

______________________________________________________________________

#### 2. **Missing Import for `gather_logprobs_entropy`** (Line 99)

**Location:** `customized_areal/on_policy_distill/engine/fsdp_engine.py:99-106`

**Problem:** The customized engine uses `gather_logprobs_entropy_multi_candidates` but
the parent class's `_compute_logprobs_entropy` uses `gather_logprobs_entropy` from
`areal.utils.functional`. The import for `gather_logprobs_entropy` is missing in the
customized file.

**Current imports:**

```python
from areal.engine.fsdp_engine import FSDPEngine
from ..training.logprobs import gather_logprobs_entropy_multi_candidates
```

**Missing import:**

```python
from areal.utils.functional import gather_logprobs_entropy
```

**Note:** Since the customized engine completely overrides `_compute_logprobs_entropy`
and doesn't call the parent's method, this may not cause runtime errors. However, it's a
code hygiene issue.

______________________________________________________________________

#### 3. **Potential Issue with `position_rewards` Extraction** (Line 271-278)

**Location:** `customized_areal/on_policy_distill/engine/fsdp_engine.py:271-278`

**Problem:**

```python
position_rewards = ctx.mb_input.get("position_rewards")
seq_len = logits.shape[0] if logits.ndim == 2 else logits.shape[1]

if position_rewards:
    # Prepare multi-candidate labels
    multi_candidate_labels = self._prepare_multi_candidate_labels(
        ctx.model_inputs, position_rewards, seq_len
    )
```

**Issues:**

1. **Logits shape assumption:** The code assumes `logits.ndim` is 2 or 3, but doesn't
   handle other cases
1. **Seq_len extraction:** For 2D logits `[seq_len, vocab]`, `logits.shape[0]` is
   correct. For 3D logits `[batch, seq_len, vocab]`, `logits.shape[1]` is correct.
   However, if logits has been squeezed or unsqueezed unexpectedly, this could fail.
1. **position_rewards truthiness:** Using `if position_rewards:` could fail if
   `position_rewards` is an empty list `[]` (which is falsy in Python, which may be the
   intended behavior).

**Recommendation:**

```python
position_rewards = ctx.mb_input.get("position_rewards")

# More robust seq_len extraction
if logits.ndim == 2:
    seq_len = logits.shape[0]
elif logits.ndim == 3:
    seq_len = logits.shape[1]
else:
    raise ValueError(f"Unexpected logits ndim: {logits.ndim}")

# Explicit None check and non-empty check
if position_rewards is not None and len(position_rewards) > 0:
    # ... prepare multi-candidate labels
```

______________________________________________________________________

### Minor Issues

#### 4. **Inconsistent Documentation**

The docstrings in the customized engine sometimes reference parent class behavior but
don't explicitly note the multi-candidate extensions.

**Example:** `_compute_logprobs_and_loss` docstring doesn't mention that it handles
multi-candidate logprobs differently from the parent.

______________________________________________________________________

#### 5. **Missing `__all__` Definition**

The customized engine doesn't define `__all__`, which could lead to unintended exports
if `from module import *` is used.

______________________________________________________________________

## Recommendations

### High Priority (Fix Before Production)

1. **Fix sequence parallelism transpose logic** - The condition `logprobs.shape[0] > 1`
   is ambiguous and could cause incorrect all_gather behavior

1. **Add robust shape validation** - The `logits.ndim` checks in
   `_compute_logprobs_and_loss` should handle edge cases more explicitly

1. **Add unit tests for multi-candidate + sequence parallelism** - The
   transpose/all_gather logic is complex and needs testing

### Medium Priority (Code Quality)

1. Add explicit imports for all used functions (even if inherited)
1. Add `__all__` definition
1. Improve docstrings to explicitly document multi-candidate behavior
1. Add type hints for `position_rewards` parameter

### Low Priority (Future Enhancements)

1. Consider extracting multi-candidate logic into a separate mixin class
1. Add performance benchmarks comparing single vs multi-candidate gathering
1. Add visualization tools for debugging multi-candidate label preparation

______________________________________________________________________

## Test Coverage Recommendations

Based on the bug analysis, the following test cases should be added:

1. **Test Multi-Candidate + Sequence Parallelism:**

   ```python
   def test_multi_candidate_sp_all_gather():
       # Test with logprobs shape [seq_len, num_candidates] where num_candidates > 1
       # Verify all_gather produces correct concatenated result
   ```

1. **Test Position Rewards Edge Cases:**

   ```python
   def test_empty_position_rewards():
       # Test with empty list, None, and single candidate cases
   ```

1. **Test Logits Shape Variations:**

   ```python
   def test_logits_ndim_variations():
       # Test with 2D [seq_len, vocab] and 3D [batch, seq_len, vocab]
   ```

______________________________________________________________________

## Conclusion

The `MultiCandidateFSDPEngine` is a well-structured extension of `FSDPEngine` that adds
support for multi-candidate logprob gathering. However, there are **critical bugs in the
sequence parallelism handling** that could cause incorrect results in distributed
training scenarios.

The most important fix needed is the transpose logic in `_compute_logprobs_entropy` when
handling multi-candidate tensors with sequence parallelism. The current condition
`logprobs.shape[0] > 1` is ambiguous and may not correctly identify multi-candidate
cases.

**Priority Actions:**

1. Fix the sequence parallelism transpose logic
1. Add comprehensive unit tests for multi-candidate + SP scenarios
1. Add shape validation assertions for critical tensor operations
