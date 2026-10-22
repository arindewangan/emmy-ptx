# RiftStack Emmy Hackathon 2026 — build notes

Compiler: `emmy-ptx/compile.py` — contract `python compile.py <kernel>.ir.json > out.ptx`.
Own lowering throughout (recursive descent over Emmy Kernel IR statements, PTX text
emitted directly). Emmy is used ONLY to load the IR (`Graph.from_dict`) plus `meta.json`
for the launch contract (entry name, `arg_order`, grid/block, buffer shapes).

## Status per problem (as of Oct 22 ~01:30 IST)

| problem | category | local ptxas (sm_89) | runner verdict |
|---|---|---|---|
| p00_scalar_add | 1 pointwise | ASSEMBLED | BLOCKED — see below |
| p01_vector_add | 1 | ASSEMBLED | BLOCKED |
| p02_scale_shift | 1 | ASSEMBLED | BLOCKED |
| p03_relu_mul | 1 | ASSEMBLED | BLOCKED |
| r01_array_sum | 2 reduction | ASSEMBLED | BLOCKED |
| r02_big_sum | 2 | ASSEMBLED | BLOCKED |
| r03_mean | 2 | ASSEMBLED | BLOCKED |
| r04_row_sum | 2 | ASSEMBLED | BLOCKED |
| m01_matmul_512 | 3 matmul | ASSEMBLED | BLOCKED |
| m02_matmul_rect | 3 | ASSEMBLED | BLOCKED |
| f01_rmsnorm | 4 fused | ASSEMBLED | BLOCKED |
| f02_row_softmax | 4 | ASSEMBLED | BLOCKED |
| f03_l2norm | 4 | ASSEMBLED | BLOCKED |
| a01_attention | 5 attention | ASSEMBLED | BLOCKED |
| a02_attention_128 | 5 | ASSEMBLED | BLOCKED |

All 15 open problems compile without crashing and the PTX assembles cleanly under
`ptxas --gpu-name sm_89` (triton-bundled ptxas found on this VM). Runtime correctness
on the GPU could NOT be verified: see "Blocker".

## Blocker (genuine, verified multiple ways)

The remote runner (`http://runner.riftstack.ai:48701`, i.e. `submit.py --compiler`) is
unreachable from this VM for the verification loop:

- `GET /health` → 200 in ~1s (runner is up: RTX 5090, sm_120, can_compile=true).
- **Every POST hangs**: `/compile` and `/run` (even malformed bodies that should 422
  instantly), and even `POST /health`. Verified with both `requests` and `curl`,
  over ~1h, multiple attempts.
- **It is not the runner**: `POST https://httpbin.org/post` and `PUT https://httpbin.org/put`
  also hang (000), while `GET https://httpbin.org/get` returns 200 in 1.2s.
- Conclusion: this VM's egress proxy (`hatch-egress-proxy:3128`) blackholes all
  requests with a body / non-GET methods. `submit.py` cannot run here. (Final
  confirmation: a 270s-timeout POST to /compile returned http=000, 0 bytes.)

What this means: the compiler is complete and ptxas-clean on all 15, plus manual
semantic review (r01, f02, m01, a01, a02 traced against ref.cu — all match), but **zero
runner PASS verdicts** exist. Do NOT claim any problem passes. The parent agent should
re-run `submit.py --compiler` from an environment where POST works (or when the proxy
is fixed); the command is:

    cd task-repo && python submit.py <problem-id> --compiler $PWD/../emmy-ptx/compile.py

Suggested order: p00, p01, p02, p03, r01, r02, r03, r04, f01, f03, f02, m02, m01, a02, a01.

## Lowering decisions

- **Entry/params**: entry name from `KernelOp.name`; params in `meta.json` `arg_order`,
  all `.u64`, loaded with `ld.param.u64` + `cvta.to.global.u64`.
- **Tile decode**: `_gid = ctaid.x*ntid.x + tid.x`; axes decoded row-major from axis
  `extent`s (`axis_i = (_gid / stride_i) % extent_i`, div skipped for the first axis,
  rem skipped where provable). Bounds guard `if (_gid < prod(extents))` emitted iff
  `grid.x*block.x > prod(extents)` (p00: 64 cells vs 256 threads).
- **Accumulators**: initialized right before the loop that directly contains their
  `Accum` statement (descending into `Cond`, not into nested loops); identities
  add→0.0, maximum→-1e30 (matches emmy), multiply→1.0, minimum→1e30. Explicit `Init`
  statements (m01) take precedence and are never re-initialized. `Accum` with
  `base` set (f02's online-softmax `acc1 = add(sa, sb)`) uses `base` as the LHS
  operand instead of the old accumulator.
- **WarpShuffle/TreeHalve**: fully generic via the IR's `combine_states` Assign
  lists (single-state add, and the two-state online-softmax max+sum combine both
  work without special-casing). Butterfly `shfl.sync.bfly.b32` at 16,8,4,2,1 with
  mask `0x1f, 0xffffffff`; tree does guarded halve + `bar.sync` per step, then every
  thread reloads `buf[0]`.
- **Elementwise ops**: add/sub/mul, `div.rn.f32`, relu→`max.f32(x,0)`, maximum/minimum,
  `sqrt.rn.f32`, `rcp.approx.f32`, exp→`x*log2(e)` + `ex2.approx.f32` (tolerances are
  rtol=1e-3, so approx is fine), copy→mov.
- **Memory**: row-major flattening from `meta.json` arg shapes (smem from `Smem`
  extents); vector ld/st `.v2`/`.v4` for multi-name loads and grouped writes;
  `red.global.add.f32` for atomic writes (m02; runner zero-fills via `zero_outputs`).
- **cp.async** (m01/m02): `cp.async.cg.shared.global [smem32],[gmem64], <4|8|16>`;
  `smem_index + lane_index` (lane_index may be None — m02 folds it into smem_index);
  commit/wait map directly. **m01 needs >48KB smem** (73728B): static `.shared`
  exceeds sm_89's static limit, so the compiler switches to one
  `.extern .shared .b8 __emmy_dyn_smem[]` array carved at aligned offsets
  (`_a_smem`@0, `_b_smem`@24576 — same as emmy's pool); `meta.json`'s `smem_bytes`
  (73728) covers it. All other problems use static `.shared` decls.
- **Loops**: `Loop`/`StridedLoop` → real PTX loops (`setp.ge` + `@p bra`);
  `TernaryExpr` (m01's masked tail) → `selp`.
- **Registers**: fresh virtual regs per value (`%f/%r/%rd/%p`), counts declared up
  front; no liveness analysis (ptxas allocates). Output is ASCII-only.

## Known limitations / risks for the hidden set

- All params assumed `.u64` pointers (true for every open problem; `serial` is empty
  everywhere). A scalar runtime param would need a `.param .u32` + `ld.param.u32` path.
- `exp` uses `ex2.approx` (~1e-6 relative); fine at rtol=1e-3 but would be the first
  thing to upgrade if a hidden problem tightens tolerance.
- `Sync` with `warp=True` emits nothing (`__syncwarp` is a nop); `barrier_count`
  maps to `bar.sync id, count`.
- Only dtypes seen are f32 (plus int indices); the dtype table covers f16/f64/ints
  generically but those paths are untested.
- `swizzle != NONE` on CpAsyncCopy/Write raises loudly (not present in open set).
- KernelOp bodies with more than one Tile, or a Tile not covering the whole body,
  are not specially handled (not present in open set).
- Guard + cross-thread ops (shfl/barrier) in one kernel would be UB — emmy does not
  generate such kernels (guarded kernels here are pure pointwise).

## Files

- `emmy-ptx/compile.py` — the compiler (source of truth; `task-repo/` untouched).
- `emmy-ptx/.venv/` — Python 3.12 venv with emmy-ml==0.3.11, numpy, requests.
- `emmy-ptx/ptx_out/<problem>.ptx` — generated PTX for all 15 (regenerable; not for submission).
- Prep material (untouched): `PREP.md`, `PREP_PT_X.md`, `BLOG_NOTES.md`, `emmy/`, `emmy-upstream/`.
