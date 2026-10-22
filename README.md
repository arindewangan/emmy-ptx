# emmy-ptx

My entry for the **RiftStack Emmy Hackathon 2026** — a Python compiler that translates
[Emmy](https://github.com/cloudrift-ai/emmy) Kernel IR into PTX assembly for NVIDIA GPUs.

Contract: `python compile.py <kernel>.ir.json > out.ptx`

## What this is

An original, from-scratch lowering pass: recursive descent over Emmy's Kernel IR
statements, emitting PTX text directly. Emmy is used **only** to load the IR
(`Graph.from_dict`) and to read the launch contract from `meta.json`
(entry name, `arg_order`, grid/block, buffer shapes). This entry does **not**
chain `emmy.render_kernelop`, `emmy compile`, or `nvcc` — per the hackathon's
honesty rule, judges read the compiler code.

Key lowering decisions (full detail in `BUILD_NOTES.md`):

- Params from `meta.json` `arg_order`, all `.u64` pointers (`ld.param.u64` + `cvta.to.global.u64`)
- Tile decode: `_gid = ctaid.x*ntid.x + tid.x`, row-major axis decode from extents, with an emitted bounds guard only when threads exceed cells
- Accumulators initialized at the loop that directly contains their `Accum` (add→0.0, maximum→-1e30, multiply→1.0, minimum→1e30); online-softmax `base` operands respected
- Warp-shuffle butterfly + tree-halve reductions lowered generically from the IR's `combine_states` assign lists (`shfl.sync.bfly.b32`, guarded halve + `bar.sync`)
- Elementwise ops, `div.rn`/`sqrt.rn`/`rcp.approx`, `exp` via `ex2.approx` (tolerances are rtol=1e-3)
- `cp.async` lowered with static `.shared` under 48KB; the 73KB m01 kernel uses dynamic `.extern .shared` carved at emmy's own offsets
- `Loop`/`StridedLoop` → real PTX loops, `TernaryExpr` → `selp`

## Verification status (honest)

- ✅ All **15 open problems** compile without crashing
- ✅ Generated PTX **assembles cleanly under `ptxas --gpu-name sm_89`** for all 15
- ✅ Manual semantic review (r01, f02, m01, a01, a02) traced against the reference
  CUDA kernels — all match
- ⏳ **GPU runtime verification pending**: the remote runner's POST endpoints are
  unreachable from my build machine (egress proxy blackholes non-GET requests —
  `GET /health` works, every POST hangs). The compiler will be verified via
  `submit.py --compiler` once it can reach the runner; RiftStack evaluates every
  submission on its own hardware regardless.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install emmy-ml==0.3.11 numpy requests   # Python 3.12+
python compile.py <kernel>.ir.json > out.ptx
```

`compile_req.json` pins the dependency versions used for this build.

## Files

- `compile.py` — the compiler (source of truth; required at repo root by the hackathon contract)
- `BUILD_NOTES.md` — per-problem status, the egress-proxy blocker, lowering decisions, known risks for the hidden problem set
- `compile_req.json` — dependency pins

## License

MIT — see `LICENSE`.
