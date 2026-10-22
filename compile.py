#!/usr/bin/env python3
"""Kernel IR -> PTX compiler (RiftStack Emmy Hackathon 2026).

Usage: python compile.py path/to/<kernel>.ir.json > out.ptx

Own lowering: a recursive-descent walk over Emmy's Kernel IR statements that
emits PTX text directly. Emmy is used ONLY to load the IR (Graph.from_dict)
plus meta.json (in the IR's folder) for the launch contract (arg_order, grid,
block, buffer shapes). No rendering through emmy's CUDA backend, no nvcc.

Pipeline stages:
  1. load      - Graph.from_dict, find the KernelOp, read meta.json
  2. prologue  - .version/.target/.address_size, .shared decls, entry
                 signature in meta arg_order, param loads, thread ids,
                 lane/warp, bounds guard derived from the Tile extents
  3. walk      - recursive descent over the body statement tree; env maps
                 IR value names -> (virtual register, ptx type)
  4. emit      - one handler per statement kind

Registers are fresh virtual registers (%f*, %r*, %rd*, %p*); counts are
declared up front and ptxas does the real allocation.
"""

import json
import struct
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# dtype helpers
# --------------------------------------------------------------------------

# emmy dtype name -> (ptx scalar type, nbytes)
DTYPES = {
    "f32": ("f32", 4), "float": ("f32", 4),
    "f64": ("f64", 8), "double": ("f64", 8),
    "f16": ("f16", 2), "bf16": ("bf16", 2),
    "i8": ("s8", 1), "u8": ("u8", 1),
    "i16": ("s16", 2), "u16": ("u16", 2),
    "i32": ("s32", 4), "u32": ("u32", 4),
    "i64": ("s64", 8), "u64": ("u64", 8),
    "int": ("s32", 4),
    "bool": ("pred", 1), "pred": ("pred", 1),
}

# ptx scalar type -> register class prefix
REGCLASS = {
    "f32": "f", "f64": "d", "f16": "h", "bf16": "h",
    "s8": "r", "u8": "r", "s16": "r", "u16": "r",
    "s32": "r", "u32": "r", "b32": "r",
    "s64": "rd", "u64": "rd", "b64": "rd",
    "pred": "p",
}


def ptx_dtype(dt):
    """Emmy DataType (or dtype name str) -> (ptx type, nbytes)."""
    name = dt.name if hasattr(dt, "name") else str(dt)
    if name not in DTYPES:
        raise ValueError("unsupported dtype %r" % (name,))
    return DTYPES[name]


def f32hex(x):
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


def f64hex(x):
    return "0d%016X" % struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def const_mov(ptype, reg, value):
    if ptype == "f32":
        return "mov.f32 %s, %s;" % (reg, f32hex(value))
    if ptype == "f64":
        return "mov.f64 %s, %s;" % (reg, f64hex(value))
    if ptype in ("f16", "bf16"):
        h = struct.unpack("<H", struct.pack("<e", float(value)))[0]
        return "mov.b16 %s, 0h%04X;" % (reg, h)
    if ptype in ("s32", "u32", "b32"):
        return "mov.b32 %s, %d;" % (reg, int(value))
    if ptype in ("s64", "u64", "b64"):
        return "mov.b64 %s, %d;" % (reg, int(value))
    raise ValueError("const_mov: bad ptype %r" % ptype)


# accumulator identities: elementwise op -> (ptx type, identity value)
IDENTITY = {
    "add": ("f32", 0.0),
    "multiply": ("f32", 1.0),
    "maximum": ("f32", -1e30),   # matches emmy's own codegen
    "minimum": ("f32", 1e30),
}


# --------------------------------------------------------------------------
# emitter
# --------------------------------------------------------------------------

class Emitter:
    def __init__(self):
        self.lines = []          # lines inside the .entry body
        self.counts = {"f": 0, "d": 0, "h": 0, "r": 0, "rd": 0, "p": 0}
        self.labels = 0
        self.env = {}            # IR name -> (reg, ptx type)
        self.smem = set()        # smem buffer names
        self.smem_shapes = {}    # name -> tuple of extents
        self.smem_nbytes = {}    # name -> element nbytes
        self.smem_decls = []     # file-scope .shared lines
        self.smem_base = {}      # name -> u32 base-address reg (cached)
        self.smem_ptype = {}     # name -> ptx scalar type
        self.smem_info = []      # (name, size_bytes, align) in decl order
        self.smem_offsets = {}   # name -> byte offset (dynamic mode)
        self.smem_dynamic = False
        self.buf_shapes = {}     # global buffer name -> shape tuple
        self.buf_nbytes = {}     # global buffer name -> element nbytes
        self.buf_dtypes = {}     # global buffer name -> emmy dtype name
        self.inited = set()      # accumulator names with explicit Init

    # 48KB: max *statically* allocated smem on sm_89 (target of our PTX).
    STATIC_SMEM_LIMIT = 49152

    # -- registers / labels -------------------------------------------------
    def newreg(self, ptype):
        cls = REGCLASS[ptype]
        i = self.counts[cls]
        self.counts[cls] += 1
        return "%%%s%d" % (cls, i)

    def newpred(self):
        return self.newreg("pred")

    def newlabel(self, stem):
        self.labels += 1
        return "L_%s_%d" % (stem, self.labels)

    def emit(self, line):
        self.lines.append("        " + line)

    def label(self, name):
        self.lines.append(name + ":")

    # -- expressions --------------------------------------------------------
    def to_s32(self, reg, ptype):
        if ptype in ("s32", "u32", "b32"):
            return reg
        if ptype in ("s64", "u64", "b64"):
            d = self.newreg("s32")
            self.emit("cvt.s32.s64 %s, %s;" % (d, reg))
            return d
        raise ValueError("to_s32: bad ptype %r" % ptype)

    def compile_expr(self, e):
        """Compile an index/scalar expression -> (reg, ptx type)."""
        tn = type(e).__name__
        if tn == "Var":
            if e.name not in self.env:
                raise ValueError("unbound Var %r" % (e.name,))
            return self.env[e.name]
        if tn == "Literal":
            ptype, _ = ptx_dtype(e.dtype)
            r = self.newreg(ptype)
            self.emit(const_mov(ptype, r, e.value))
            return r, ptype
        if tn == "BinaryExpr":
            return self.compile_binary(e)
        if tn == "TernaryExpr":
            return self.compile_ternary(e)
        raise ValueError("unsupported expression %s" % (tn,))

    def compile_binary(self, e):
        op = e.op
        if op in ("==", "!=", "<", "<=", ">", ">="):
            # comparisons are predicates; only valid via compile_pred
            raise ValueError("comparison BinaryExpr used as value")
        lr, lt = self.compile_expr(e.left)
        rr, rt = self.compile_expr(e.right)
        lr = self.to_s32(lr, lt)
        rr = self.to_s32(rr, rt)
        d = self.newreg("s32")
        ins = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32",
               "/": "div.s32", "%": "rem.s32"}[op]
        self.emit("%s %s, %s, %s;" % (ins, d, lr, rr))
        return d, "s32"

    def compile_pred(self, e):
        """Compile a predicate expression -> pred register."""
        tn = type(e).__name__
        if tn == "Var":
            r, t = self.env[e.name]
            if t != "pred":
                raise ValueError("Var %r used as predicate is %s" % (e.name, t))
            return r
        if tn == "BinaryExpr":
            op = e.op
            lr, lt = self.compile_expr(e.left)
            rr, rt = self.compile_expr(e.right)
            # float compare if either side is float
            if lt in ("f32", "f64") or rt in ("f64", "f32"):
                ftype = "f64" if "f64" in (lt, rt) else "f32"
                lr = self.to_f(lr, lt, ftype)
                rr = self.to_f(rr, rt, ftype)
                ins = {"==": "equ", "!=": "neu", "<": "ltu",
                       "<=": "leu", ">": "gtu", ">=": "geu"}[op]
                p = self.newpred()
                self.emit("setp.%s.%s %s, %s, %s;" % (ins, ftype, p, lr, rr))
                return p
            lr = self.to_s32(lr, lt)
            rr = self.to_s32(rr, rt)
            ins = {"==": "eq", "!=": "ne", "<": "lt",
                   "<=": "le", ">": "gt", ">=": "ge"}[op]
            p = self.newpred()
            self.emit("setp.%s.s32 %s, %s, %s;" % (ins, p, lr, rr))
            return p
        raise ValueError("unsupported predicate %s" % (tn,))

    def to_f(self, reg, ptype, ftype):
        if ptype == ftype:
            return reg
        d = self.newreg(ftype)
        self.emit("cvt.%s.s32 %s, %s;" % (ftype, d, reg))
        return d

    def compile_ternary(self, e):
        p = self.compile_pred(e.cond)
        tr, tt = self.compile_expr(e.if_true)
        fr, ft = self.compile_expr(e.if_false)
        if tt != ft:
            raise ValueError("ternary branch type mismatch %s vs %s" % (tt, ft))
        d = self.newreg(tt)
        cls = REGCLASS[tt]
        # selp works on b16/b32/b64
        bits = {"f": "b32", "d": "b64", "h": "b16",
                "r": "b32", "rd": "b64"}[cls]
        tb = self.newreg("b32") if bits == "b32" else self.newreg("b64")
        fb = self.newreg("b32") if bits == "b32" else self.newreg("b64")
        self.emit("mov.%s %s, %s;" % (bits, tb, tr))
        self.emit("mov.%s %s, %s;" % (bits, fb, fr))
        db = self.newreg("b32") if bits == "b32" else self.newreg("b64")
        self.emit("selp.%s %s, %s, %s, %s;" % (bits, db, tb, fb, p))
        self.emit("mov.b32 %s, %s;" % (d, db) if bits == "b32"
                  else "mov.b64 %s, %s;" % (d, db))
        return d, tt

    # -- addresses ----------------------------------------------------------
    def flat_offset(self, idx_regs, shape):
        """Row-major flatten of per-dim s32 regs -> (s32 reg) element offset."""
        acc = None
        for k, r in enumerate(idx_regs):
            r = self.to_s32(r, "s32")
            if acc is None:
                acc = r
            else:
                t = self.newreg("s32")
                dim = self.newreg("s32")
                self.emit("mov.s32 %s, %d;" % (dim, shape[k]))
                self.emit("mad.lo.s32 %s, %s, %s, %s;" % (t, acc, dim, r))
                acc = t
        if acc is None:  # scalar buffer
            acc = self.newreg("s32")
            self.emit("mov.s32 %s, 0;" % acc)
        return acc

    def index_regs(self, index):
        out = []
        for e in index:
            r, t = self.compile_expr(e)
            out.append(self.to_s32(r, t))
        return out

    def global_addr(self, base_rd, elem_off_r32, nbytes):
        nb = self.newreg("s32")
        self.emit("mov.s32 %s, %d;" % (nb, nbytes))
        wide = self.newreg("b64")
        self.emit("mul.wide.s32 %s, %s, %s;" % (wide, elem_off_r32, nb))
        a = self.newreg("b64")
        self.emit("add.s64 %s, %s, %s;" % (a, base_rd, wide))
        return a

    def smem_base_reg(self, name):
        if name not in self.smem_base:
            if self.smem_dynamic:
                r = self.newreg("b32")
                self.emit("mov.u32 %s, __emmy_dyn_smem;" % r)
                off = self.smem_offsets[name]
                if off:
                    o = self.newreg("b32")
                    self.emit("mov.u32 %s, %d;" % (o, off))
                    r2 = self.newreg("b32")
                    self.emit("add.s32 %s, %s, %s;" % (r2, r, o))
                    r = r2
                self.smem_base[name] = r
            else:
                r = self.newreg("b32")
                self.emit("mov.u32 %s, %s;" % (r, name))
                self.smem_base[name] = r
        return self.smem_base[name]

    def smem_addr(self, name, elem_off_r32, nbytes):
        nb = self.newreg("s32")
        self.emit("mov.s32 %s, %d;" % (nb, nbytes))
        byte = self.newreg("s32")
        self.emit("mul.lo.s32 %s, %s, %s;" % (byte, elem_off_r32, nb))
        a = self.newreg("b32")
        self.emit("add.s32 %s, %s, %s;" % (a, self.smem_base_reg(name), byte))
        return a

    # -- elementwise ops ----------------------------------------------------
    def emit_elementwise(self, ptype, op, args):
        """args: list of (reg, ptype). Returns (reg, ptype) of result."""
        d = self.newreg(ptype)
        a = [r for r, _ in args]
        if op == "add":
            self.emit("add.%s %s, %s, %s;" % (ptype, d, a[0], a[1]))
        elif op == "subtract":
            self.emit("sub.%s %s, %s, %s;" % (ptype, d, a[0], a[1]))
        elif op == "multiply":
            self.emit("mul.%s %s, %s, %s;" % (ptype, d, a[0], a[1]))
        elif op == "divide":
            if ptype == "f32":
                self.emit("div.rn.f32 %s, %s, %s;" % (d, a[0], a[1]))
            elif ptype == "f64":
                self.emit("div.rn.f64 %s, %s, %s;" % (d, a[0], a[1]))
            else:
                self.emit("div.%s %s, %s, %s;" % (ptype, d, a[0], a[1]))
        elif op == "relu":
            z = self.newreg(ptype)
            self.emit(const_mov(ptype, z, 0.0))
            self.emit("max.%s %s, %s, %s;" % (ptype, d, a[0], z))
        elif op == "maximum":
            self.emit("max.%s %s, %s, %s;" % (ptype, d, a[0], a[1]))
        elif op == "minimum":
            self.emit("min.%s %s, %s, %s;" % (ptype, d, a[0], a[1]))
        elif op == "sqrt":
            if ptype == "f32":
                self.emit("sqrt.rn.f32 %s, %s;" % (d, a[0]))
            elif ptype == "f64":
                self.emit("sqrt.rn.f64 %s, %s;" % (d, a[0]))
            else:
                raise ValueError("sqrt on " + ptype)
        elif op == "reciprocal":
            if ptype == "f32":
                self.emit("rcp.approx.f32 %s, %s;" % (d, a[0]))
            elif ptype == "f64":
                self.emit("rcp.approx.f64 %s, %s;" % (d, a[0]))
            else:
                raise ValueError("reciprocal on " + ptype)
        elif op == "exp":
            # e^x = 2^(x * log2(e))
            if ptype == "f32":
                t = self.newreg("f32")
                self.emit("mul.f32 %s, %s, 0f3FC62C6D;" % (t, a[0]))
                self.emit("ex2.approx.f32 %s, %s;" % (d, t))
            elif ptype == "f64":
                t = self.newreg("f64")
                self.emit("mul.f64 %s, %s, 0d3FF71547652B82FE;" % (t, a[0]))
                self.emit("ex2.approx.f64 %s, %s;" % (d, t))
            else:
                raise ValueError("exp on " + ptype)
        elif op == "copy":
            self.emit("mov.%s %s, %s;" % (ptype, d, a[0]))
        else:
            raise ValueError("unsupported elementwise op %r" % (op,))
        return d, ptype

    # -- statements ---------------------------------------------------------
    def emit_stmt(self, s):
        tn = type(s).__name__
        fn = getattr(self, "stmt_" + tn, None)
        if fn is None:
            raise ValueError("unsupported statement %s" % (tn,))
        fn(s)

    def stmt_Tile(self, tile):
        axes = list(tile.axes)
        extents = []
        for ax in axes:
            v = ax.extent.expr.value if hasattr(ax.extent, "expr") else ax.extent.hint
            extents.append(int(v))
        total = 1
        for e in extents:
            total *= e
        gid, _ = self.env["_gid"]
        # decode row-major from _gid
        for i, ax in enumerate(axes):
            e = extents[i]
            stride = 1
            for f in extents[i + 1:]:
                stride *= f
            if e == 1:
                z = self.newreg("s32")
                self.emit("mov.s32 %s, 0;" % z)
                self.env[ax.name] = (z, "s32")
            elif i == 0 and stride == 1:
                self.env[ax.name] = (gid, "s32")
            elif i == 0:
                d = self.newreg("s32")
                st = self.newreg("s32")
                self.emit("mov.s32 %s, %d;" % (st, stride))
                self.emit("div.s32 %s, %s, %s;" % (d, gid, st))
                self.env[ax.name] = (d, "s32")
            elif stride == 1:
                d = self.newreg("s32")
                st = self.newreg("s32")
                self.emit("mov.s32 %s, %d;" % (st, e))
                self.emit("rem.s32 %s, %s, %s;" % (d, gid, st))
                self.env[ax.name] = (d, "s32")
            else:
                t = self.newreg("s32")
                st = self.newreg("s32")
                sd = self.newreg("s32")
                self.emit("mov.s32 %s, %d;" % (st, stride))
                self.emit("mov.s32 %s, %d;" % (sd, e))
                self.emit("div.s32 %s, %s, %s;" % (t, gid, st))
                d = self.newreg("s32")
                self.emit("rem.s32 %s, %s, %s;" % (d, t, sd))
                self.env[ax.name] = (d, "s32")
        # bounds guard when the launch carries more threads than cells
        threads = self.grid[0] * self.block[0]
        if threads > total:
            p = self.newpred()
            lbody = self.newlabel("guard")
            lend = self.newlabel("guard")
            tot = self.newreg("s32")
            self.emit("mov.s32 %s, %d;" % (tot, total))
            self.emit("setp.lt.s32 %s, %s, %s;" % (p, gid, tot))
            self.emit("@%s bra %s;" % (p, lbody))
            self.emit("bra %s;" % lend)
            self.label(lbody)
            for s in tile.body:
                self.emit_stmt(s)
            self.label(lend)
        else:
            for s in tile.body:
                self.emit_stmt(s)

    def stmt_Init(self, s):
        ptype, nbytes = ptx_dtype(s.dtype)
        d = self.newreg(ptype)
        self.emit(const_mov(ptype, d, s.identity))
        self.env[s.name] = (d, ptype)
        self.inited.add(s.name)

    def stmt_Let(self, s):
        v = s.value
        if type(v).__name__ != "Literal":
            raise ValueError("Let with non-literal value %s" % (type(v).__name__,))
        ptype, _ = ptx_dtype(s.dtype)
        d = self.newreg(ptype)
        self.emit(const_mov(ptype, d, v.value))
        self.env[s.name] = (d, ptype)

    def resolve_dtype(self, dt, bufname):
        """dtype may be None (smem loads); fall back to the buffer's dtype."""
        if dt is not None:
            return ptx_dtype(dt)
        if bufname in self.smem:
            ptype = self.smem_ptype[bufname]
            return ptype, DTYPES[ptype][1]
        # global buffer: from meta args
        for aname in self.arg_order:
            if aname == bufname:
                return ptx_dtype(self.buf_dtypes[aname])
        raise ValueError("cannot resolve dtype for buffer %r" % (bufname,))

    def stmt_Load(self, s):
        ptype, nbytes = self.resolve_dtype(s.dtype, s.input)
        names = list(s.names)
        regs = [self.newreg(ptype) for _ in names]
        idx = self.index_regs(s.index)
        if s.input in self.smem:
            shape = self.smem_shapes[s.input]
            addr = self.smem_addr(s.input, self.flat_offset(idx, shape), nbytes)
            space = "shared"
        else:
            shape = self.buf_shapes[s.input]
            base, _ = self.env["buf:" + s.input]
            addr = self.global_addr(base, self.flat_offset(idx, shape), nbytes)
            space = "global"
        n = len(names)
        if n == 1:
            self.emit("ld.%s.%s %s, [%s];" % (space, ptype, regs[0], addr))
        elif n == 2:
            self.emit("ld.%s.v2.%s {%s, %s}, [%s];"
                      % (space, ptype, regs[0], regs[1], addr))
        elif n == 4:
            self.emit("ld.%s.v4.%s {%s, %s, %s, %s}, [%s];"
                      % (space, ptype, regs[0], regs[1], regs[2], regs[3], addr))
        elif n == 3:
            if len(idx) != 1:
                raise ValueError("3-value load with multi-dim index")
            self.emit("ld.%s.v2.%s {%s, %s}, [%s];"
                      % (space, ptype, regs[0], regs[1], addr))
            # third element: scalar load at +1 element
            one = self.newreg("s32")
            self.emit("mov.s32 %s, 1;" % one)
            off2 = self.newreg("s32")
            self.emit("add.s32 %s, %s, %s;" % (off2, idx[0], one))
            if space == "global":
                b, _ = self.env["buf:" + s.input]
                a2 = self.global_addr(b, off2, nbytes)
            else:
                a2 = self.smem_addr(s.input, off2, nbytes)
            self.emit("ld.%s.%s %s, [%s];" % (space, ptype, regs[2], a2))
        else:
            raise ValueError("Load with %d names" % n)
        for nm, r in zip(names, regs):
            self.env[nm] = (r, ptype)

    def stmt_Assign(self, s):
        op = s.op.name if s.op is not None and hasattr(s.op, "name") else str(s.op)
        ptype, _ = ptx_dtype(s.dtype)
        args = [self.env[n] for n in s.args]
        d, t = self.emit_elementwise(ptype, op, args)
        self.env[s.name] = (d, t)

    def stmt_Accum(self, s):
        op = s.op.name if hasattr(s.op, "name") else str(s.op)
        base = s.base if s.base is not None else s.name
        breg, btype = self.env[base]
        vreg, vtype = self.env[s.value]
        d, t = self.emit_elementwise(btype, op, [(breg, btype), (vreg, vtype)])
        self.env[s.name] = (d, t)

    def accums_in(self, body):
        """Accum statements directly in body (descend into Cond, not loops)."""
        out = []
        for s in body:
            tn = type(s).__name__
            if tn == "Accum":
                out.append(s)
            elif tn == "Cond":
                out += self.accums_in(s.body)
                out += self.accums_in(s.else_body)
        return out

    def init_accums(self, body):
        for a in self.accums_in(body):
            # Explicit Init statements own their accumulator; every other
            # accumulator that a loop folds is (re-)initialized here, right
            # before the loop that directly contains its Accum statement.
            if a.name in self.inited:
                continue
            op = a.op.name if hasattr(a.op, "name") else str(a.op)
            ptype, ident = IDENTITY[op]
            d = self.newreg(ptype)
            self.emit(const_mov(ptype, d, ident))
            self.env[a.name] = (d, ptype)

    def stmt_Loop(self, s):
        axis = s.axis.name
        extent = int(s.axis.extent.expr.value)
        self.init_accums(s.body)
        i = self.newreg("s32")
        self.emit("mov.s32 %s, 0;" % i)
        self.env[axis] = (i, "s32")
        ltop = self.newlabel("loop")
        lend = self.newlabel("loop")
        lim = self.newreg("s32")
        self.emit("mov.s32 %s, %d;" % (lim, extent))
        p = self.newpred()
        self.label(ltop)
        self.emit("setp.ge.s32 %s, %s, %s;" % (p, i, lim))
        self.emit("@%s bra %s;" % (p, lend))
        for b in s.body:
            self.emit_stmt(b)
        one = self.newreg("s32")
        self.emit("mov.s32 %s, 1;" % one)
        self.emit("add.s32 %s, %s, %s;" % (i, i, one))
        self.emit("bra %s;" % ltop)
        self.label(lend)

    def stmt_StridedLoop(self, s):
        axis = s.axis.name
        extent = int(s.axis.extent.expr.value)
        start_r, start_t = self.compile_expr(s.start)
        step_r, step_t = self.compile_expr(s.step)
        start_r = self.to_s32(start_r, start_t)
        step_r = self.to_s32(step_r, step_t)
        self.init_accums(s.body)
        i = self.newreg("s32")
        self.emit("mov.s32 %s, %s;" % (i, start_r))
        self.env[axis] = (i, "s32")
        ltop = self.newlabel("loop")
        lend = self.newlabel("loop")
        lim = self.newreg("s32")
        self.emit("mov.s32 %s, %d;" % (lim, extent))
        p = self.newpred()
        self.label(ltop)
        self.emit("setp.ge.s32 %s, %s, %s;" % (p, i, lim))
        self.emit("@%s bra %s;" % (p, lend))
        for b in s.body:
            self.emit_stmt(b)
        self.emit("add.s32 %s, %s, %s;" % (i, i, step_r))
        self.emit("bra %s;" % ltop)
        self.label(lend)

    def stmt_Cond(self, s):
        p = self.compile_pred(s.cond)
        lthen = self.newlabel("cond")
        lend = self.newlabel("cond")
        self.emit("@%s bra %s;" % (p, lthen))
        for b in s.else_body:
            self.emit_stmt(b)
        self.emit("bra %s;" % lend)
        self.label(lthen)
        for b in s.body:
            self.emit_stmt(b)
        self.label(lend)

    def stmt_Smem(self, s):
        pass  # declared in the pre-pass

    def stmt_Sync(self, s):
        if s.warp:
            return  # __syncwarp() needs no instruction
        if s.count is not None:
            self.emit("bar.sync %d, %d;" % (s.barrier_id, s.count))
        else:
            self.emit("bar.sync %d;" % (s.barrier_id,))

    def stmt_WarpShuffle(self, ws):
        states = list(ws.state)
        sbs = list(ws.state_b)
        ptype, _ = ptx_dtype(ws.dtype)
        if ptype not in ("f32",):
            raise ValueError("WarpShuffle on " + ptype)
        off = ws.length // 2
        while off >= 1:
            for s, sb in zip(states, sbs):
                sreg, st = self.env[s]
                rb = self.newreg("b32")
                self.emit("mov.b32 %s, %s;" % (rb, sreg))
                ro = self.newreg("b32")
                self.emit("shfl.sync.bfly.b32 %s, %s, %d, 0x1f, 0xffffffff;"
                          % (ro, rb, off))
                fo = self.newreg("f32")
                self.emit("mov.b32 %s, %s;" % (fo, ro))
                self.env[sb] = (fo, "f32")
            for c in ws.combine_states:
                self.combine_assign(c, ptype)
            off //= 2

    def stmt_TreeHalve(self, th):
        states = list(th.state)
        bufs = list(th.bufs)
        ptype, nbytes = ptx_dtype(th.dtype)
        tidreg, _ = self.env[th.tid_var]
        s = th.length // 2
        while s >= 1:
            p = self.newpred()
            lbody = self.newlabel("tree")
            lafter = self.newlabel("tree")
            lim = self.newreg("s32")
            self.emit("mov.s32 %s, %d;" % (lim, s))
            self.emit("setp.lt.s32 %s, %s, %s;" % (p, tidreg, lim))
            self.emit("@%s bra %s;" % (p, lbody))
            self.emit("bra %s;" % lafter)
            self.label(lbody)
            sreg_now = {}
            for i, sname in enumerate(states):
                _, st = self.env[sname]
                addr = self.smem_addr(bufs[i], tidreg, nbytes)
                d = self.newreg(st)
                self.emit("ld.shared.%s %s, [%s];" % (st, d, addr))
                self.env[sname] = (d, st)
                sreg_now[sname] = (d, st)
                idx2 = self.newreg("s32")
                off = self.newreg("s32")
                self.emit("mov.s32 %s, %d;" % (off, s))
                self.emit("add.s32 %s, %s, %s;" % (idx2, tidreg, off))
                addr2 = self.smem_addr(bufs[i], idx2, nbytes)
                d2 = self.newreg(st)
                self.emit("ld.shared.%s %s, [%s];" % (st, d2, addr2))
                self.env[th.state_b[i]] = (d2, st)
            for c in th.combine_states:
                self.combine_assign(c, ptype)
            for i, sname in enumerate(states):
                sreg, st = self.env[sname]
                addr = self.smem_addr(bufs[i], tidreg, nbytes)
                self.emit("st.shared.%s [%s], %s;" % (st, addr, sreg))
            self.label(lafter)
            if th.barrier_count is not None:
                self.emit("bar.sync %d, %d;" % (th.barrier_id, th.barrier_count))
            else:
                self.emit("bar.sync %d;" % (th.barrier_id,))
            s //= 2
        # broadcast bufs[i][0] to every thread
        z = self.newreg("s32")
        self.emit("mov.s32 %s, 0;" % z)
        for i, sname in enumerate(states):
            _, st = self.env[sname]
            addr = self.smem_addr(bufs[i], z, nbytes)
            d = self.newreg(st)
            self.emit("ld.shared.%s %s, [%s];" % (st, d, addr))
            self.env[sname] = (d, st)

    def combine_assign(self, c, ptype):
        op = c.op.name if c.op is not None and hasattr(c.op, "name") else str(c.op)
        args = [self.env[n] for n in c.args]
        d, t = self.emit_elementwise(ptype, op, args)
        self.env[c.name] = (d, t)

    def stmt_CpAsyncCopy(self, c):
        if c.swizzle not in ("NONE", None):
            raise ValueError("CpAsyncCopy swizzle %r" % (c.swizzle,))
        if c.nbytes not in (4, 8, 16):
            raise ValueError("CpAsyncCopy nbytes %r" % (c.nbytes,))
        # smem coords = smem_index (+ lane_index, when present), flattened row-major
        coords = []
        lane_idx = c.lane_index if c.lane_index is not None else (None,) * len(c.smem_index)
        for si, li in zip(c.smem_index, lane_idx):
            sr, st = self.compile_expr(si)
            sr = self.to_s32(sr, st)
            if li is not None:
                lr, lt = self.compile_expr(li)
                t = self.newreg("s32")
                self.emit("add.s32 %s, %s, %s;" % (t, sr, self.to_s32(lr, lt)))
                coords.append(t)
            else:
                coords.append(sr)
        shape = self.smem_shapes[c.smem]
        nbytes = self.smem_nbytes[c.smem]
        daddr = self.smem_addr(c.smem, self.flat_offset(coords, shape), nbytes)
        # global source address
        gshape = self.buf_shapes[c.src]
        sidx = self.index_regs(c.src_index)
        base, _ = self.env["buf:" + c.src]
        saddr = self.global_addr(base, self.flat_offset(sidx, gshape),
                                 self.buf_nbytes[c.src])
        self.emit("cp.async.cg.shared.global [%s], [%s], %d;"
                  % (daddr, saddr, c.nbytes))

    def stmt_CpAsyncCommit(self, s):
        self.emit("cp.async.commit_group;")

    def stmt_CpAsyncWait(self, s):
        self.emit("cp.async.wait_group %d;" % (s.group,))

    def stmt_Write(self, w):
        ptype, nbytes = self.resolve_dtype(w.value_dtype, w.output)
        vals = [self.env[n] for n in w.values]
        idx = self.index_regs(w.index)
        if w.output in self.smem:
            shape = self.smem_shapes[w.output]
            addr = self.smem_addr(w.output, self.flat_offset(idx, shape), nbytes)
            space = "shared"
        else:
            shape = self.buf_shapes[w.output]
            base, _ = self.env["buf:" + w.output]
            addr = self.global_addr(base, self.flat_offset(idx, shape), nbytes)
            space = "global"
        regs = [r for r, _ in vals]
        n = len(regs)
        if w.atomic:
            if space != "global" or n != 1:
                raise ValueError("atomic write: only single global supported")
            self.emit("red.global.add.%s [%s], %s;" % (ptype, addr, regs[0]))
        elif n == 1:
            self.emit("st.%s.%s [%s], %s;" % (space, ptype, addr, regs[0]))
        elif n == 2:
            self.emit("st.%s.v2.%s [%s], {%s, %s};"
                      % (space, ptype, addr, regs[0], regs[1]))
        elif n == 4:
            self.emit("st.%s.v4.%s [%s], {%s, %s, %s, %s};"
                      % (space, ptype, addr, regs[0], regs[1], regs[2], regs[3]))
        elif n == 3:
            self.emit("st.%s.v2.%s [%s], {%s, %s};"
                      % (space, ptype, addr, regs[0], regs[1]))
            # scalar store of the third at +1 element
            one = self.newreg("s32")
            self.emit("mov.s32 %s, 1;" % one)
            # recompute offset+1 (only single-dim supported here)
            if len(idx) != 1:
                raise ValueError("3-value write with multi-dim index")
            off2 = self.newreg("s32")
            self.emit("add.s32 %s, %s, %s;" % (off2, idx[0], one))
            if space == "global":
                b, _ = self.env["buf:" + w.output]
                a2 = self.global_addr(b, off2, nbytes)
            else:
                a2 = self.smem_addr(w.output, off2, nbytes)
            self.emit("st.%s.%s [%s], %s;" % (space, ptype, a2, regs[2]))
        else:
            raise ValueError("Write with %d values" % n)

    # -- top level ----------------------------------------------------------
    def prepass_smem(self, node):
        tn = type(node).__name__
        if tn == "Smem":
            ptype, nbytes = ptx_dtype(node.dtype)
            size = nbytes
            for e in node.extents:
                size *= int(e)
            align = node.align or 4
            self.smem_info.append((node.name, size, align))
            self.smem.add(node.name)
            self.smem_shapes[node.name] = tuple(int(e) for e in node.extents)
            self.smem_nbytes[node.name] = nbytes
            self.smem_ptype[node.name] = ptype
            return
        for f in getattr(node, "__dataclass_fields__", {}):
            v = getattr(node, f)
            xs = v if isinstance(v, (list, tuple)) else [v]
            for x in xs:
                if hasattr(x, "__dataclass_fields__"):
                    self.prepass_smem(x)

    def layout_smem(self):
        """Static decls when they fit; else one dynamic extern array."""
        total = sum(size for _, size, _ in self.smem_info)
        if total <= self.STATIC_SMEM_LIMIT:
            for name, size, align in self.smem_info:
                self.smem_decls.append(
                    ".shared .align %d .b8 %s[%d];" % (align, name, size))
            self.smem_dynamic = False
        else:
            # meta.json's smem_bytes covers the pool; carve it up like emmy.
            self.smem_decls.append(
                ".extern .shared .align 16 .b8 __emmy_dyn_smem[];")
            self.smem_dynamic = True
            off = 0
            for name, size, align in self.smem_info:
                a = align or 1
                off = (off + a - 1) // a * a
                self.smem_offsets[name] = off
                off += size

    def compile(self, graph, kernel_node, kmeta):
        try:
            from emmy.compiler.graph import Graph  # noqa
        except ImportError:
            pass
        kop = kernel_node.op
        entry = kop.name
        self.arg_order = list(kmeta["arg_order"])
        self.grid = list(kmeta["grid"])
        self.block = list(kmeta["block"])
        for a in kmeta["args"]:
            ptype, nbytes = ptx_dtype(a["dtype"])
            self.buf_shapes[a["name"]] = tuple(a["shape"])
            self.buf_nbytes[a["name"]] = nbytes
            self.buf_dtypes[a["name"]] = a["dtype"]

        tile = None
        for s in kop.body:
            self.prepass_smem(s)
            if type(s).__name__ == "Tile":
                tile = s
        if tile is None:
            raise ValueError("no Tile in kernel body")
        self.layout_smem()

        # ---- prologue ----
        for i, aname in enumerate(self.arg_order):
            rd = self.newreg("u64")
            self.emit("ld.param.u64 %s, [p_%s];" % (rd, aname))
            cv = self.newreg("u64")
            self.emit("cvta.to.global.u64 %s, %s;" % (cv, rd))
            self.env["buf:" + aname] = (cv, "u64")
        tid = self.newreg("u32")
        ctaid = self.newreg("u32")
        ntid = self.newreg("u32")
        self.emit("mov.u32 %s, %%tid.x;" % tid)
        self.emit("mov.u32 %s, %%ctaid.x;" % ctaid)
        self.emit("mov.u32 %s, %%ntid.x;" % ntid)
        tmp = self.newreg("s32")
        gid = self.newreg("s32")
        self.emit("mul.lo.s32 %s, %s, %s;" % (tmp, ctaid, ntid))
        self.emit("add.s32 %s, %s, %s;" % (gid, tmp, tid))
        self.env["_gid"] = (gid, "s32")
        lane = self.newreg("s32")
        warp = self.newreg("s32")
        self.emit("and.b32 %s, %s, 31;" % (lane, tid))
        self.emit("shr.u32 %s, %s, 5;" % (warp, tid))
        self.env["lane"] = (lane, "s32")
        self.env["warp"] = (warp, "s32")

        # tile body (also inits loop-free accumulators, if any)
        self.init_accums(tile.body)
        self.stmt_Tile(tile)
        self.emit("ret;")

        # ---- assemble ----
        out = []
        out.append(".version 8.0")
        out.append(".target sm_89")
        out.append(".address_size 64")
        out.append("")
        out.extend(self.smem_decls)
        if self.smem_decls:
            out.append("")
        params = ",\n".join("        .param .u64 p_%s" % a
                            for a in self.arg_order)
        out.append(".visible .entry %s(" % entry)
        out.append(params)
        out.append(")")
        out.append("{")
        if self.counts["p"]:
            out.append("        .reg .pred %%p<%d>;" % self.counts["p"])
        if self.counts["f"]:
            out.append("        .reg .f32 %%f<%d>;" % self.counts["f"])
        if self.counts["d"]:
            out.append("        .reg .f64 %%d<%d>;" % self.counts["d"])
        if self.counts["h"]:
            out.append("        .reg .f16 %%h<%d>;" % self.counts["h"])
        if self.counts["r"]:
            out.append("        .reg .b32 %%r<%d>;" % self.counts["r"])
        if self.counts["rd"]:
            out.append("        .reg .b64 %%rd<%d>;" % self.counts["rd"])
        out.append("")
        out.extend(self.lines)
        out.append("}")
        out.append("")
        text = "\n".join(out)
        # PTX rejects non-ASCII anywhere
        text.encode("ascii")
        return text


def main():
    ir_path = Path(sys.argv[1])
    data = json.loads(ir_path.read_text())
    from emmy.compiler.graph import Graph
    graph = Graph.from_dict(data)
    kernel_node = None
    for name, node in graph.nodes.items():
        if type(node.op).__name__ == "KernelOp":
            kernel_node = node
            break
    if kernel_node is None:
        raise SystemExit("no KernelOp found in %s" % ir_path)
    meta = json.loads((ir_path.parent / "meta.json").read_text())
    entry = kernel_node.op.name
    kmeta = None
    for k in meta["kernels"]:
        if k["entry"] == entry:
            kmeta = k
            break
    if kmeta is None:
        kmeta = meta["kernels"][0]
    em = Emitter()
    sys.stdout.write(em.compile(graph, kernel_node, kmeta))


if __name__ == "__main__":
    main()
