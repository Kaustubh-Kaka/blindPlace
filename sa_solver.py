"""
Simulated Annealing Macro Placer
=================================
Submission for the Partcl / HRT Macro Placement Challenge 2026.

    uv run evaluate submissions/sa_placer.py -b ibm01
    uv run evaluate submissions/sa_placer.py --all

Algorithm
---------
1. Warm start  — minimum-displacement legaliser: places macros in
   area-descending order, using a spiral grid search to find the nearest
   legal position to each macro's initial location (preserves connectivity).

2. Simulated annealing on hard macros (pure numpy hot path):
     • translate          — Gaussian displacement, amplitude ∝ √T
     • neighbour-biased swap — prefer swapping net-connected pairs
     • move-toward-neighbour — pull macro toward a connected macro

   Overlap is checked per-macro with numpy broadcasting (O(N)), not O(N²).
   The precomputed sep_x / sep_y separation matrices make this near-free.
   Any move that creates an overlap is immediately rejected — no penalty
   term needed, so the cost function is pure wirelength.

3. Soft macro co-optimisation — plc.optimize_stdcells() every
   stdcell_interval accepted moves.

4. Guaranteed legalisation — push-apart loop with no iteration cap.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ─────────────────────────────────────────────────────────────────────────────
# plc loading (mirrors will_seed approach, supports IBM + NG45)
# ─────────────────────────────────────────────────────────────────────────────

def _load_plc(name: str):
    try:
        from macro_place.loader import load_benchmark_from_dir, load_benchmark
        root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        if root.exists():
            _, plc = load_benchmark_from_dir(str(root))
            return plc
        ng45 = {
            "ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile",
        }
        d = ng45.get(name)
        if d:
            base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
            if (base / "netlist.pb.txt").exists():
                _, plc = load_benchmark(
                    str(base / "netlist.pb.txt"), str(base / "initial.plc"))
                return plc
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Edge extraction — weighted adjacency among hard macros
# ─────────────────────────────────────────────────────────────────────────────

def _extract_edges(benchmark: Benchmark, plc):
    """Return (edges [E,2] int64, weights [E] float32) in hard-macro index space.
    Weight = 1/(fanout-1) per net, accumulated over shared nets."""
    n_hard = benchmark.num_hard_macros
    # Map plc module index → hard macro tensor index
    plc_to_hard = {}
    for t_idx, p_idx in enumerate(benchmark.hard_macro_indices):
        plc_to_hard[p_idx] = t_idx

    edge_dict: dict = {}
    try:
        for driver, sinks in plc.nets.items():
            macros: set = set()
            for pin in [driver] + sinks:
                parent = pin.split("/")[0]
                # find plc module index by name
                for p_idx, t_idx in plc_to_hard.items():
                    try:
                        if plc.modules_w_pins[p_idx].get_name() == parent:
                            macros.add(t_idx)
                            break
                    except Exception:
                        pass
            if len(macros) >= 2:
                ml = sorted(macros)
                w = 1.0 / (len(ml) - 1)
                for a in range(len(ml)):
                    for b in range(a + 1, len(ml)):
                        pair = (ml[a], ml[b])
                        edge_dict[pair] = edge_dict.get(pair, 0.0) + w
    except Exception:
        pass

    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)

    edges = np.array(list(edge_dict.keys()), dtype=np.int64)
    weights = np.array([edge_dict[tuple(e)] for e in edges], dtype=np.float32)
    return edges, weights


def _build_neighbours(edges: np.ndarray, n: int) -> list:
    """Adjacency list: neighbours[i] = list of hard macro indices connected to i."""
    neighbours: list = [[] for _ in range(n)]
    for i, j in edges:
        neighbours[i].append(int(j))
        neighbours[j].append(int(i))
    return neighbours


# ─────────────────────────────────────────────────────────────────────────────
# Warm start: minimum-displacement legaliser (from will_seed)
# ─────────────────────────────────────────────────────────────────────────────

def _legalize(pos: np.ndarray,
              movable: np.ndarray,
              sizes: np.ndarray,
              cw: float, ch: float) -> np.ndarray:
    """Place macros in area-descending order. For each macro find the nearest
    legal position to its initial location via a spiral grid search."""
    n = len(pos)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2   # [n, n]
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    GAP = 0.05

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        # Check if current position is already legal
        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            conflict = (dx < sep_x[idx] + GAP) & (dy < sep_y[idx] + GAP) & placed
            conflict[idx] = False
            if not conflict.any():
                placed[idx] = True
                continue

        # Spiral search for nearest legal position
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best_pos = legal[idx].copy()
        best_dist = float('inf')

        for r in range(1, 200):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx])
                    if placed.any():
                        dx = np.abs(cx - legal[:, 0])
                        dy = np.abs(cy - legal[:, 1])
                        conflict = (dx < sep_x[idx] + GAP) & (dy < sep_y[idx] + GAP) & placed
                        conflict[idx] = False
                        if conflict.any():
                            continue
                    dist = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = np.array([cx, cy])
                        found = True
            if found:
                break

        legal[idx] = best_pos
        placed[idx] = True

    return legal


# ─────────────────────────────────────────────────────────────────────────────
# HPWL cost (numpy, edges only)
# ─────────────────────────────────────────────────────────────────────────────

def _wl_cost(pos: np.ndarray, edges: np.ndarray, weights: np.ndarray) -> float:
    if len(edges) == 0:
        return 0.0
    dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
    dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
    return float((weights * (dx + dy)).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Per-macro O(N) overlap check via numpy broadcasting
# ─────────────────────────────────────────────────────────────────────────────

def _has_overlap(idx: int, pos: np.ndarray,
                 sep_x: np.ndarray, sep_y: np.ndarray,
                 n: int, gap: float = 0.05) -> bool:
    """True if macro idx overlaps any other macro."""
    dx = np.abs(pos[idx, 0] - pos[:n, 0])
    dy = np.abs(pos[idx, 1] - pos[:n, 1])
    ov = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
    ov[idx] = False
    return bool(ov.any())


def _count_overlaps(pos: np.ndarray,
                    sep_x: np.ndarray, sep_y: np.ndarray,
                    n: int, gap: float = 0.0) -> int:
    """Count total overlapping hard macro pairs."""
    count = 0
    for i in range(n):
        dx = np.abs(pos[i, 0] - pos[:n, 0])
        dy = np.abs(pos[i, 1] - pos[:n, 1])
        ov = (dx < sep_x[i] + gap) & (dy < sep_y[i] + gap)
        ov[i] = False
        count += int(ov[:i].sum())   # count each pair once
    return count


# ─────────────────────────────────────────────────────────────────────────────
# plc sync
# ─────────────────────────────────────────────────────────────────────────────

def _push_hard_to_plc(pos: np.ndarray, benchmark: Benchmark, plc) -> None:
    for t_idx, p_idx in enumerate(benchmark.hard_macro_indices):
        try:
            plc.modules_w_pins[p_idx].set_pos(float(pos[t_idx, 0]), float(pos[t_idx, 1]))
        except Exception:
            pass


def _pull_all_from_plc(pos: np.ndarray, benchmark: Benchmark, plc) -> np.ndarray:
    pos = pos.copy()
    for t_idx, p_idx in enumerate(benchmark.hard_macro_indices):
        try:
            x, y = plc.modules_w_pins[p_idx].get_pos()
            pos[t_idx, 0] = x
            pos[t_idx, 1] = y
        except Exception:
            pass
    return pos


def _optimize_soft(plc, canvas_size: float,
                   num_steps: List[int],
                   max_dist: Optional[List[float]] = None) -> None:
    if max_dist is None:
        s = canvas_size / 100
        max_dist = [s, s, s]
    try:
        plc.optimize_stdcells(
            use_current_loc=False, move_stdcells=True, move_macros=False,
            log_scale_conns=False, use_sizes=False, io_factor=1.0,
            num_steps=num_steps, max_move_distance=max_dist,
            attract_factor=[100.0, 1e-3, 1e-5],
            repel_factor=[0.0, 1e6, 1e7],
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Guaranteed push-apart legaliser (post-SA)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_overlaps(placement: torch.Tensor,
                      sizes: torch.Tensor,
                      hard_mask: torch.Tensor,
                      canvas_w: float,
                      canvas_h: float) -> torch.Tensor:
    """Push overlapping hard macros fully apart.
    Runs until no overlaps remain — no iteration cap."""
    placement = placement.clone()
    hard_idx = torch.where(hard_mask)[0].tolist()

    while True:
        moved = False
        for ii, i in enumerate(hard_idx):
            xi, yi = placement[i, 0].item(), placement[i, 1].item()
            wi, hi = sizes[i, 0].item(), sizes[i, 1].item()
            ax0, ax1 = xi - wi / 2, xi + wi / 2
            ay0, ay1 = yi - hi / 2, yi + hi / 2

            for j in hard_idx[ii + 1:]:
                xj, yj = placement[j, 0].item(), placement[j, 1].item()
                wj, hj = sizes[j, 0].item(), sizes[j, 1].item()
                bx0, bx1 = xj - wj / 2, xj + wj / 2
                by0, by1 = yj - hj / 2, yj + hj / 2

                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)

                if ox > 0 and oy > 0:
                    moved = True
                    gap = 1e-4
                    if ox <= oy:
                        push = ox / 2 + gap
                        if xi <= xj:
                            placement[i, 0] -= push;  placement[j, 0] += push
                            xi -= push; ax0 -= push; ax1 -= push
                        else:
                            placement[i, 0] += push;  placement[j, 0] -= push
                            xi += push; ax0 += push; ax1 += push
                    else:
                        push = oy / 2 + gap
                        if yi <= yj:
                            placement[i, 1] -= push;  placement[j, 1] += push
                            yi -= push; ay0 -= push; ay1 -= push
                        else:
                            placement[i, 1] += push;  placement[j, 1] -= push
                            yi += push; ay0 += push; ay1 += push

                    for k in (i, j):
                        wk, hk = sizes[k, 0].item(), sizes[k, 1].item()
                        placement[k, 0] = float(np.clip(placement[k, 0].item(), wk / 2, canvas_w - wk / 2))
                        placement[k, 1] = float(np.clip(placement[k, 1].item(), hk / 2, canvas_h - hk / 2))

        if not moved:
            break  # no overlaps remain

    return placement


# ─────────────────────────────────────────────────────────────────────────────
# Main SA engine
# ─────────────────────────────────────────────────────────────────────────────

class SAMacroPlacer:
    def __init__(
        self,
        n_steps: int = 50_000,
        T_start_frac: float = 0.15,   # T_start = max(W,H) * frac  (mirrors will_seed)
        T_end_frac: float = 0.001,    # T_end   = max(W,H) * frac
        stdcell_interval: int = 1000,
        stdcell_steps: Optional[List[int]] = None,
        time_limit: float = 50.0,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.n_steps = n_steps
        self.T_start_frac = T_start_frac
        self.T_end_frac = T_end_frac
        self.stdcell_interval = stdcell_interval
        self.stdcell_steps = stdcell_steps or [10, 10, 10]
        self.time_limit = time_limit
        self.seed = seed
        self.verbose = verbose

    def place(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        bm = benchmark
        W, H = float(bm.canvas_width), float(bm.canvas_height)
        canvas_size = max(W, H)
        n_hard = bm.num_hard_macros

        sizes_np = bm.macro_sizes[:n_hard].numpy().astype(np.float64)
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2
        movable_np = bm.get_movable_mask()[:n_hard].numpy()
        movable_idx = np.where(movable_np)[0].tolist()

        if not movable_idx:
            return bm.macro_positions.clone().float()

        # Precompute separation matrices once — reused every SA step
        sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2   # [n_hard, n_hard]
        sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

        # Extract net edges
        edges = np.zeros((0, 2), dtype=np.int64)
        weights = np.zeros(0, dtype=np.float32)
        neighbours: list = [[] for _ in range(n_hard)]
        if plc is not None:
            try:
                edges, weights = _extract_edges(bm, plc)
                neighbours = _build_neighbours(edges, n_hard)
            except Exception:
                pass

        # ── 1. Minimum-displacement legalisation warm start ──────────────────
        init_pos = bm.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        pos = _legalize(init_pos, movable_np, sizes_np, W, H)

        if plc is not None:
            _push_hard_to_plc(pos, bm, plc)
            _optimize_soft(plc, canvas_size,
                           num_steps=[100, 100, 100],
                           max_dist=[canvas_size / 50] * 3)
            pos = _pull_all_from_plc(pos, bm, plc)

        # ── 2. Temperature schedule ──────────────────────────────────────────
        T_start = canvas_size * self.T_start_frac
        T_end   = canvas_size * self.T_end_frac

        # ── 3. Initial cost ──────────────────────────────────────────────────
        current_cost = _wl_cost(pos, edges, weights)
        best_cost = current_cost
        best_pos = pos.copy()

        accepted = 0
        since_stdcell = 0
        start = time.time()
        log_every = max(1, self.n_steps // 10)

        # ── 4. SA loop (pure numpy) ──────────────────────────────────────────
        for step in range(self.n_steps):
            if time.time() - start > self.time_limit:
                if self.verbose:
                    print(f"[SA] Time limit at step {step:,}")
                break

            frac = step / self.n_steps
            T = T_start * (T_end / T_start) ** frac
            shift = T * (0.3 + 0.7 * (1 - frac))   # mirrors will_seed

            r = random.random()
            i = random.choice(movable_idx)
            old_x, old_y = pos[i, 0], pos[i, 1]

            if r < 0.50:
                # ── Translate ────────────────────────────────────────────────
                pos[i, 0] = np.clip(old_x + random.gauss(0, shift), half_w[i], W - half_w[i])
                pos[i, 1] = np.clip(old_y + random.gauss(0, shift), half_h[i], H - half_h[i])

                if _has_overlap(i, pos, sep_x, sep_y, n_hard):
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    continue

                new_cost = _wl_cost(pos, edges, weights)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_cost = new_cost
                    accepted += 1; since_stdcell += 1
                    if plc is not None:
                        try:
                            plc.modules_w_pins[bm.hard_macro_indices[i]].set_pos(
                                pos[i, 0], pos[i, 1])
                        except Exception:
                            pass
                else:
                    pos[i, 0] = old_x; pos[i, 1] = old_y

            elif r < 0.80:
                # ── Neighbour-biased swap ────────────────────────────────────
                nb = [j for j in neighbours[i] if movable_np[j]]
                if nb and random.random() < 0.7:
                    j = random.choice(nb)
                else:
                    j = random.choice(movable_idx)
                if i == j:
                    continue

                old_jx, old_jy = pos[j, 0], pos[j, 1]
                pos[i, 0] = np.clip(old_jx, half_w[i], W - half_w[i])
                pos[i, 1] = np.clip(old_jy, half_h[i], H - half_h[i])
                pos[j, 0] = np.clip(old_x,  half_w[j], W - half_w[j])
                pos[j, 1] = np.clip(old_y,  half_h[j], H - half_h[j])

                if _has_overlap(i, pos, sep_x, sep_y, n_hard) or \
                   _has_overlap(j, pos, sep_x, sep_y, n_hard):
                    pos[i, 0] = old_x;  pos[i, 1] = old_y
                    pos[j, 0] = old_jx; pos[j, 1] = old_jy
                    continue

                new_cost = _wl_cost(pos, edges, weights)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_cost = new_cost
                    accepted += 1; since_stdcell += 1
                    if plc is not None:
                        try:
                            plc.modules_w_pins[bm.hard_macro_indices[i]].set_pos(
                                pos[i, 0], pos[i, 1])
                            plc.modules_w_pins[bm.hard_macro_indices[j]].set_pos(
                                pos[j, 0], pos[j, 1])
                        except Exception:
                            pass
                else:
                    pos[i, 0] = old_x;  pos[i, 1] = old_y
                    pos[j, 0] = old_jx; pos[j, 1] = old_jy

            else:
                # ── Move toward connected neighbour ──────────────────────────
                if not neighbours[i]:
                    continue
                j = random.choice(neighbours[i])
                alpha = random.uniform(0.05, 0.3)
                pos[i, 0] = np.clip(old_x + alpha * (pos[j, 0] - old_x), half_w[i], W - half_w[i])
                pos[i, 1] = np.clip(old_y + alpha * (pos[j, 1] - old_y), half_h[i], H - half_h[i])

                if _has_overlap(i, pos, sep_x, sep_y, n_hard):
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    continue

                new_cost = _wl_cost(pos, edges, weights)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_cost = new_cost
                    accepted += 1; since_stdcell += 1
                    if plc is not None:
                        try:
                            plc.modules_w_pins[bm.hard_macro_indices[i]].set_pos(
                                pos[i, 0], pos[i, 1])
                        except Exception:
                            pass
                else:
                    pos[i, 0] = old_x; pos[i, 1] = old_y

            # ── Soft macro co-optimisation ────────────────────────────────────
            if plc is not None and since_stdcell >= self.stdcell_interval:
                since_stdcell = 0
                _optimize_soft(plc, canvas_size, num_steps=self.stdcell_steps)
                pos = _pull_all_from_plc(pos, bm, plc)
                current_cost = _wl_cost(pos, edges, weights)

            if current_cost < best_cost:
                best_cost = current_cost
                best_pos = pos.copy()

            if self.verbose and (step + 1) % log_every == 0:
                elapsed = time.time() - start
                ov = _count_overlaps(pos, sep_x, sep_y, n_hard)
                print(
                    f"[SA] {step+1:>6,}/{self.n_steps:,}  "
                    f"T={T:.3e}  wl={current_cost:.2f}  "
                    f"best={best_cost:.2f}  overlaps={ov}  "
                    f"acc={accepted/(step+1)*100:.0f}%  {time.time()-start:.0f}s"
                )

        elapsed = time.time() - start
        ov_final = _count_overlaps(best_pos, sep_x, sep_y, n_hard)
        if self.verbose:
            print(f"[SA] Done {elapsed:.1f}s — wl={best_cost:.2f}  overlaps={ov_final}")

        # Reconstruct full [num_macros, 2] tensor
        full = bm.macro_positions.clone().float()
        full[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        return full


# ─────────────────────────────────────────────────────────────────────────────
# Challenge harness entry point
# ─────────────────────────────────────────────────────────────────────────────

class Placer:
    """Called by the harness as Placer().place(benchmark)."""

    def __init__(self):
        self._sa = SAMacroPlacer(
            n_steps=50_000,
            stdcell_interval=1000,
            stdcell_steps=[10, 10, 10],
            time_limit=50.0,
            verbose=True,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark.name)

        placement = self._sa.place(benchmark, plc)

        # Guaranteed legalisation — push apart any residual overlaps
        placement = _resolve_overlaps(
            placement,
            benchmark.macro_sizes.float(),
            benchmark.get_hard_macro_mask(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
        )

        # Restore fixed macros
        fixed = benchmark.macro_fixed
        placement[fixed] = benchmark.macro_positions[fixed].float()

        # Report any remaining overlaps
        sizes_t = benchmark.macro_sizes.float()
        hard_mask = benchmark.get_hard_macro_mask()
        pos_np = placement[:benchmark.num_hard_macros].numpy().astype(np.float64)
        sizes_np = sizes_t[:benchmark.num_hard_macros].numpy().astype(np.float64)
        sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
        sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2
        ov = _count_overlaps(pos_np, sep_x, sep_y, benchmark.num_hard_macros)
        if ov > 0:
            print(f"[Placer] WARNING: {ov} overlapping pairs remain after legalisation")

        return placement