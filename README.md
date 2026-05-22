# Simulated Annealing Macro Placer

This directory contains a custom Simulated Annealing (SA) macro placer submitted for the Partcl / HRT Macro Placement Challenge.

The solver leverages a pure-NumPy hot path to rapidly explore the solution space of hard macro placements. By strictly rejecting overlapping moves via $O(N)$ broadcasting, the solver completely avoids the need for complex overlap penalty terms in its cost function, allowing it to optimize purely for Half-Perimeter Wirelength (HPWL) proxy costs.

## Algorithm Breakdown

### 1. Warm Start: Spiral Legalization
Rather than starting from a random configuration, the solver initializes by pulling the default coordinates provided by the benchmark (which often contain connectivity hints). 
It then processes the hard macros in descending order of area. For each macro, it performs a deterministic, outward spiral grid search to find the nearest legal coordinate that has absolutely zero overlap with any previously placed macros. This ensures the SA engine starts from a valid layout.

### 2. Connectivity Extraction & HPWL Cost
Before annealing, the solver parses the netlist to extract an edge graph of all connected hard macros. 
* **Edge Weights:** Edges are weighted inversely to their net fanout using the formula `1 / (fanout - 1)`.
* **Cost Function:** The primary cost function evaluated during the annealing loop is the sum of the weighted Manhattan distances (`dx + dy`) between connected hard macros. This serves as an extremely fast proxy for wirelength.

### 3. Pure NumPy Simulated Annealing Engine
The core engine runs for a set number of steps (e.g., 50,000) using an exponential cooling schedule. At each step, a movable macro is selected at random, and one of three spatial perturbations is applied:

* **Translate (50% probability):** A pure Gaussian shift in the X and Y dimensions. The variance of the displacement scales proportionally to the square root of the current temperature (`∝ √T`).
* **Neighbour-Biased Swap (30% probability):** Swaps the positions of two macros. It is heavily biased (70% chance) to swap macros that share a net, which is highly effective at untangling dense, inter-connected clusters.
* **Pull to Neighbour (20% probability):** Selects a connected neighbour and pulls the current macro a random fractional distance (5%–30%) toward it.

**Zero-Overlap Strict Rejection:**
The defining feature of this SA engine is that it never allows overlaps to occur during the hot path. Any proposed move is instantly checked for overlaps against all other macros. Because the solver uses pre-computed separation matrices and NumPy broadcasting, this $O(N)$ check is practically free. If an overlap is detected, the move is immediately rejected. This guarantees that every accepted state is physically valid and eliminates the need to tune an overlap penalty hyperparameter.

### 4. Soft-Macro Co-Optimization
Hard macro placement cannot be done in isolation without considering the standard cell logic. Periodically (e.g., every 1,000 accepted SA moves), the annealing loop pauses to invoke the underlying force-directed standard cell engine (`optimize_stdcells()`). This allows the soft-macro clusters to organically drift and reposition themselves in response to the updated hard macro coordinates, ensuring the wirelength cost reflects the true logic topology.

### 5. Guaranteed Push-Apart Legalization
Although the SA engine strictly forbids overlaps, minor rounding or boundary conditions might occasionally leave macros microscopically touching. As a final post-processing step, the solver runs a deterministic push-apart pass. This loop iterates over all hard macros and mechanically expands the gaps between any intersecting bounding boxes. The loop runs without an iteration cap until exactly 0 overlaps remain, guaranteeing a perfectly legal final placement tensor.

## How to Run

To evaluate this solver against the challenge benchmarks using the provided evaluation harness:

```bash
# Evaluate on a single benchmark (e.g., ibm01)
uv run evaluate submissions/mine/sa_solver.py -b ibm01

# Evaluate on all ICCAD04 benchmarks
uv run evaluate submissions/mine/sa_solver.py --all

# Evaluate on modern commercial NG45 designs
uv run evaluate submissions/mine/sa_solver.py --ng45
```
