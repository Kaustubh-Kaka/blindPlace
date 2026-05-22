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

## Results

Evaluation on the IBM ICCAD04 benchmarks demonstrates the effectiveness of the SA engine. The solver produces valid (0 overlaps) placements rapidly (under 35 seconds total runtime) and outperforms the SA baseline by 29.1% on average, while remaining highly competitive with RePlAce.

```text
--------------------------------------------------------------------------------
    Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
--------------------------------------------------------------------------------
        ibm01    1.2253    1.3166    0.9976     +6.9%      -22.8%         0
        ibm02    1.6800    1.9072    1.8370    +11.9%       +8.5%         0
        ibm03    1.4100    1.7401    1.3222    +19.0%       -6.6%         0
        ibm04    1.4101    1.5037    1.3024     +6.2%       -8.3%         0
        ibm06    1.7197    2.5057    1.6187    +31.4%       -6.2%         0
        ibm07    1.4950    2.0229    1.4633    +26.1%       -2.2%         0
        ibm08    1.5582    1.9239    1.4285    +19.0%       -9.1%         0
        ibm09    1.1363    1.3875    1.1194    +18.1%       -1.5%         0
        ibm10    1.4037    2.1108    1.5009    +33.5%       +6.5%         0
        ibm11    1.2354    1.7111    1.1774    +27.8%       -4.9%         0
        ibm12    1.6507    2.8261    1.7261    +41.6%       +4.4%         0
        ibm13    1.4011    1.9141    1.3355    +26.8%       -4.9%         0
        ibm14    1.6033    2.2750    1.5436    +29.5%       -3.9%         0
        ibm15    1.6061    2.3000    1.5159    +30.2%       -5.9%         0
        ibm16    1.5323    2.2337    1.4780    +31.4%       -3.7%         0
        ibm17    1.7437    3.6726    1.6446    +52.5%       -6.0%         0
        ibm18    1.7941    2.7755    1.7722    +35.4%       -1.2%         0
--------------------------------------------------------------------------------
          AVG    1.5062    2.1251    1.4578    +29.1%       -3.3%         0

Total runtime: 34.71s
```
