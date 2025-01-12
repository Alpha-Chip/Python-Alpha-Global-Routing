# PAGR: Python Alpha Global Routing 

Python global router which was made for [ISPD 2024](https://liangrj2014.github.io/ISPD24_contest/) contest.

## Code description

Different variants of routers

* `router_simple.py` - pure python global routing. Less than 200 lines of code. Single thread and very slow. Made for educational purposes to present basic working algorithm. Deterministic solution.
* `router.py` - multithread global routing with different Steiner Tree Problem (STP) solvers.

Control of router parameters is done from `config.py`:

* **STP_ROUTER** - can be one of: `networkx`, `c_stp`, `scip_jack`. (Default: `c_stp`). 
* * `networkx` - fully python STP solver based on networkx Python module. It uses function [`steiner_tree`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.steinertree.steiner_tree.html) with method `mehlhorn` (Mehlhorn, Kurt. 1988. ‘A Faster Approximation Algorithm for the Steiner Problem in Graphs’. Information Processing Letters 27 (3): 125–28.).
* * `c_stp` - C-based Steiner Tree Solver on graphs is taken from [this repo](https://github.com/ruykaji/steiner-tree-problem). Thanks to @ruykaji. Algorithm is based on following paper: _Wu Y. F., Widmayer P., Wong C. K. A faster approximation algorithm for the Steiner problem in graphs //Acta informatica. – 1986. – Т. 23. – №. 2. – С. 223-229._ More information: [stp_solver/README.md](stp_solver/README.md). Method is faster and gives better quality than networkx.
* * `scip_jack` - [SCIP-JACK STP solver](https://scipjack.zib.de/). .Algorithm searches for exact STP solution, but requires license for commercial usage If algorithm can't find solution in reasonable time it fall back on `c_stp`.

* **SORT_NETS_BY** - can be one of: `random`, `area`, `min_intersection`. (Default: `random`).
* * `random` - use random order of nets for processing. Works ok in case USE_PROCESSING_LOCK is True.
* * `area` - use descending area order of nets for processing. Larger nets processed first. 
* * `min_intersection` - try to arrange nets in a way that they don't intersect in big blocks. Potentially good, but sorting is slow for large circuits.

* **SKIP_DOUBLE_STAGE** - Skip double stage algorithm if rectangle side is less than this value. (Default: 30). 
You can put value very large to use single stage slow but more precise algorithm for all nets.

* **MULTIPLY_COST** = 5 # We need this because we do rounding of capacity and store capacity in int16 matrix
* **VIA_COST** = 50 # How much cost for via
* **GAP** = 0 # Size of border about set of terminals - larger give more flexibility to find lower cost path, but complexity of problem is increased (doesn't really give better result for > 0)
* **BATCH_SIZE** = 250000 # Set as max as possible (only needed to solve strange problem with "out of memory" in windows)
* **USE_PROCESSING_LOCK** = True # If true processing regions will be locked for other processes
* **PROCESSING_LOCK_TIMEOUT** = 0.0 # How much time to wait to try again to process region
* **TIMEOUT_FOR_SCIPJACK** = 30. # If `scip_jack` STP-solver works more than TIMEOUT_FOR_SCIPJACK seconds it's terminated and used `c_stp` solver.
* **SHARED_MEMORY_NAME** = 'capacity_matrix_router' # can be any name (if you run two routers at the same time names must be different!)
* **DEBUG** = False # Do some printing if True

## Usage

```bash
python router.py -net mempool_tile.net -cap mempool_tile.cap -output out.txt
```

To see all available parameters run any router without parameters:

```bash
python router.py
```

## Input data

We use as [input format](https://drive.google.com/file/d/1yEgcjHAZOyFHKlfYhzHe8ZeZuefEK2sP/view) for global routing proposed by ISPD 2024 Contest organizers.
There are 2 input files:

<details>
<summary style="margin-left: 25px;">Cap file (press to look format)</summary>
<div style="margin-left: 25px;">
The routing resource Cap file follows this format:

```
    # Dimensions of GCell graph
    nLayers xSize ySize
    # Weights of performance metrics
    UnitLengthWireCost UnitViaCost OFWeight[0] OFWeight[1] OFWeight[2] · · ·
    # Lengths of horizontal GCell edges (edge count = xSize - 1)  
    HorizontalGCellEdgeLengths[0] HorizontalGCellEdgeLengths[1] HorizontalGCellEdgeLengths[2] · · ·
    # Lengths of vertical GCell edges (edge count = ySize - 1)
    VerticalGCellEdgeLengths[0] VerticalGCellEdgeLengths[1] VerticalGCellEdgeLengths[2] · · ·
    # Information for the 0-th layer
    ## Layer name, prefered direction and minimum length of a wire at this metal
    layer. For direction, 0 represents horizontal, while 1 represents vertical.
    layerName layerDirection layerMinLength
    ## Routing capacities of GCell edges at the 0-th layer
    ### Capacities of GCell at [x(0), y(0)], Capacities of GCell at [x(1), y(0)], ...
    10 10 10 · · ·
    ### Capacities of GCell at [x(0), y(1)], Capacities of GCell at [x(1), y(1)], ...
    10 10 10 · · ·
    · · · · · · · · ·
    ## Information for the 1-th layer
    · · · · · · · · ·
```
</div>
</details>

<details>
<summary style="margin-left: 25px;">Net file (press to look format)</summary>
<div style="margin-left: 25px;">
The Net file follows this format:

```
    # Net name
    Net0
    (
   `    # access point locations (layer, x, y) for pin 0. Selecting any one of these
        locations is sufficient for pin 0.
        [(location of access point 0), (location of access point 1), · · · ]
        # access point locations for pin 1
        [(location of access point 0), (location of access point 1), · · · ]
        · · ·`
    )
    Net1
    (
        [(location of access point 0), (location of access point 1), · · · ]
        [(location of access point 0), (location of access point 1), · · · ]
        · · ·
    )
    · · · · · ·
```

</div>
</details>

You can found more tests [here](https://drive.google.com/drive/folders/1bon65UEAx8cjSvVhYJ-lgC8QMDX0fvUm)

| Circuit      | # Nets | Field size | Max net rectangle |
|--------------|--------|------------|-------------------|
| ariane133_51 | 128K | 844 x 1144 | 533 x 595 |
| ariane133_68 | 127K | 716 x 971  | 439 x 545 |
| mempool_tile | 136K | 475 x 644  | 330 x 322 |
| nvdla        | 176K | 1240 x 1682 | 572 x 772 |
| bsg_chip | 768K | 1532 x 2077 | 1087 x 905 |
| mempool_group | 3M | 1782 x 2417 | 1294 x 1695 |
| cluster | 10M | 3511 x 4764 | 3508 x 4762 |
| mempool_cluster_large | 59M | 7891 x 10708 | 7888 x 10706 |

* **Note**: all tests contain 10 metal layers (Nangate45)

## Evaluation

You can do evaluation (calculate contests metrics) with `evaluate_solution.py` script. It uses precompiled exe files for windows and ubuntu in `evaluator` folder. 
You can compile evaluator for your platform using `evaluator/compile.sh` 

Usage:
```bash
python evaluate_solution.py -net mempool_tile.net -cap mempool_tile.cap -output out.txt
```

## Results

### Different algorithms on mempool_tile

| Double stage algo | STP Solver | CPUs       | Time (sec) | Cost Total |
|-------------------|------------|------------|------------|------------|
| -                 | networkx   | 1          | 9872       | 13535553   |
| -                 | networkx   | 8          | 1224       | 13535553   |
| + (min: 30)       | networkx   | 16         | 805        | 14020965   |
| -                 | c_stp      | 16         | 459        | 12309192   |
| - gap: 1          | c_stp      | 16         | 426        | 12316057   |
| - gap: 10         | c_stp      | 16         | 692        | 12320175   |
| + (min: 10)       | c_stp      | 16         | 165        | 12645923   |
| + (min: 30)       | c_stp      | 16         | 184        | 12501200   |
| + (min: 50)       | c_stp      | 16         | 170        | 12412052   |
| -                 | scip_jack  | 16         | 2414       | 12112467   |
| + (min: 30)       | scip_jack  | 16         | 497        | 12291628   |

### Different algorithms on bsg_chip

| Double stage | STP Solver | CPUs | Time (sec) | Wirelength Cost | VIA Cost   | Overflow cost | Cost Total  |
|--------------|------------|------|------------|-----------------|------------|---------------|-------------|
| -            | c_stp      | 16   | 4766       | 59392972        | 20923252   | 818572        | 81134797    |
| + (min: 30)  | c_stp      | 16   | 1044       | 62909317        | 62909317   | 783531        | 85060008    |
| -            | scip_jack  | 16   | 30128      | 58793974        | 20375872   | 742907        | 79912753    |
| + (min: 30)  | scip_jack  | 16   | 2800       | 63391486        | 19580632   | 811815        | 83783934    |

### Different algorithms on mempool_group

| Double stage                     | STP Solver | CPUs | Time (sec) | Wirelength Cost | VIA Cost   | Overflow cost | Cost Total |
|----------------------------------|------------|------|------------|-----------------|------------|---------------|------------|
| + (min: 30)                      | c_stp      | 16   | 4666       | 280626677       | 82536076   | 2495859       | 365658612  |
| - (sort: area, no lock)          | c_stp      | 16   | 8049       | 275374232       | 84538176   | 1992585       | 361904993  |
| - (sort: random, with lock)      | c_stp      | 16   | 12033      | 275055598       | 84712932   | 1579240       | 361347770  |
| + (min: 10, sort: area, no lock) | c_stp      | 16   | 4419       | 287358236       | 82166864   | 4220232       | 373745332  |

### Different algorithms on cluster

| Double stage | STP Solver | CPUs | Time (sec) | Wirelength Cost | VIA Cost   | Overflow cost | Cost Total |
|--------------|------------|------|------------|-----------------|------------|---------------|------------|
| + (min: 30)  | c_stp      | 16   | 21796      | 1142989638      | 280951996  | 41467736      | 1465409370 |


* **Note 1**: Tests were made on AMD Ryzen Threadripper PRO 5955WX 16-Cores
* **Note 2**: Different algorithms used in NetworkX and in C-code it explains difference in cost (C-code algorithm is more efficient).

## Citation

For more details, please refer to the publication: [https://doi.org/10.1109/ACCESS.2025.3526722](https://doi.org/10.1109/ACCESS.2025.3526722)

If you find this code useful, please cite it as:
```
@article{solovyev2025pagr,
  title={PAGR: Accelerating Global Routing for VLSI Design Flow},
  author={Solovyev, Roman A and Mkrtchan, Ilya A and Telpukhov, Dmitry V and Shafeev, Ilya I and Romanov, Aleksandr Y and Stolbikov, Yevgeniy V and Stempkovsky, Alexander L},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```
