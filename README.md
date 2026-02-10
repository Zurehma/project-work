# Evolutionary Algorithm for the Traveling Thief Problem

This project implements a **Genetic Algorithm (GA)** to solve a variant of the **Traveling Thief Problem (TTP)**. The goal is to collect gold from $N$ cities and return it to a base (City 0) while minimizing the total cost.

The challenge lies in the cost function, which increases non-linearly with the weight of the gold carried. This forces the thief to balance collecting gold with returning to base to empty their bag.

## Problem Recap

We are given $N$ cities on a map with variable connectivity (controlled through the density parameter).
* **Base:** City 0 
* **Goal:** Visit all cities, collect all gold, and return to base.
* **Cost Function:**
  $$C(u, v) = d_{uv} + (\alpha \cdot d_{uv} \cdot w)^\beta$$
  Where:
  * $d_{uv}$: Geometric distance between cities $u$ and $v$.
  * $w$: Current weight (gold) carried.
  * $\alpha, \beta$: Parameters controlling the weight penalty.

Because $\beta$ is often $>1$, carrying a heavy load over long distances is expensive. The optimal strategy involves frequent returns to the base to drop off gold.



## Solution Methodology

The solution uses a Genetic Algorithm (GA) with a greedy heuristic for tour construction to handle the dual optimization challenge:
1.  **GA Component:** Evolves the sequence of cities to visit.
2.  **Tour Construction:** Dynamically determines optimal base return points based on current payload.

### 1. Evolutionary Search 
The algorithm evolves a population of candidate solutions to find the optimal visitation order.
* **Representation:** Individuals are represented as a **permutation of city indices** (excluding the base).
* **Selection:** Uses **tournament selection ($k=3$)** 
* **Crossover:** Implements **Ordered Crossover (OX)** to recombine parent traits without breaking the validity of the permutation.
* **Mutation:** Applies **Swap Mutation** (swapping two cities) to introduce random variations and prevent premature convergence.
* **Elitism:** The top 2 performing individuals are preserved in every generation.

### 2. Tour Construction Logic 

 The algorithm uses a **Greedy Split Heuristic** to split the sequence into a valid set of trips.

For each city in the sequence, the algorithm evaluates two scenarios:
 * **Option A:** Go directly to the next city (carrying current gold).
* **Option B:** Return to base to drop off gold, then go to the next city empty.

The option with the lower marginal cost is chosen at every step, creating an efficient balance between route directness and payload management.

### 3. Pre-computation
* To ensure scalability and handle dense graphs, the shortest paths between all pairs are pre-computed using **Dijkstra's Algorithm**.
* The geometric distance and the non-linear "Beta Distance" ($\sum d^\beta$) are pre-computed and stored in lookup matrices.


## Performance
The algorithm was tested with varying values of cities, graph connectivity (density), $\alpha$, and $\beta$.  

The algorithm consistently outperforms the baseline, especially in high-penalty scenarios ($\beta > 1$). As $\beta$ increases, the improvement gap widens significantly, reaching over 90% in the most extreme cases.

In cases where the weight penalty is linear ($\beta=1$), the baseline already produces a highly competitive solution. Consequently, the GA achieves only marginal improvements ($\approx 0.1\%$) in these scenarios.

### Key Results

#### Table 1 - Small Scale
| Cities ($N$) | Density | $\alpha$ | $\beta$ | Baseline Cost | GA Cost | Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100** | 0.2 | 1 | 1 | 25266.4056 | 25241.8112 | **0.10%** |
| **100** | 0.2 | 1 | 0.5 | 2231.0479 | 1614.7369 | **27.62%** |
| **100** | 0.2 | 1 | 2 | 5334401.9270 | 4926179.5876 | **7.65%** |
| **100** | 0.2 | 1 | 3 | 1504133861.1458 | 1046395851.4764 | **30.43%** |
| **100** | 0.2 | 1 | 5 | 187088165532885.75 | 51979452045712.1 | **72.22%** |
| **100** | 0.2 | 2 | 1 | 50425.3096 | 50398.2538 | **0.05%** |
| **100** | 1.0 | 1 | 1 | 18266.1857 | 18253.9539 | **0.07%** |
| **100** | 1.0 | 1 | 0.5 | 1292.6308 | 1077.6822 | **16.63%** |
| **100** | 1.0 | 1 | 2 | 5404978.0889 | 4231719.5062 | **21.71%** |
| **100** | 1.0 | 1 | 3 | 1957078935.0665 | 1041365721.3525 | **46.79%** |
| **100** | 1.0 | 1 | 5 | 347651302495115.3| 69767469033008.7 | **79.93%** |
| **100** | 1.0 | 2 | 1 | 36457.9184 | 36449.5266 | **0.02%** |  


#### Table 2 - Large Scale
| Cities ($N$) | Density | $\alpha$ | $\beta$ | Baseline Cost | GA Cost | Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1000** | 0.2 | 1 | 1 | 195402.9581 | 195378.7949 | **0.01%** |
| **1000** | 0.2 | 1 | 0.5 | 18717.2098 | 15897.7222 | **15.06%** |
| **1000** | 0.2 | 1 | 2 | 37545927.7021 | 26874647.6422 | **28.42%** |
| **1000** | 0.2 | 1 | 3 | 10600895130.6220 | 4322062922.5325 | **59.23%** |
| **1000** | 0.2 | 1 | 5 | 1575163721306340.5| 148146567008243.6 | **90.59%** |
| **1000** | 0.2 | 2 | 1 | 390028.7212 | 390003.9423 | **0.01%** |
| **1000** | 1.0 | 1 | 1 | 192936.2337 | 192878.9431 | **0.03%** |
| **1000** | 1.0 | 1 | 0.5 | 13456.9585 | 12031.8214 | **10.59%** |
| **1000** | 1.0 | 1 | 2 | 57580018.8687 | 47372332.6239 | **17.73%** |
| **1000** | 1.0 | 1 | 3 | 20943050223.626 | 12884117191.415 | **38.48%** |
| **1000** | 1.0 | 1 | 5 | 3798422113393744.0 | 1114911232894644.5 | **70.65%** |
| **1000** | 1.0 | 2 | 1 | 385105.6414 | 385056.41916 | **0.01%** |