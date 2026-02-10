from Problem import Problem
import random
import networkx as nx


POPULATION_SIZE = 100
GENERATIONS = 1000
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.2
ELITISM_COUNT = 2

def solution(p: Problem):
    """
    Solves the Traveling Thief Problem variant using a Genetic Algorithm.
    Returns a path in the format [(city_id, gold_collected), ...].
    """
    
    # Pre-computation of shortest paths and beta costs
    alpha = p.alpha
    beta = p.beta
    all_nodes = list(p.graph.nodes)
    city_nodes = [n for n in all_nodes if n != 0]
    gold_map = nx.get_node_attributes(p.graph, 'gold')
    
    print("Pre-computing path costs...")
    
    dist_matrix = {}      # Geometric distance
    beta_dist_matrix = {} # Sum of (edge_dist ** beta)
    
    for src in all_nodes:
        # Run Dijkstra once per source to get the "Shortest Path Tree"
        # preds: a dict where preds[node] = [parent_nodes...]
        # dists: a dict where dists[node] = total_geometric_distance
        preds, dists = nx.dijkstra_predecessor_and_distance(p.graph, src, weight='dist')
        
        # Sort nodes by distance. 
        # This ensures a parent is always processed before child, allowing to build costs incrementally
        ordered_nodes = sorted(dists.keys(), key=lambda k: dists[k])
        
        beta_vals = {src: 0.0}
        
        for city in ordered_nodes:
            if city == src:
                continue
                
            parent = preds[city][0]
            
            edge_dist = dists[city] - dists[parent]
            
            # BetaCost(Source -> City) = BetaCost(Source -> Parent) + (Edge^Beta)
            beta_vals[city] = beta_vals[parent] + (edge_dist ** beta)
            
        dist_matrix[src] = dists
        beta_dist_matrix[src] = beta_vals

    print("Pre-computation complete.")

    # Helper functions    
    def calculate_segment_cost(source, target, current_weight):
        """
        Calculates cost for a trip from source to target.
        Uses the pre-computed 'beta_dist' to efficiently compute the penalty term.
        The formula is: dist + (alpha * w)^beta * Sum(edge^beta)
        """
        if source == target:
            return 0.0
            
        d_total = dist_matrix[source][target]
        d_beta_sum = beta_dist_matrix[source][target]
        
        # Factored out constant weight: sum( (alpha * d * w)^beta ) becomes: (alpha * w)^beta * sum( d^beta )
        penalty = ((alpha * current_weight) ** beta) * d_beta_sum
        
        return d_total + penalty

    def evaluate(sequence):     #Evaluates total cost of a tour
        total_cost = 0.0
        current_node = 0
        current_weight = 0.0
        
        for next_node in sequence:
            gold_at_next = gold_map[next_node]
            
            # Option A: Go Direct
            cost_direct = calculate_segment_cost(current_node, next_node, current_weight)
            
            # Option B: Via Base
            # 1. Return to base with current weight
            cost_return = calculate_segment_cost(current_node, 0, current_weight)
            # 2. Leave base with 0 weight
            cost_outbound = calculate_segment_cost(0, next_node, 0)
            
            cost_via_base = cost_return + cost_outbound

            if cost_direct <= cost_via_base:
                total_cost = total_cost + cost_direct
                current_weight = current_weight + gold_at_next
                current_node = next_node
            else:
                total_cost = total_cost + cost_via_base
                current_weight = gold_at_next
                current_node = next_node
        
        # Final return to base
        total_cost = total_cost + calculate_segment_cost(current_node, 0, current_weight)
        return total_cost

    def reconstruct_path_format(sequence):
        formatted_path = []
        current_node = 0
        current_weight = 0.0
        
        for next_node in sequence:
            gold_at_next = gold_map[next_node]
            
            cost_direct = calculate_segment_cost(current_node, next_node, current_weight)
            cost_via_base = calculate_segment_cost(current_node, 0, current_weight) + calculate_segment_cost(0, next_node, 0)

            if cost_direct <= cost_via_base:
                current_weight = current_weight + gold_at_next
                formatted_path.append((next_node, gold_at_next))
                current_node = next_node
            else:
                formatted_path.append((0, 0))
                current_weight = gold_at_next
                formatted_path.append((next_node, gold_at_next))
                current_node = next_node
                
        formatted_path.append((0, 0))
        return formatted_path

    
    # GA loop
    population = []
    for _ in range(POPULATION_SIZE):
        ind = list(city_nodes)
        random.shuffle(ind)
        population.append(ind)
        
    best_solution = None
    best_cost = float('inf')

    print(f"Running Genetic Algorithm ({GENERATIONS} generations)...")
    
    for gen in range(GENERATIONS):
        # Evaluate Fitness
        scored_population = []
        for ind in population:
            cost = evaluate(ind)
            scored_population.append((cost, ind))
            if cost < best_cost:
                best_cost = cost
                best_solution = list(ind) # Make a copy
        
        # Sort by cost
        scored_population.sort(key=lambda x: x[0])
        
        # Selection (Elitism)
        # Keep the best individuals unchanged
        new_population = [x[1] for x in scored_population[:ELITISM_COUNT]]
        
        # Breeding
        while len(new_population) < POPULATION_SIZE:
            # Tournament Selection
            # Pick random candidates and take the best
            candidates_a = random.sample(scored_population, TOURNAMENT_SIZE)
            parent_a = min(candidates_a, key=lambda x: x[0])[1]
            
            candidates_b = random.sample(scored_population, TOURNAMENT_SIZE)
            parent_b = min(candidates_b, key=lambda x: x[0])[1]
            
            # Ordered Crossover (preserves the relative order of cities)
            start, end = sorted(random.sample(range(len(parent_a)), 2))
            child = [-1] * len(parent_a)
            child[start:end] = parent_a[start:end]
            
            pointer = 0
            for gene in parent_b:
                if gene not in child:
                    while child[pointer] != -1:
                        pointer += 1
                    child[pointer] = gene
            
            # Mutation (Swap)
            if random.random() < MUTATION_RATE:
                idx1, idx2 = random.sample(range(len(child)), 2)
                child[idx1], child[idx2] = child[idx2], child[idx1]
                
            new_population.append(child)
            
        population = new_population
        
        # Log progress every 100 generations
        if gen % 100 == 0:
            print(f"Gen {gen}: Best Cost = {best_cost:.4f}")


    
    # Calculate Baseline for comparison
    print("\nCalculating Baseline...")
    baseline_cost = p.baseline()
    
    print("-" * 40)
    print(f"RESULTS:")
    print(f"Baseline Cost:  {baseline_cost}")
    print(f"GA Solution:    {best_cost}")
    
    if best_cost < baseline_cost:
        print(f"IMPROVEMENT:    {baseline_cost - best_cost} ({(baseline_cost - best_cost)/baseline_cost*100:.2f}%)")
    else:
        print("No improvement found.")
    print("-" * 40)

    # Convert the best sequence into the required path format
    final_path = reconstruct_path_format(best_solution)
    
    return final_path


if __name__ == "__main__":
    print("===Test Case: 1000 cities, density 0.2, alpha 1, beta 1===")
    case=Problem(1000, density=0.2, alpha=1, beta=1)    
    best_path = solution(case)    