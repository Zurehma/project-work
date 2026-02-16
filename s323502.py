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

    max_node_id = max(all_nodes)
    
    print("Pre-computing path costs...")
    
    #using arrays instead of dictionaries to reduce computation time in large cases e.g 1000 cities
    dist_matrix = [[0.0] * (max_node_id + 1) for _ in range(max_node_id + 1)]       #2D array for geometric distance
    beta_dist_matrix = [[0.0] * (max_node_id + 1) for _ in range(max_node_id + 1)]  #sum of (edge_dist ** beta)
    parent_matrix = [[-1] * (max_node_id + 1) for _ in range(max_node_id + 1)]       #2D arrray storing parent node of each node for faster path reconstruction
    
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
            if city == src: continue
            
            parent = preds[city][0]
            dist_matrix[src][city] = dists[city]
            parent_matrix[src][city] = parent  
            
            # Beta Calculation
            edge_dist = dists[city] - dists[parent]
            beta_vals[city] = beta_vals[parent] + (edge_dist ** beta)
            beta_dist_matrix[src][city] = beta_vals[city]
        

    print("Pre-computation complete.")

    # Helper functions 
    def get_path(u, v):
        """Reconstructs path from u to v using parent matrix"""
        path = []
        curr = v
        while curr != u:
            path.append(curr)
            curr = parent_matrix[u][curr]
        path.append(u)
        return path[::-1] #Reverse to get correct order
       
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

    def evaluate(sequence):     #Evaluates total cost of a route
        total_cost = 0.0
        current_node = 0
        current_weight = 0.0
        
        for next_node in sequence:
            gold_at_next = gold_map[next_node]
            
            # Option A: Go Direct
            cost_direct = calculate_segment_cost(current_node, next_node, current_weight)
            
            # Option B: Via Base
            cost_return = calculate_segment_cost(current_node, 0, current_weight)
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
                shortestPath = get_path(current_node, next_node)
                for step in shortestPath[1:]:
                    if step==next_node:
                        current_weight = current_weight + gold_at_next
                        formatted_path.append((next_node, gold_at_next))
                    else:
                        formatted_path.append((step, 0))                
                current_node = next_node

            else:
                pathToBase = get_path(current_node, 0)
                for step in pathToBase[1:]:
                    formatted_path.append((step, 0))
                
                pathToNext = get_path(0, next_node)
                current_weight = 0
                
                for step in pathToNext[1:]:
                    if step==next_node:
                        current_weight = current_weight + gold_at_next
                        formatted_path.append((next_node, gold_at_next))
                    else:
                        formatted_path.append((step, 0))
                
                current_node = next_node
            
        #Final return to base
        pathToBase = get_path(current_node, 0)
        for step in pathToBase[1:]:
            formatted_path.append((step, 0))
                
        return formatted_path
    
    def is_valid(problem, path):
        if not path: return False

        print(f"Validating path...")
    
        valid_edges = set()     #using set decreases total computation time for 1000 cities (density 1) case from 45+ minutes to 15 minutes
                                #using nx.has_edge for each edge in path is too slow
        for u, v in problem.graph.edges:
            valid_edges.add((u, v))
            valid_edges.add((v, u)) 
            
        if (0, path[0][0]) not in valid_edges and (path[0][0], 0) not in valid_edges:
            return False

        for i in range(len(path) - 1):
            u = path[i][0]
            v = path[i+1][0]
            if (u, v) not in valid_edges:
                return False
                
        return True

    
    # GA loop
    # Initialize population as (cost, individual)
    population = []
    
    for _ in range(POPULATION_SIZE):
        ind = list(city_nodes)
        random.shuffle(ind)
        cost = evaluate(ind)
        population.append((cost, ind))
        
    best_cost = float('inf')
    best_solution = None

    print(f"Running Genetic Algorithm ({GENERATIONS} generations)...")
    
    for gen in range(GENERATIONS):
        # Sort by cost (ascending)
        population.sort(key=lambda x: x[0])
        
        current_best_cost, current_best_ind = population[0]
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = list(current_best_ind)
            
        # Elitism: Keep top 2 
        new_population = population[:ELITISM_COUNT]
        
        # Breeding
        while len(new_population) < POPULATION_SIZE:
            # Tournament Selection
            candidates_a = random.sample(population, TOURNAMENT_SIZE)
            parent_a = min(candidates_a, key=lambda x: x[0])[1]
            
            candidates_b = random.sample(population, TOURNAMENT_SIZE)
            parent_b = min(candidates_b, key=lambda x: x[0])[1]
            
            #Ordered crossover
            size = len(parent_a)
            start, end = sorted(random.sample(range(size), 2))
            child = [-1] * size
            child[start:end] = parent_a[start:end]
            
            genes_in_child = set(child[start:end])
            
            pointer = 0
            for gene in parent_b:
                if gene not in genes_in_child:
                    while child[pointer] != -1:
                        pointer += 1
                    child[pointer] = gene
            
            # Swap Mutation
            if random.random() < MUTATION_RATE:
                idx1, idx2 = random.sample(range(size), 2)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            
            child_cost = evaluate(child)
            new_population.append((child_cost, child))
            
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

    if is_valid(p, final_path):
        print("Path is valid")
    else:
        print("Path is invalid") 

    return final_path
