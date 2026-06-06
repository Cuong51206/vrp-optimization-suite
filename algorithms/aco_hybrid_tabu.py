import numpy as np
import random
import time
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple
from envs.deliveryNetwork import DeliveryNetwork
from configs.vrpconfig import Config

@dataclass
class Solution:
    routes: List[List[int]]
    cost: float

class HybridGraph:
    def __init__(self, env: DeliveryNetwork, config: Config):
        self.cfg = config
        self.env = env
        self.adjacency_map = self.create_adjacency_map()
        self.demand_map = {self.cfg.DEPOT_ID: 0}
        for i in self.env.get_delivery().keys():
            self.demand_map[i] = self.env.get_delivery()[i]['vol']
            
        # Tham số riêng cho ACS
        self.tau_0 = 1.0 # Pheromone cơ sở ban đầu
        self.rho_local = 0.1 # Tốc độ bay hơi cục bộ của ACS
        self.pheromone_map = self.create_pheromone_map()

    def create_adjacency_map(self) -> Dict[int, Dict[int, float]]:
        adjacency_map = {}
        nodes = list(sorted(self.env.delivery_info.keys()))
        nodes.insert(0,0)
        
        for node in nodes:
            adjacency_map[node] = {node: 0.0}
            
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1+1:]:
                adjacency_map[node_1][node_2] = self.env.distance_matrix[node_1][node_2]
                adjacency_map[node_2][node_1] = self.env.distance_matrix[node_2][node_1]
        return adjacency_map

    def create_pheromone_map(self) -> Dict[int, Dict[int, float]]:
        pheromone_map = {}
        nodes = list(self.adjacency_map.keys())
        for node in nodes:
            pheromone_map[node] = {node: self.tau_0}
            
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1+1:]:
                pheromone_map[node_1][node_2] = self.tau_0
                pheromone_map[node_2][node_1] = self.tau_0
        return pheromone_map

    # CƠ CHẾ CỦA ACS: Cập nhật mùi cục bộ (Làm mờ vết mùi ngay khi đi qua)
    def local_update_pheromone(self, node_1, node_2):
        current_tau = self.pheromone_map[node_1][node_2]
        new_tau = (1 - self.rho_local) * current_tau + self.rho_local * self.tau_0
        self.pheromone_map[node_1][node_2] = new_tau
        self.pheromone_map[node_2][node_1] = new_tau

    # CƠ CHẾ CỦA ACS: Cập nhật mùi toàn cục CHỈ cho con kiến vô địch
    def global_update_pheromone(self, best_solution):
        pheromone_increase = 1.0 / best_solution.cost
        
        for route in best_solution.routes:
            for i in range(len(route) - 1):
                n1, n2 = route[i], route[i+1]
                current_tau = self.pheromone_map[n1][n2]
                new_tau = (1 - self.cfg.RHO) * current_tau + self.cfg.RHO * pheromone_increase
                self.pheromone_map[n1][n2] = new_tau
                self.pheromone_map[n2][n1] = new_tau

# HÀM TÍNH ĐIỂM (COST) ĐƠN GIẢN - GIỐNG NGƯỜI 1
def calculate_simple_cost(routes, env, graph):
    total_cost = 0
    for v in range(len(routes)):
        if len(routes[v]) > 2: 
            total_cost += env.get_vehicles()[v]['cost']
            for i in range(len(routes[v]) - 1):
                total_cost += graph.adjacency_map[routes[v][i]][routes[v][i+1]]
    return total_cost

class ACSAnt:
    def __init__(self, graph: HybridGraph, config: Config):
        self.graph = graph
        self.cfg = config
        self.env = graph.env
        self.reset_state()

    def get_available_nodes(self, current_node):
        allowed = [node for node in self.nodes_left if (self.tour_time[self.vehicle] + self.graph.adjacency_map[current_node][node] + self.env.delivery_info.get(node)['crowd_cost']) <= self.env.delivery_info.get(node)['time_window_max']]
        return [node for node in allowed if self.capacity[self.vehicle] >= self.graph.demand_map[node]]

    def select_next_delivery(self, current_node):
        available_nodes = self.get_available_nodes(current_node)
        if not available_nodes:
            return None
                        
        scores = []
        for node in available_nodes:
            tau = self.graph.pheromone_map[current_node][node]
            eta = 1.0 / (self.graph.adjacency_map[current_node][node] + 1e-10)
            scores.append((pow(tau, self.cfg.ALPHA) * pow(eta, self.cfg.BETA), node))

        # LUẬT CHỌN ĐƯỜNG CỦA ACS (Pseudorandom proportional rule)
        q = random.random()
        if q <= self.cfg.Q0:
            next_delivery = max(scores, key=lambda item: item[0])[1]
        else:
            total_score = sum(item[0] for item in scores)
            probs = [item[0] / total_score for item in scores]
            next_delivery = np.random.choice([item[1] for item in scores], p=probs)
        
        return next_delivery

    def move_to_delivery(self, current_node, next_delivery):
        self.routes[self.vehicle].append(next_delivery)
        if next_delivery != self.cfg.DEPOT_ID:
            self.nodes_left.remove(next_delivery)
            crowd_cost = self.env.delivery_info.get(next_delivery)['crowd_cost']
        else:
            crowd_cost = 0
            
        self.capacity[self.vehicle] -= self.graph.demand_map[next_delivery]
        self.tour_time[self.vehicle] += self.graph.adjacency_map[current_node][next_delivery] + crowd_cost
        
        self.graph.local_update_pheromone(current_node, next_delivery)

    def find_solution(self):
        for vehicle in range(self.env.n_vehicles):
            self.vehicle = vehicle
            self.routes.append([self.cfg.DEPOT_ID])
            
        while self.nodes_left:
            moved = False
            for vehicle in range(self.env.n_vehicles):
                self.vehicle = vehicle
                current_node = self.routes[self.vehicle][-1]
                
                if len(self.routes[self.vehicle]) > 2 and current_node == self.cfg.DEPOT_ID:
                    continue
                    
                next_delivery = self.select_next_delivery(current_node)
                if next_delivery is None:
                    if current_node != self.cfg.DEPOT_ID:
                        self.move_to_delivery(current_node, self.cfg.DEPOT_ID)
                        moved = True
                else:
                    self.move_to_delivery(current_node, next_delivery)
                    moved = True
            
            if not moved: break

        for vehicle in range(self.env.n_vehicles):
            if self.routes[vehicle][-1] != self.cfg.DEPOT_ID:
                self.vehicle = vehicle
                self.move_to_delivery(self.routes[vehicle][-1], self.cfg.DEPOT_ID)      
        
        # SỬ DỤNG HÀM TÍNH COST MỚI ĐỂ CÔNG BẰNG
        cost = calculate_simple_cost(self.routes, self.env, self.graph)
        return Solution(self.routes, cost)

    def reset_state(self):
        self.capacity = [self.env.get_vehicles()[i]['capacity'] for i in range(self.env.n_vehicles)]
        self.nodes_left = set(self.graph.adjacency_map.keys())
        self.nodes_left.remove(self.cfg.DEPOT_ID)
        self.routes = []
        self.tour_time = [0 for _ in range(self.env.n_vehicles)]

# TABU SEARCH
def tabu_search_optimize(solution: Solution, env: DeliveryNetwork, graph: HybridGraph, max_iter=20) -> Solution:
    best_sol = copy.deepcopy(solution)
    current_sol = copy.deepcopy(solution)
    
    tabu_list = []
    tabu_tenure = 5 
    
    for _ in range(max_iter):
        neighborhood = []
        routes = current_sol.routes
        
        for v1 in range(len(routes)):
            for v2 in range(v1 + 1, len(routes)):
                for i in range(1, len(routes[v1]) - 1):
                    for j in range(1, len(routes[v2]) - 1):
                        node1, node2 = routes[v1][i], routes[v2][j]
                        
                        cap_v1_new = sum(graph.demand_map[n] for n in routes[v1]) - graph.demand_map[node1] + graph.demand_map[node2]
                        cap_v2_new = sum(graph.demand_map[n] for n in routes[v2]) - graph.demand_map[node2] + graph.demand_map[node1]
                        
                        if cap_v1_new <= env.get_vehicles()[v1]['capacity'] and cap_v2_new <= env.get_vehicles()[v2]['capacity']:
                            new_routes = copy.deepcopy(routes)
                            new_routes[v1][i], new_routes[v2][j] = new_routes[v2][j], new_routes[v1][i]
                            
                            # SỬ DỤNG HÀM TÍNH COST MỚI ĐỂ CÔNG BẰNG
                            cost = calculate_simple_cost(new_routes, env, graph)
                            neighborhood.append({
                                'routes': new_routes,
                                'cost': cost,
                                'move': (node1, node2)
                            })

        if not neighborhood:
            break
            
        neighborhood.sort(key=lambda x: x['cost'])
        
        move_made = False
        for neighbor in neighborhood:
            move = neighbor['move']
            if move not in tabu_list or neighbor['cost'] < best_sol.cost:
                current_sol = Solution(neighbor['routes'], neighbor['cost'])
                
                tabu_list.append(move)
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)
                    
                if current_sol.cost < best_sol.cost:
                    best_sol = copy.deepcopy(current_sol)
                move_made = True
                break
                
        if not move_made:
            break
            
    return best_sol

def run_hybrid(cfg: Config, verbose: bool = True):
    env = DeliveryNetwork(cfg)
    graph = HybridGraph(env, cfg) 
    ants = [ACSAnt(graph, cfg) for _ in range(cfg.NUM_ANTS)]

    global_best_solution = None

    print("Bắt đầu chạy Thuật toán ACS + Tabu Search...")
    for i in range(1, cfg.NUM_ITERATIONS + 1):
        for ant in ants:
            ant.reset_state()
            
        solutions = []
        for ant in ants:
            solutions.append(ant.find_solution())

        iteration_best = min(solutions, key=lambda s: s.cost)
        
        refined_solution = tabu_search_optimize(iteration_best, env, graph)

        if not global_best_solution or refined_solution.cost < global_best_solution.cost:
            global_best_solution = copy.deepcopy(refined_solution)

        if verbose:
            print("Best Solution in Iteration {}/{} = {:.2f}".format(i, cfg.NUM_ITERATIONS, global_best_solution.cost))
            
        graph.global_update_pheromone(global_best_solution)

    if verbose:
        print("\n--- FINAL RESULT (ACS + TABU SEARCH) ---")
        print("Best Solution Cost: {:.2f}".format(global_best_solution.cost))
        print("Best Solution Routes: \n", global_best_solution.routes)

if __name__ == "__main__":
    config = Config()
    start_time = time.time()
    
    run_hybrid(config, verbose=True)
    
    # Kết thúc bấm giờ
    end_time = time.time()
    print(f"\nThời gian chạy (ACS + Tabu): {end_time - start_time:.4f} giây")