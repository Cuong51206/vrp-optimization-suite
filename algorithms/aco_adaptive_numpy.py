import numpy as np
import time
import random
from dataclasses import dataclass
from typing import List
from envs.deliveryNetwork import DeliveryNetwork
from configs.vrpconfig import Config

@dataclass
class Solution:
    routes: List[List[int]]
    cost: float

# Hàm tính điểm chuẩn hóa để so sánh công bằng
def calculate_simple_cost(routes, env, distance_matrix):
    total_cost = 0
    for v in range(len(routes)):
        if len(routes[v]) > 2: 
            total_cost += env.get_vehicles()[v]['cost']
            for i in range(len(routes[v]) - 1):
                total_cost += distance_matrix[routes[v][i]][routes[v][i+1]]
    return total_cost

class AdaptiveNumpyGraph:
    def __init__(self, env: DeliveryNetwork, config: Config):
        self.cfg = config
        self.env = env
        self.n_nodes = self.env.n_deliveries + 1 # Bao gồm 100 khách + 1 kho (ID 0)
        
        self.demand_map = {self.cfg.DEPOT_ID: 0}
        for i in self.env.get_delivery().keys():
            self.demand_map[i] = self.env.get_delivery()[i]['vol']

        # ---------------------------------------------------------
        # KHỞI TẠO ĐỘNG CƠ NUMPY (Lấy cảm hứng từ Akavall)
        # ---------------------------------------------------------
        # 1. Ma trận khoảng cách (Lấy trực tiếp từ môi trường)
        self.distance_matrix = self.env.distance_matrix
        
        # 2. Ma trận Nghịch đảo khoảng cách (Eta) - Tính 1 lần dùng mãi mãi
        # Cộng thêm 1e-10 để tránh lỗi chia cho 0.
        self.eta_matrix = 1.0 / (self.distance_matrix + 1e-10)
        np.fill_diagonal(self.eta_matrix, 0) # Không tự đi đến chính mình
        
        # 3. Ma trận Mùi (Tau) - Khởi tạo toàn bộ bằng 1.0
        self.tau_matrix = np.ones((self.n_nodes, self.n_nodes))
        np.fill_diagonal(self.tau_matrix, 0)

    # Cập nhật mùi bằng ma trận (Cực nhanh)
    def update_pheromone_matrix(self, best_solution, dynamic_rho):
        # 1. Bay hơi toàn bộ bản đồ cùng lúc
        self.tau_matrix = np.maximum((1 - dynamic_rho) * self.tau_matrix, 1e-10)

        # 2. Rải mùi cho lộ trình tốt nhất
        pheromone_increase = 1.0 / best_solution.cost
        for route in best_solution.routes:
            for i in range(len(route) - 1):
                n1, n2 = route[i], route[i + 1]
                self.tau_matrix[n1, n2] += (dynamic_rho * pheromone_increase)
                self.tau_matrix[n2, n1] += (dynamic_rho * pheromone_increase)

class AdaptiveNumpyAnt:
    def __init__(self, graph: AdaptiveNumpyGraph, config: Config):
        self.graph = graph
        self.cfg = config
        self.env = graph.env
        self.reset_state()

    # Tạo Mặt nạ (Mask) VRP: 1 là đi được, 0 là cấm đi
    def get_available_mask(self, current_node):
        mask = np.zeros(self.graph.n_nodes)
        has_valid_node = False
        
        for node in self.nodes_left:
            dist = self.graph.distance_matrix[current_node][node]
            info = self.env.delivery_info.get(node)
            
            # Check thời gian và tải trọng
            if (self.tour_time[self.vehicle] + dist + info['crowd_cost']) <= info['time_window_max']:
                if self.capacity[self.vehicle] >= self.graph.demand_map[node]:
                    mask[node] = 1
                    has_valid_node = True
                    
        return mask, has_valid_node

    # ---------------------------------------------------------
    # TRÁI TIM CỦA THUẬT TOÁN: Chọn đường bằng Ma trận + Thích nghi
    # ---------------------------------------------------------
    def select_next_delivery_numpy(self, current_node, dynamic_alpha, dynamic_beta):
        mask, has_valid_node = self.get_available_mask(current_node)
        
        if not has_valid_node:
            return None
            
        # Lấy hàng ngang dữ liệu của trạm hiện tại
        tau_row = self.graph.tau_matrix[current_node]
        eta_row = self.graph.eta_matrix[current_node]
        
        # TÍNH ĐIỂM CHO 100 KHÁCH CÙNG LÚC (Không dùng vòng lặp for)
        scores = (tau_row ** dynamic_alpha) * (eta_row ** dynamic_beta)
        
        # Lọc bỏ các khách vi phạm VRP bằng cách nhân với mặt nạ (scores * 0 = 0)
        valid_scores = scores * mask
        
        total_score = np.sum(valid_scores)
        if total_score == 0:
            return None
            
        # Luật Q0 (Khai thác vs Khám phá)
        q = random.random()
        if q <= self.cfg.Q0:
            # Chọn điểm cao nhất
            next_delivery = np.argmax(valid_scores)
        else:
            # Quay Roulette bằng mảng xác suất Numpy
            probabilities = valid_scores / total_score
            next_delivery = np.random.choice(self.graph.n_nodes, p=probabilities)
        
        return int(next_delivery)

    def move_to_delivery(self, current_node, next_delivery):
        self.routes[self.vehicle].append(next_delivery)
        if next_delivery != self.cfg.DEPOT_ID:
            self.nodes_left.remove(next_delivery)
            crowd_cost = self.env.delivery_info.get(next_delivery)['crowd_cost']
        else:
            crowd_cost = 0
            
        self.capacity[self.vehicle] -= self.graph.demand_map[next_delivery]
        self.tour_time[self.vehicle] += self.graph.distance_matrix[current_node][next_delivery] + crowd_cost

    def find_solution(self, dynamic_alpha, dynamic_beta):
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
                    
                # Gọi hàm chọn đường siêu tốc
                next_delivery = self.select_next_delivery_numpy(current_node, dynamic_alpha, dynamic_beta)
                
                if next_delivery is None:
                    if current_node != self.cfg.DEPOT_ID:
                        self.move_to_delivery(current_node, self.cfg.DEPOT_ID)
                        moved = True
                else:
                    self.move_to_delivery(current_node, next_delivery)
                    moved = True
            
            if not moved:
                break

        for vehicle in range(self.env.n_vehicles):
            if self.routes[vehicle][-1] != self.cfg.DEPOT_ID:
                self.vehicle = vehicle
                self.move_to_delivery(self.routes[vehicle][-1], self.cfg.DEPOT_ID)      
        
        cost = calculate_simple_cost(self.routes, self.env, self.graph.distance_matrix)
        return Solution(self.routes, cost)

    def reset_state(self):
        self.capacity = [self.env.get_vehicles()[i]['capacity'] for i in range(self.env.n_vehicles)]
        self.nodes_left = set(range(1, self.graph.n_nodes)) # Lấy ID từ 1 đến 100
        self.routes = []
        self.tour_time = [0 for _ in range(self.env.n_vehicles)]

def run_adaptive_numpy(cfg: Config, verbose: bool = True):
    env = DeliveryNetwork(cfg)
    graph = AdaptiveNumpyGraph(env, cfg) 
    ants = [AdaptiveNumpyAnt(graph, cfg) for _ in range(cfg.NUM_ANTS)]

    best_solution = None

    print("Bắt đầu chạy Thuật toán: Adaptive ACO + Numpy Engine...")
    
    # Hộp số tự động: Trượt thông số từ Khám phá sang Khai thác
    alpha_start, alpha_end = 0.5, 2.0  
    beta_start, beta_end = 2.0, 0.5    
    rho_start, rho_end = 0.2, 0.05     

    for i in range(1, cfg.NUM_ITERATIONS + 1):
        progress = (i - 1) / max(1, cfg.NUM_ITERATIONS - 1) 
        
        current_alpha = alpha_start + progress * (alpha_end - alpha_start)
        current_beta = beta_start + progress * (beta_end - beta_start)
        current_rho = rho_start + progress * (rho_end - rho_start)
        
        for ant in ants:
            ant.reset_state()
            
        solutions = []
        for ant in ants:
            an_sol = ant.find_solution(current_alpha, current_beta)
            solutions.append(an_sol)

        candidate_best_solution = min(solutions, key=lambda s: s.cost)

        if not best_solution or candidate_best_solution.cost < best_solution.cost:
            best_solution = candidate_best_solution

        if verbose:
            print("Best Solution in Iteration {}/{} = {:.2f} (a={:.2f}, b={:.2f}, r={:.2f})".format(
                i, cfg.NUM_ITERATIONS, best_solution.cost, current_alpha, current_beta, current_rho))
            
        # Cập nhật bằng ma trận
        graph.update_pheromone_matrix(best_solution, current_rho)

    if verbose:
        print("\n--- KẾT QUẢ CUỐI CÙNG ---")
        print("Chi phí: {:.2f}".format(best_solution.cost))
        print("Lộ trình: \n", best_solution.routes)

if __name__ == "__main__":
    config = Config()
    start_time = time.time()
    
    run_adaptive_numpy(config, verbose=True)
    
    # Kết thúc bấm giờ
    end_time = time.time()
    print(f"\nThời gian chạy (Numpy Adaptive): {end_time - start_time:.4f} giây")