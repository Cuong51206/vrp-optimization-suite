import json
import numpy as np
import pandas as pd
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from envs.deliveryNetwork import DeliveryNetwork
from configs.vrpconfig import Config

class Graph:
    def __init__(self, env: DeliveryNetwork, config: Config):
        self.cfg = config
        self.env = env
        self.adjacency_map = self.create_adjacency_map() # ma trận khoảng cách
        self.pheromone_map = self.create_pheromone_map() # ma trận mùi hương
        self.demand_map = {} # lưu khối lượng hàng hóa
        self.demand_map[self.cfg.DEPOT_ID] = 0
        for i in self.env.get_delivery().keys():
            self.demand_map[i] = self.env.get_delivery()[i]['vol'] # gía trị hàng hóa của khách hàng i
            
    def create_adjacency_map(self) -> Dict[int, Dict[int, float]]:
        adjacency_map = {} #tạo dict lồng nhau để dễ truy xuất, ví dụ từ 5->8 là dicr[5][8]
        nodes = list(sorted(self.env.delivery_info.keys())) # sắp xếp vị trí các nodes tăng dần
        nodes.insert(0,0)
        
        for node in nodes:
            adjacency_map[node] = {}
            adjacency_map[node][node] = 0.0 # khoảng cách từ chính nó = 0
            
        for index_1, node_1 in enumerate(nodes): #duyệt từng node
            for node_2 in nodes[index_1+1:]: # chỉ xét các node phía sau nó
                adjacency_map[node_1][node_2] = self.env.distance_matrix[node_1][node_2] # trích xuất khoảng cách và lưu 2 chiều
                adjacency_map[node_2][node_1] = self.env.distance_matrix[node_2][node_1] # duyệt theo tịnh tiến để không phải quay lại
        return adjacency_map

    def create_pheromone_map(self) -> Dict[int, Dict[int, float]]:
        pheromone_map = {}
        nodes = list(sorted(self.env.delivery_info.keys()))
        nodes.insert(0,0)
        
        # Khởi tạo pheromone cơ sở cho cả bản thân nó
        for node in nodes:
            pheromone_map[node] = {}
            pheromone_map[node][node] = 1.0
            
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1+1:]:
                pheromone_init = 1 # Điểm hấp dẫn = Mùi * Độ gần
                pheromone_map[node_1][node_2] = pheromone_init
                pheromone_map[node_2][node_1] = pheromone_init
        return pheromone_map

    def update_pheromone_map(self, best_solution): # hàm update mùi | 4.2 công thức 4
        nodes = list(sorted(self.pheromone_map.keys()))
        for index_1, node_1 in enumerate(nodes):
            for node_2 in nodes[index_1 + 1:]:
                new_value = max(round((1 - self.cfg.RHO) * self.pheromone_map[node_1][node_2], 5), 1e-10) # update mùi mới sau khi bay hơi, lấy max để đẩm bảo mùi != 0
                self.pheromone_map[node_1][node_2] = new_value
                self.pheromone_map[node_2][node_1] = new_value

        pheromone_increase = 1.0 / best_solution.cost # chi phí (phân số) càng nhỏ thì thưởng mùi càng lớn
        for route in best_solution.routes:
            edges = [(route[index], route[index + 1]) for index in range(0, len(route) - 1)] # [0, 5, 8, 0] -> [(0, 5), (5, 8), (8, 0)]
            for edge in edges:
                self.pheromone_map[edge[0]][edge[1]] += (self.cfg.RHO * pheromone_increase) # thưởng mùi mới
                self.pheromone_map[edge[1]][edge[0]] += (self.cfg.RHO * pheromone_increase)

class Ant:
    def __init__(self, graph: Graph, config: Config):
        self.graph = graph
        self.cfg = config
        self.env = graph.env
        self.time_conv_to_cost = self.cfg.conv_time_to_cost
        self.capacity = [self.env.get_vehicles()[i]['capacity'] for i in range(0, self.env.n_vehicles)] # Lấy sức chứa của từng xe cho vào 1 list
        self.tour_time = [0 for i in range(0, self.env.n_vehicles)] # Reset lại đồng hồ thời gian
        self.vehicle = 0 # Bát dầu bằng xe 0
        self.reset_state()

    def get_available_nodes(self, current_node): # dùng để xem từ current_node có đi đến điểm tiếp theo được không | mục 2 trang 42
        allowed_by_time_max = [node for node in self.nodes_left if (self.tour_time[self.vehicle]+self.graph.adjacency_map[current_node][node]+self.env.delivery_info.get(node)['crowd_cost']) <= self.env.delivery_info.get(node)['time_window_max']]
        allowed_by_capacity_time_max_min = [node for node in allowed_by_time_max if self.capacity[self.vehicle] >= self.graph.demand_map[node]]
        return allowed_by_capacity_time_max_min # mảng chứa các khách hàng có thể đến kịp lúc và có thể chứa hàng

    def select_first_node(self):
        available_nodes = self.get_available_nodes(self.cfg.DEPOT_ID)
        if not available_nodes:
            return None
        return np.random.choice(available_nodes) # Khách đầu random để tăng tính khám phá

    def select_next_delivery(self, current_node): # mục 4.1
        available_nodes = self.get_available_nodes(current_node)
        if not available_nodes:
            return None
                        
        scores = [] # Tính điểm
        for node in available_nodes:
            tau = pow(self.graph.pheromone_map[current_node][node], self.cfg.ALPHA) # nồng độ mùi
            eta = pow(1 / self.graph.adjacency_map[current_node][node], self.cfg.BETA) # càng gần càng cao
            scores.append(tau * eta) 

        q = random.random() # random 1 số bất kì
        
        if q <= self.cfg.Q0: # nếu q < Q0 (0.9)
            max_index = scores.index(max(scores)) # lấy index chỉ số điểm có score cao nhất làm điểm tiếp theo
            next_delivery = available_nodes[max_index] 
        else:
            denominator = sum(scores)
            probabilities = [score / denominator for score in scores] # tính tỷ lệ phần trăm của score đó
            next_delivery = np.random.choice(available_nodes, p=probabilities) # quay ngẫu nhiên dựa trên trọng số (list)
        
        return next_delivery

    def move_to_delivery(self, current_node, next_delivery): # cập nhật trạng thái khi tài xế đến nơi
        self.routes[self.vehicle].append(next_delivery) # lưu vào danh sách lịch trình
        if next_delivery != self.cfg.DEPOT_ID:
            self.nodes_left.remove(next_delivery) # khi đến nơi thì xóa khách hàng khỏi danh sách
            crowd_cost = self.env.delivery_info.get(next_delivery)['crowd_cost'] # cập nhật chi phí ngoài
        else:
            crowd_cost = 0
        # Sau khi giao xong 1 khách thì cập nhật những thông số sau
        self.capacity[self.vehicle] -= self.graph.demand_map[next_delivery] # khi giao thành công thì giảm bớt trọng tải xe
        self.tour_time[self.vehicle] += self.graph.adjacency_map[current_node][next_delivery]+crowd_cost # khoảng cách chạy xe trên đường + thời gian kẹt xe
        self.total_path_cost[self.vehicle] += (self.graph.adjacency_map[current_node][next_delivery]) # biến này dùng để tính tiền xăng

    def find_solution(self):
        for vehicle in range(0, self.env.n_vehicles):
            self.vehicle = vehicle
            self.routes.append([self.cfg.DEPOT_ID]) #cho điểm bắt đầu vào mảng đầu tiên
            current_node = self.routes[self.vehicle][-1] # vị trí hiện tại là vị trí cuối cùng trong route
            available_nodes = self.get_available_nodes(current_node) #Tìm những khách hàng có thể giao
            if available_nodes:
                first_delivery = np.random.choice(available_nodes) # chọn ngẫu nhiên khách hàng đầu tiên
                self.move_to_delivery(self.cfg.DEPOT_ID, first_delivery) # tự động trừ trọng lượng trên xe, gạch tên ra khỏi danh sách và + thêm thời gian
                self.total_path_cost[self.vehicle] += self.env.get_vehicles()[self.vehicle]['cost'] # cộng phí xuất bãi
                
        while self.nodes_left:
            moved_this_turn = False
            for vehicle in range(0, self.env.n_vehicles):
                self.vehicle = vehicle
                current_node = self.routes[self.vehicle][-1]
                
                if len(self.routes[self.vehicle])>2 and current_node==self.cfg.DEPOT_ID: # Xe đã giao xong và đang ở bãi xuất phát nên không cần đi tiếp
                    continue # chuyển sang xe tiếp theo
                else:
                    next_delivery = self.select_next_delivery(current_node) # chọn khách hàng giao tiếp theo
                    if (next_delivery is None): # nếu xe không thể đi giao tiếp
                        # Vá lỗi infinite loop (Chỉ đi về kho nếu xe đang không ở kho)
                        if current_node != self.cfg.DEPOT_ID:
                            self.move_to_delivery(current_node, self.cfg.DEPOT_ID) # đi về kho
                            moved_this_turn = True # xe này có di chuyển
                    else:
                        self.move_to_delivery(current_node, next_delivery)
                        moved_this_turn = True
            
            # Ngắt vòng lặp nếu tất cả xe đều đầy tải và về kho, tránh treo máy
            if not moved_this_turn:
                break

        for vehicle in range(0, self.env.n_vehicles): # duyệt lại toàn bộ xe cho về kho
            self.vehicle = vehicle
            if self.routes[self.vehicle][-1] != self.cfg.DEPOT_ID:
                self.move_to_delivery(self.routes[self.vehicle][-1], self.cfg.DEPOT_ID)      
        return Solution(self.routes, sum(self.total_path_cost)) # tổng hợp lại để chấm điểm và so sánh

    def reset_state(self):
        self.capacity = [self.env.get_vehicles()[i]['capacity'] for i in range(0, self.env.n_vehicles)] # phục hồi lại trọng tải từng xe
        self.nodes_left = set(self.graph.adjacency_map.keys()) # reset lại danh sách khách hàng
        self.nodes_left.remove(self.cfg.DEPOT_ID)
        self.routes = [] #reset lại route
        self.tour_time = [0 for i in range(0, self.env.n_vehicles)] # reset lại đồng hồ
        self.total_path_cost = [0 for i in range(0, self.env.n_vehicles)]

@dataclass
class Solution:
    routes: List[int]
    cost: float

def run(cfg: Config, verbose: bool = True) -> Tuple[Solution, List[Solution]]:
    env = DeliveryNetwork(cfg)
    graph = Graph(env, cfg) 
    ants = [Ant(graph, cfg) for i in range(0, cfg.NUM_ANTS)] # đẻ ra 100 con sử dụng chung 1 bản đồ Graph

    best_solution = None
    all_solutions = []

    print("Bắt đầu chạy Thuật toán ACO...")
    for i in range(1, cfg.NUM_ITERATIONS + 1):
        for ant in ants:
            ant.reset_state()
            
        solutions = []
        for ant in ants:
            an_sol = ant.find_solution()
            solutions.append(an_sol)

        candidate_best_solution = min(solutions, key=lambda solution: solution.cost) # key=lambda solution: solution.cost trả về cost của từng solution

        if not best_solution or candidate_best_solution.cost < best_solution.cost:
            best_solution = candidate_best_solution

        if verbose:
            print("Best Solution in Iteration {}/{} = {:.2f}".format(i, cfg.NUM_ITERATIONS, best_solution.cost))
            
        all_solutions.append(best_solution.cost)
        graph.update_pheromone_map(best_solution)

    if verbose:
        print("\n--- FINAL RESULT ---")
        print("Best Solution Cost: {:.2f}".format(best_solution.cost))
        print("Best Solution Routes: \n", best_solution.routes)
        
    return all_solutions, best_solution

if __name__ == "__main__":
    config = Config()
    start_time = time.time()
    
    run(config, verbose=True)
    
    # Kết thúc bấm giờ
    end_time = time.time()
    print(f"\nThời gian chạy (ACO cơ bản): {end_time - start_time:.4f} giây")