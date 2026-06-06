import numpy as np
import random
import time
import torch
import copy
from dataclasses import dataclass
from typing import Dict, List
from envs.deliveryNetwork import DeliveryNetwork
from configs.vrpconfig import Config

@dataclass
class Solution:
    routes: List[List[int]]
    cost: float

def calculate_simple_cost(routes, env, dist_matrix):
    total_cost = 0
    for v in range(len(routes)):
        if len(routes[v]) > 2: 
            total_cost += env.get_vehicles()[v]['cost']
            for i in range(len(routes[v]) - 1):
                total_cost += dist_matrix[routes[v][i]][routes[v][i+1]]
    return total_cost

class TensorAntColony:
    def __init__(self, env: DeliveryNetwork, config: Config):
        self.cfg = config
        self.env = env
        self.n_nodes = self.env.n_deliveries + 1
        self.n_ants = self.cfg.NUM_ANTS
        
        # 1. KHỞI TẠO DEVICE (Ưu tiên dùng GPU CUDA nếu có, không thì dùng CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Tensor Engine đang chạy trên: {self.device.type.upper()}")

        # 2. CHUẨN BỊ DỮ LIỆU TĨNH LÊN TENSOR
        self.dist_matrix = self.env.distance_matrix
        eta_matrix = 1.0 / (self.dist_matrix + 1e-10)
        np.fill_diagonal(eta_matrix, 0)
        
        # Chuyển đổi sang PyTorch Tensors
        self.tau_tensor = torch.ones((self.n_nodes, self.n_nodes), device=self.device, dtype=torch.float32)
        self.eta_tensor = torch.tensor(eta_matrix, device=self.device, dtype=torch.float32)
        self.dist_tensor = torch.tensor(self.dist_matrix, device=self.device, dtype=torch.float32)
        
        # Trích xuất ràng buộc VRP
        demands = np.zeros(self.n_nodes)
        crowds = np.zeros(self.n_nodes)
        time_maxs = np.zeros(self.n_nodes)
        
        for i, info in self.env.delivery_info.items():
            demands[i] = info['vol']
            crowds[i] = info['crowd_cost']
            time_maxs[i] = info['time_window_max']
            
        time_maxs[0] = float('inf') # Kho không có giới hạn thời gian
        
        self.demands_t = torch.tensor(demands, device=self.device, dtype=torch.float32)
        self.crowds_t = torch.tensor(crowds, device=self.device, dtype=torch.float32)
        self.time_maxs_t = torch.tensor(time_maxs, device=self.device, dtype=torch.float32)
        
        self.vehicles_config = self.env.get_vehicles()

    def update_pheromone_tensor(self, best_solution):
        # Bay hơi toàn cục
        self.tau_tensor = torch.clamp((1 - self.cfg.RHO) * self.tau_tensor, min=1e-10)
        
        # Rải mùi cho lộ trình vô địch
        pheromone_increase = 1.0 / best_solution.cost
        for route in best_solution.routes:
            for i in range(len(route) - 1):
                n1, n2 = route[i], route[i + 1]
                self.tau_tensor[n1, n2] += (self.cfg.RHO * pheromone_increase)
                self.tau_tensor[n2, n1] += (self.cfg.RHO * pheromone_increase)

    # ----------------------------------------------------------------
    # TRÁI TIM CỦA TENSORACO: Chạy 100 con kiến song song trên GPU
    # ----------------------------------------------------------------
    def run_all_ants_batched(self):
        # Trạng thái của 100 con kiến được lưu bằng Python list để dễ xử lý logic VRP phức tạp
        routes = [[[0]] for _ in range(self.n_ants)]
        nodes_left = [set(range(1, self.n_nodes)) for _ in range(self.n_ants)]
        vehicles = [0 for _ in range(self.n_ants)]
        capacities = [self.vehicles_config[0]['capacity'] for _ in range(self.n_ants)]
        times = [0.0 for _ in range(self.n_ants)]
        active = [True for _ in range(self.n_ants)]

        while any(active):
            # 1. Đẩy trạng thái hiện tại của 100 con kiến lên GPU Tensors
            curr_nodes = [routes[i][vehicles[i]][-1] if active[i] else 0 for i in range(self.n_ants)]
            curr_t = torch.tensor(curr_nodes, device=self.device, dtype=torch.long)
            cap_t = torch.tensor(capacities, device=self.device, dtype=torch.float32).unsqueeze(1)
            time_t = torch.tensor(times, device=self.device, dtype=torch.float32).unsqueeze(1)

            # 2. Xây dựng Mask Ràng buộc VRP cho cả 100 con kiến CÙNG LÚC
            dists_from_curr = self.dist_tensor[curr_t] # Shape: (100, 101)
            
            valid_cap = cap_t >= self.demands_t.unsqueeze(0)
            valid_time = (time_t + dists_from_curr + self.crowds_t.unsqueeze(0)) <= self.time_maxs_t.unsqueeze(0)
            
            mask = valid_cap & valid_time
            mask[:, 0] = False # Không chọn lại kho qua hàm xác suất
            
            # Cập nhật mask cho các node đã đi
            mask_cpu = mask.cpu().numpy()
            for i in range(self.n_ants):
                if not active[i]:
                    mask_cpu[i, :] = False
                    continue
                # Tắt các trạm đã giao hàng
                for v in range(1, self.n_nodes):
                    if v not in nodes_left[i]:
                        mask_cpu[i, v] = False
                        
            mask_t = torch.tensor(mask_cpu, device=self.device, dtype=torch.float32)

            # 3. TÍNH XÁC SUẤT BẰNG MA TRẬN (Sức mạnh cốt lõi của TensorACO)
            tau_rows = self.tau_tensor[curr_t]
            eta_rows = self.eta_tensor[curr_t]
            
            # Tính điểm cho 100 kiến x 101 node chỉ trong 1 thao tác GPU
            scores = (tau_rows ** self.cfg.ALPHA) * (eta_rows ** self.cfg.BETA) * mask_t
            
            sum_scores = scores.sum(dim=1)
            has_moves = sum_scores > 0
            
            # 4. QUAY ROULETTE HÀNG LOẠT (Multinomial Sampling)
            safe_scores = scores + 1e-9 # Tránh lỗi cho các hàng có sum = 0
            # GPU chọn 100 điểm đến tiếp theo cùng một lúc
            next_nodes_t = torch.multinomial(safe_scores, 1).squeeze(1)
            next_nodes = next_nodes_t.cpu().numpy()
            scores_cpu = scores.cpu().numpy()

            # 5. Cập nhật lại trạng thái VRP (về CPU để nạp hàng, đổi xe)
            for i in range(self.n_ants):
                if not active[i]: continue

                curr_node = routes[i][vehicles[i]][-1]

                if has_moves[i]:
                    nxt = next_nodes[i]
                    
                    # Áp dụng luật Q0 (Khám phá / Khai thác)
                    if random.random() <= self.cfg.Q0:
                        nxt = np.argmax(scores_cpu[i])

                    routes[i][vehicles[i]].append(nxt)
                    nodes_left[i].remove(nxt)
                    capacities[i] -= self.env.delivery_info[nxt]['vol']
                    times[i] += self.dist_matrix[curr_node][nxt] + self.env.delivery_info[nxt]['crowd_cost']
                else:
                    # Logic xe VRP: Nếu hết đường, quay về kho
                    if curr_node != 0:
                        routes[i][vehicles[i]].append(0)
                        times[i] += self.dist_matrix[curr_node][0]
                    else:
                        # Đã ở kho, đổi sang xe tiếp theo
                        if vehicles[i] < self.env.n_vehicles - 1:
                            vehicles[i] += 1
                            routes[i].append([0])
                            capacities[i] = self.vehicles_config[vehicles[i]]['capacity']
                            times[i] = 0.0
                        else:
                            active[i] = False # Hết xe

                # Kiểm tra hoàn thành
                if not nodes_left[i]:
                    if routes[i][vehicles[i]][-1] != 0:
                        routes[i][vehicles[i]].append(0)
                    active[i] = False
                    
        # Trả về danh sách 100 Giải pháp
        return [Solution(routes[i], calculate_simple_cost(routes[i], self.env, self.dist_matrix)) for i in range(self.n_ants)]

def run_tensor_aco(cfg: Config, verbose: bool = True):
    env = DeliveryNetwork(cfg)
    colony = TensorAntColony(env, cfg) 
    
    best_solution = None
    print(f"Bắt đầu chạy Thuật toán: GPU-Accelerated TensorACO...")

    for i in range(1, cfg.NUM_ITERATIONS + 1):
        # 100 con kiến chạy đồng loạt trong hàm này
        solutions = colony.run_all_ants_batched()

        # Lọc ra con kiến tốt nhất
        candidate_best = min(solutions, key=lambda s: s.cost)

        if not best_solution or candidate_best.cost < best_solution.cost:
            best_solution = copy.deepcopy(candidate_best)

        if verbose:
            print("Best Solution in Iteration {}/{} = {:.2f}".format(i, cfg.NUM_ITERATIONS, best_solution.cost))
            
        # Cập nhật mùi bằng Tensor
        colony.update_pheromone_tensor(best_solution)

    if verbose:
        print("\n--- FINAL RESULT (TENSOR ACO) ---")
        print("Best Solution Cost: {:.2f}".format(best_solution.cost))
        print("Best Solution Routes: \n", best_solution.routes)

if __name__ == "__main__":
    config = Config()
    
    start_time = time.time()
    run_tensor_aco(config, verbose=True)
    end_time = time.time()
    
    print(f"\nThời gian chạy (TensorACO GPU/CPU Batched): {end_time - start_time:.4f} giây")