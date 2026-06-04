import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import List
from envs.deliveryNetwork import DeliveryNetwork
from vrpconfig import Config

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

# =====================================================================
# 1. MẠNG NƠ-RON CHÍNH (Thay thế hoàn toàn Pheromone & Q-Table)
# Kế thừa ý tưởng Policy Network từ OptMLGroup/VRP-RL
# =====================================================================
class VRPRoutingNetwork(nn.Module):
    def __init__(self, input_size, n_nodes):
        super(VRPRoutingNetwork, self).__init__()
        # Mạng nơ-ron 3 lớp (Bộ não của AI)
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_nodes) # Đầu ra là điểm số cho 101 trạm
        
    def forward(self, state, mask):
        # Truyền trạng thái hiện tại qua mạng nơ-ron
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x) # Logits: Điểm số thô của các điểm đến
        
        # KỸ THUẬT MASKING (Lấy từ VRP-RL): Ép điểm số của các đường cấm về -vô cùng
        # Điều này đảm bảo AI không bao giờ chọn khách hàng quá tải hoặc trễ giờ
        logits = logits.masked_fill(mask == 0, -1e9)
        
        # Biến đổi điểm số thô thành Xác suất phần trăm (0% -> 100%)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

# =====================================================================
# 2. MÔI TRƯỜNG & TÁC NHÂN AI
# =====================================================================
class DeepRLAgent:
    def __init__(self, env: DeliveryNetwork, config: Config):
        self.cfg = config
        self.env = env
        self.n_nodes = self.env.n_deliveries + 1
        self.dist_matrix = self.env.distance_matrix
        
        # Khởi tạo Mạng Nơ-ron (Đầu vào gồm 4 thông số: Tọa độ X, Tọa độ Y, Tải trọng còn lại, Thời gian)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = VRPRoutingNetwork(input_size=4, n_nodes=self.n_nodes).to(self.device)
        
        # Thuật toán tối ưu hóa Adam (Giúp AI cập nhật trọng số sau mỗi lần chạy)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        # Lấy thông tin tọa độ để feed vào Neural Network
        self.node_coords = np.zeros((self.n_nodes, 2))
        self.demand_map = np.zeros(self.n_nodes)
        self.node_coords[0] = self.env.depot
        for i, info in self.env.delivery_info.items():
            self.node_coords[i] = [info['lat'], info['lng']]
            self.demand_map[i] = info['vol']

    def get_state_tensor(self, current_node, current_capacity, current_time):
        # Trích xuất trạng thái hiện tại thành Tensor cho Neural Network hiểu
        lat, lng = self.node_coords[current_node]
        state = np.array([lat, lng, current_capacity, current_time], dtype=np.float32)
        return torch.tensor(state, device=self.device)

    def get_valid_mask_tensor(self, current_node, current_capacity, current_time, nodes_left):
        # Tạo mặt nạ nhị phân (1: Được phép đi, 0: Cấm đi)
        mask = np.zeros(self.n_nodes, dtype=np.float32)
        for node in nodes_left:
            dist = self.dist_matrix[current_node][node]
            info = self.env.delivery_info.get(node)
            if (current_time + dist + info['crowd_cost']) <= info['time_window_max']:
                if current_capacity >= self.demand_map[node]:
                    mask[node] = 1.0
        return torch.tensor(mask, device=self.device)

    def train_episode(self):
        # Một Episode là một lần AI tự xếp xe đi giao hàng cho toàn bộ bản đồ
        routes = []
        tour_time = [0 for _ in range(self.env.n_vehicles)]
        capacities = [self.env.get_vehicles()[i]['capacity'] for i in range(self.env.n_vehicles)]
        nodes_left = set(range(1, self.n_nodes))
        
        log_probs = [] # Lưu lại "nhật ký quyết định" để lát nữa phạt/thưởng
        
        for vehicle in range(self.env.n_vehicles):
            routes.append([self.cfg.DEPOT_ID])
                
        while nodes_left:
            moved = False
            for vehicle in range(self.env.n_vehicles):
                current_node = routes[vehicle][-1]
                
                if len(routes[vehicle]) > 2 and current_node == self.cfg.DEPOT_ID:
                    continue
                
                # 1. Lấy Trạng thái và Mặt nạ
                state_tensor = self.get_state_tensor(current_node, capacities[vehicle], tour_time[vehicle])
                mask_tensor = self.get_valid_mask_tensor(current_node, capacities[vehicle], tour_time[vehicle], nodes_left)
                
                # Nếu không có đường nào hợp lệ (Mask toàn 0), bắt buộc về kho
                if mask_tensor.sum() == 0:
                    if current_node != self.cfg.DEPOT_ID:
                        routes[vehicle].append(self.cfg.DEPOT_ID)
                        moved = True
                    continue
                
                # 2. HỎI MẠNG NƠ-RON (Forward Pass)
                probs = self.policy_net(state_tensor, mask_tensor)
                
                # 3. LỰA CHỌN THEO XÁC SUẤT (Sampling)
                m = Categorical(probs)
                action = m.sample() # Chọn ngã rẽ
                
                # Lưu lại logarit xác suất của quyết định này để tính Toán học (Gradient)
                log_probs.append(m.log_prob(action))
                
                next_node = action.item()
                
                # 4. Thực hiện bước đi
                routes[vehicle].append(next_node)
                nodes_left.remove(next_node)
                capacities[vehicle] -= self.demand_map[next_node]
                tour_time[vehicle] += self.dist_matrix[current_node][next_node] + self.env.delivery_info[next_node]['crowd_cost']
                moved = True
            
            if not moved:
                break

        for vehicle in range(self.env.n_vehicles):
            if routes[vehicle][-1] != self.cfg.DEPOT_ID:
                routes[vehicle].append(self.cfg.DEPOT_ID)
        
        cost = calculate_simple_cost(routes, self.env, self.dist_matrix)
        
        # =================================================================
        # THUẬT TOÁN REINFORCE (Tối ưu hóa theo Policy Gradient)
        # =================================================================
        # Reward = -Cost (Chi phí càng cao, phần thưởng càng âm)
        reward = -cost 
        
        # Tính toán hàm Loss (Độ trễ/Tổn thất)
        # Công thức: Loss = - SUM(log_prob) * Reward
        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * reward)
        
        self.optimizer.zero_grad() # Xóa bộ nhớ Gradient cũ
        
        if policy_loss: # Nếu AI có thực hiện quyết định nào đó
            loss = torch.stack(policy_loss).sum()
            loss.backward() # Lan truyền ngược (Backpropagation) để tìm lỗi
            self.optimizer.step() # Cập nhật trọng số của Mạng Nơ-ron
            
        return Solution(routes, cost)

def run_deep_rl(cfg: Config, verbose: bool = True):
    env = DeliveryNetwork(cfg)
    agent = DeepRLAgent(env, cfg) 
    
    num_episodes = cfg.NUM_ITERATIONS * cfg.NUM_ANTS # Chạy 1000 phiên huấn luyện
    best_solution = None

    print(f"Bắt đầu huấn luyện: Deep Reinforcement Learning (PyTorch Engine)...")
    print(f"Chạy trên nền tảng: {agent.device.type.upper()}")
    start_time = time.time()
    
    episodes_per_print = max(1, num_episodes // 10)

    for episode in range(1, num_episodes + 1):
        # AI tự chạy và tự cập nhật Neural Network
        current_solution = agent.train_episode()

        if not best_solution or current_solution.cost < best_solution.cost:
            best_solution = current_solution

        if verbose and episode % episodes_per_print == 0:
            print("Training Episode {}/{} | Best Cost: {:.2f} | Current Cost: {:.2f}".format(
                episode, num_episodes, best_solution.cost, current_solution.cost))

    end_time = time.time()
    
    if verbose:
        print("\n--- KẾT QUẢ CUỐI CÙNG (DEEP RL) ---")
        print("Chi phí tốt nhất: {:.2f}".format(best_solution.cost))
        print("Lộ trình: \n", best_solution.routes)
        print(f"\nThời gian huấn luyện (Deep RL): {end_time - start_time:.4f} giây")

if __name__ == "__main__":
    config = Config()
    run_deep_rl(config, verbose=True)