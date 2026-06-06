import json
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self):
        try:
            with open("./configs/vrpconfig.json", 'r', encoding='utf-8') as f:
                config_json = json.load(f)

            # Thông số Thuật toán ACO
            self.NUM_ANTS = config_json.get("num_ants", 100)
            self.ANT_CAPACITY = config_json.get("ant_capacity", 6000)
            self.NUM_ITERATIONS = config_json.get("num_iterations", 10)
            self.DEPOT_ID = config_json.get("depot_id", 0)
            self.ALPHA = config_json.get("alpha", 0.5)  
            self.BETA = config_json.get("beta", 0.5)  
            self.RHO = config_json.get("rho", 0.05)  
            self.Q0 = config_json.get("q0", 0.9)

            # Đường dẫn Data
            self.DELIVERY_INFO_PATH = config_json.get("deliver_info_path", "./envs/data/delivery_info.json")
            self.DISTANCE_MATRIX_PATH = config_json.get("distance_matrix_path", "./envs/data/distance_matrix.csv")

            # Thông số Môi trường (DeliveryNetwork)
            self.conv_time_to_cost = config_json.get("conv_time_to_cost", 14)
            self.n_deliveries = config_json.get("n_deliveries", 100)
            self.n_vehicles = config_json.get("n_vehicles", 8)
            self.vols_vehicles = config_json.get("vols_vehicles", [])
            self.costs_vehicles = config_json.get("costs_vehicles", [])
            self.depot = config_json.get("depot", [])

        except Exception as e:
            raise Exception(f"Lỗi đọc file Config: {e}. Vui lòng kiểm tra lại đường dẫn './configs/vrpconfig.json'")