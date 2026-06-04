import json
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self):
        try:
            json_rc = open("./envs/vrpconfig.json", 'r')
            config_json = json.load(json_rc)
            json_rc.close()

            # Thông số Thuật toán ACO
            self.NUM_ANTS = config_json["num_ants"]
            self.ANT_CAPACITY = config_json["ant_capacity"]
            self.NUM_ITERATIONS = config_json["num_iterations"]
            self.DEPOT_ID = config_json["depot_id"]
            self.ALPHA = config_json["alpha"]  
            self.BETA = config_json["beta"]  
            self.RHO = config_json["rho"]  
            self.Q0 = config_json.get("q0", 0.9)

            # Đường dẫn Data
            self.DELIVERY_INFO_PATH = config_json["deliver_info_path"]
            self.DISTANCE_MATRIX_PATH = config_json["distance_matrix_path"]

            # Thông số Môi trường (DeliveryNetwork)
            self.conv_time_to_cost = config_json["conv_time_to_cost"]
            self.n_deliveries = config_json["n_deliveries"]
            self.n_vehicles = config_json["n_vehicles"]
            self.vols_vehicles = config_json["vols_vehicles"]
            self.costs_vehicles = config_json["costs_vehicles"]
            self.depot = config_json["depot"]

        except Exception as e:
            raise Exception(f"Lỗi đọc file Config: {e}")