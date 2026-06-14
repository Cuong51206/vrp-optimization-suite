import json
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self):
        try:
            with open("./configs/vrpconfig.json", 'r', encoding='utf-8') as f:
                config_json = json.load(f)
            # num_ants         : Số lượng kiến được thả ra trong một vòng lặp.
            # ant_capacity     : Sức chứa mặc định của mỗi con kiến.
            # num_iterations   : Tổng số thế hệ (vòng lặp) AI sẽ chạy.
            # alpha            : Trọng số "vết mùi" (mức độ đi theo bầy đàn).
            # beta             : Trọng số "khoảng cách" (mức độ ưu tiên trạm gần nhất).
            # rho              : Tốc độ bay hơi vết mùi (giúp AI quên đường cũ).
            # q0               : Xác suất chọn luôn đường tốt nhất (Khai thác vs Khám phá).
            # depot_id         : ID định danh của trạm xuất phát (Kho).
            # depot            : Tọa độ GPS [Vĩ độ, Kinh độ] của Kho.
            # n_deliveries     : Tổng số khách hàng cần giao.
            # deliver_info_path: Đường dẫn file chứa tọa độ khách hàng (tùy chọn).
            # distance_matrix_path: Đường dẫn file ma trận khoảng cách (tùy chọn).
            # n_vehicles       : Số lượng xe tối đa có sẵn trong bãi.
            # vols_vehicles    : Mảng lưu sức chứa cụ thể của từng chiếc xe.
            # costs_vehicles   : Mảng lưu chi phí xuất bến (thuê xe) của từng chiếc.
            # conv_time_to_cost: Hệ số quy đổi thời gian chạy xe thành tiền phạt/chi phí.
            # Thông số Thuật toán ACO
            self.NUM_ANTS = config_json.get("num_ants", 100) # sử dụng get để tránh lỗi không có dữ liệu (val, dèault)
            self.ANT_CAPACITY = config_json.get("ant_capacity", 6000)
            self.NUM_ITERATIONS = config_json.get("num_iterations", 10)
            self.DEPOT_ID = config_json.get("depot_id", 0)
            self.ALPHA = config_json.get("alpha", 0.5)  
            self.BETA = config_json.get("beta", 0.5)  
            self.RHO = config_json.get("rho", 0.05)  
            self.Q0 = config_json.get("q0", 0.9)

            self.DELIVERY_INFO_PATH = config_json.get("deliver_info_path", "./envs/data/delivery_info.json")
            self.DISTANCE_MATRIX_PATH = config_json.get("distance_matrix_path", "./envs/data/distance_matrix.csv")

            self.conv_time_to_cost = config_json.get("conv_time_to_cost", 14)
            self.n_deliveries = config_json.get("n_deliveries", 100)
            self.n_vehicles = config_json.get("n_vehicles", 8)
            self.vols_vehicles = config_json.get("vols_vehicles", [])
            self.costs_vehicles = config_json.get("costs_vehicles", [])
            self.depot = config_json.get("depot", [])

        except Exception as e:
            raise Exception(f"Lỗi đọc file Config: {e}. Vui lòng kiểm tra lại đường dẫn './configs/vrpconfig.json'")