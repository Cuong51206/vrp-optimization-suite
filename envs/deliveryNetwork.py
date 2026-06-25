import math
import numpy as np
from scipy import spatial

class DeliveryNetwork():
    def __init__(self, settings):
        super(DeliveryNetwork, self).__init__()
        self.settings = settings # vpconfig

        self.conv_time_to_cost = settings.conv_time_to_cost
        self.n_deliveries = settings.n_deliveries
        self.n_vehicles = settings.n_vehicles
        self.depot = settings.depot

        self.delivery_info = {} # Tạo dcit để chứa thông tin khách hàng
        points = [self.depot] # Mảng tọa độ

        np.random.seed(42)

        for i in range(self.n_deliveries):
            lat = self.depot[0] + np.random.uniform(-0.1, 0.1)
            lng = self.depot[1] + np.random.uniform(-0.1, 0.1) # Tạo kinh độ, vĩ độ của từng khách hàng ngẫu nhiên dựa trên depot_id
            points.append([lat, lng])

            vol = 1 # mặc định mỗi khách hàng 1 kiện hàng
            
            dist_to_depot = math.sqrt((lat - self.depot[0])**2 + (lng - self.depot[1])**2) # Tính khoảng cách so với depot, vĩ độ = x, tung độ = y
            time_window_min = dist_to_depot * 10 # thời gian dự kiến giao sớm nhất
            time_window_max = time_window_min + 100000 # thời gian dự kiến giao trễ nhất

            self.delivery_info[i + 1] = {
                'id': i + 1,
                'lat': lat,
                'lng': lng,
                'vol': vol,
                'time_window_min': time_window_min,
                'time_window_max': time_window_max,
                'crowd_cost': 0 # chi phí phát sinh
            }

        self.distance_matrix = spatial.distance_matrix(points, points) # tính khoảng cách giữa các điểm (khách hàng) -> tạo ra mảng 2c 101*101

        self.vehicles = [] # mảng lưu sức chứa và chi phí xuất bãi của từng xe
        for i in range(self.n_vehicles):
            self.vehicles.append({
                'capacity': settings.vols_vehicles[i],
                'cost': settings.costs_vehicles[i]
            })

    def get_delivery(self): # getter
        return self.delivery_info

    def get_vehicles(self): # getter
        return self.vehicles

    def evaluate_VRP(self, VRP_solution): # hàm chấm điểm lộ trình; VRP_solution là mảng 2c lưu lộ trình của từng xe
        usage_cost = 0 # chi phí xuất bãi
        for k in range(self.n_vehicles):
            if len(VRP_solution[k]) > 0: # nếu lộ trình xe đó không rỗng thì cộng thêm chi phí xuất bãi
                usage_cost += self.vehicles[k]['cost']
                
        travel_cost = 0 # chi phí giao hàng
        for k in range(self.n_vehicles): # duyệt qua từng xe
            tour_time = 0 # thời gian chạy
            for i in range(1, len(VRP_solution[k])-1): # duyệt qua từng trạm của xe, bỏ trạm xuất phát
                tour_time += self.distance_matrix[VRP_solution[k][i - 1], VRP_solution[k][i]] # tính khoảng cách giữa trạm trước và hiện tại | mặc định v = 1
                tour_time = max(tour_time, self.delivery_info[VRP_solution[k][i]]['time_window_min']) # nếu xe đến quá sớm thì ép lên thời gian khách hàng mở cửa
                if tour_time > self.delivery_info[VRP_solution[k][i]]['time_window_max']: # nếu trễ thì loại
                    raise Exception('Too Late for Delivery: ', VRP_solution[k][i])
            travel_cost += self.conv_time_to_cost * tour_time # đổi thời gian sang chi phí
            
        return usage_cost + travel_cost # return về tổng chi phí của lần duyệt này