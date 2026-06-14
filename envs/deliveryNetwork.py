import math
import numpy as np
from scipy import spatial

class DeliveryNetwork():
    def __init__(self, settings):
        super(DeliveryNetwork, self).__init__()
        self.settings = settings

        self.conv_time_to_cost = settings.conv_time_to_cost
        self.n_deliveries = settings.n_deliveries
        self.n_vehicles = settings.n_vehicles
        self.depot = settings.depot

        self.delivery_info = {}
        points = [self.depot]

        np.random.seed(42)

        for i in range(self.n_deliveries):
            lat = self.depot[0] + np.random.uniform(-0.1, 0.1)
            lng = self.depot[1] + np.random.uniform(-0.1, 0.1)
            points.append([lat, lng])

            vol = 1 
            
            dist_to_depot = math.sqrt((lat - self.depot[0])**2 + (lng - self.depot[1])**2)
            time_window_min = dist_to_depot * 10
            time_window_max = time_window_min + 100000 

            self.delivery_info[i + 1] = {
                'id': i + 1,
                'lat': lat,
                'lng': lng,
                'vol': vol,
                'time_window_min': time_window_min,
                'time_window_max': time_window_max,
                'crowd_cost': 0 
            }

        self.distance_matrix = spatial.distance_matrix(points, points)

        self.vehicles = []
        for i in range(self.n_vehicles):
            self.vehicles.append({
                'capacity': settings.vols_vehicles[i],
                'cost': settings.costs_vehicles[i]
            })

    def get_delivery(self):
        return self.delivery_info

    def get_vehicles(self):
        return self.vehicles

    def evaluate_VRP(self, VRP_solution):
        usage_cost = 0 # chi phí xuất bãi
        for k in range(self.n_vehicles):
            if len(VRP_solution[k]) > 0:
                usage_cost += self.vehicles[k]['cost']
                
        travel_cost = 0 # chi phí giao hàng
        for k in range(self.n_vehicles):
            tour_time = 0
            for i in range(1, len(VRP_solution[k])-1): # bỏ trạm xuất phát
                tour_time += self.distance_matrix[VRP_solution[k][i - 1], VRP_solution[k][i]] # tính khoảng cách giữa trạm trước và hiện tại | mặc định v = 1
                tour_time = max(tour_time, self.delivery_info[VRP_solution[k][i]]['time_window_min']) # nếu xe đến quá sớm thì ép lên thời gian khách hàng mở cửa
                if tour_time > self.delivery_info[VRP_solution[k][i]]['time_window_max']: # nếu trễ thì loại
                    raise Exception('Too Late for Delivery: ', VRP_solution[k][i])
            travel_cost += self.conv_time_to_cost * tour_time # đổi thời gian sang chi phí
            
        return usage_cost + travel_cost