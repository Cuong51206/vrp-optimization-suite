import math
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

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

        # ĐÓNG BĂNG BẢN ĐỒ BẰNG RANDOM SEED ĐỂ 4 THUẬT TOÁN CHẠY CHUNG 1 TEST CASE
        np.random.seed(42)

        # 1. TỰ ĐỘNG SINH DỮ LIỆU KHÁCH HÀNG (Đã nới lỏng điều kiện)
        for i in range(self.n_deliveries):
            lat = self.depot[0] + np.random.uniform(-0.1, 0.1)
            lng = self.depot[1] + np.random.uniform(-0.1, 0.1)
            points.append([lat, lng])

            vol = 1 # Cố định khối lượng nhẹ để xe có thể chở hết
            
            dist_to_depot = math.sqrt((lat - self.depot[0])**2 + (lng - self.depot[1])**2)
            time_window_min = dist_to_depot * 10
            # Mở rộng cửa sổ thời gian ra vô hạn để không bị lỗi trễ giờ
            time_window_max = time_window_min + 100000 

            self.delivery_info[i + 1] = {
                'id': i + 1,
                'lat': lat,
                'lng': lng,
                'crowdsourced': 0,
                'vol': vol,
                'crowd_cost': vol * 2.0,
                'p_failed': 0.1,
                'time_window_min': time_window_min,
                'time_window_max': time_window_max,
            }

        self.distance_matrix = spatial.distance_matrix(points, points)

        self.vehicles = []
        for i in range(self.n_vehicles):
            self.vehicles.append({
                'capacity': settings.vols_vehicles[i],
                'cost': settings.costs_vehicles[i]
            })

    def prepare_crowdsourcing_scenario(self):
        self.__fail_crowdship = []
        for _, ele in self.delivery_info.items():
            if np.random.uniform() < ele['p_failed']:
                self.__fail_crowdship.append(ele['id'])

    def run_crowdsourcing(self, delivery_to_crowdship):
        id_remaining_deliveries = [key for key in self.delivery_info]
        tot_crowd_cost = 0
        for key, ele in self.delivery_info.items():
            ele['crowdsourced'] = 0
        for i in delivery_to_crowdship:
            if self.delivery_info[i]['id'] not in self.__fail_crowdship:
                id_remaining_deliveries.remove(i)
                tot_crowd_cost += self.delivery_info[i]['crowd_cost']
                self.delivery_info[i]['crowdsourced'] = 1
        remaining_deliveries = {}
        for i in id_remaining_deliveries:
            remaining_deliveries[i] = self.delivery_info[i]
        return remaining_deliveries, tot_crowd_cost

    def get_delivery(self):
        return self.delivery_info

    def get_vehicles(self):
        return self.vehicles

    def evaluate_VRP(self, VRP_solution):
        usage_cost = 0
        for k in range(self.n_vehicles):
            if len(VRP_solution[k]) > 0:
                usage_cost += self.vehicles[k]['cost']
        travel_cost = 0
        for k in range(self.n_vehicles):
            tour_time = 0
            for i in range(1, len(VRP_solution[k])-1):
                tour_time += self.distance_matrix[VRP_solution[k][i - 1], VRP_solution[k][i]]
                tour_time = max(tour_time, self.delivery_info[VRP_solution[k][i]]['time_window_min'])
                if tour_time > self.delivery_info[VRP_solution[k][i]]['time_window_max']:
                    raise Exception('Too Late for Delivery: ', VRP_solution[k][i])
            travel_cost += self.conv_time_to_cost * tour_time
        return usage_cost + travel_cost