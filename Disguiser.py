import numpy as np
import pandas as pd
import time
import random
import pulp as pl
import math


def calculate_distance(user, server):
    R = 6371
    lat1, lon1 = np.radians(user)
    lat2, lon2 = np.radians(server)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c * 1000


def read_data(num_servers, num_users):
    users = pd.read_csv('users.csv').values
    servers = pd.read_csv('servers.csv').values

    selected_servers = []
    selected_users = []
    selected_radii = []

    server_indices = np.random.choice(len(servers), num_servers, replace=False)
    selected_servers = servers[server_indices]
    selected_radii = selected_servers[:, 5]

    cov_initial = np.zeros((num_users, num_servers), dtype=int)
    server_to_users = {i: [] for i in range(num_servers)}
    assigned_users = set()
    avg_users_per_server = num_users // num_servers

    for idx, server in enumerate(selected_servers):
        if idx < num_servers - 1:
            target_users_count = np.random.randint(avg_users_per_server - 1, avg_users_per_server + 1)
        else:
            target_users_count = num_users - len(selected_users)

        for i, user in enumerate(users):
            if len(server_to_users[idx]) >= target_users_count:
                break
            if tuple(user) not in assigned_users:
                distance = calculate_distance(user[:2], server[:2])
                if distance < server[5]:
                    server_to_users[idx].append(user)
                    assigned_users.add(tuple(user))
                    if len(selected_users) < num_users:
                        selected_users.append(user)

                    if len(selected_users) <= num_users:
                        cov_initial[len(selected_users) - 1, idx] = 1

    selected_users = np.array(selected_users[:num_users])
    selected_users_df = pd.DataFrame(selected_users,
                                     columns=['Latitude', 'Longitude', 'vCPUs', 'Memory', 'GPUs', 'Privacy_Metric',
                                              'Run_Cost'])
    selected_servers_df = pd.DataFrame(selected_servers,
                                       columns=['Latitude', 'Longitude', 'vCPUs', 'Memory', 'GPUs', 'Radius'])

    selected_users_df.to_csv('selected_users.csv', index=False)
    selected_servers_df.to_csv('selected_servers.csv', index=False)

    return np.array(selected_users), np.array(selected_servers), np.array(selected_radii), cov_initial


def geo_laplace_encryption(users_file, laplace_encrypted_users_file, epsilon=0.5, min_dist=10, max_dist=60):
    def planar_laplace_mechanism(epsilon, lat, lon, min_dist, max_dist):
        earth_radius = 6371.0
        min_radians = min_dist / (earth_radius * 1000)
        max_radians = max_dist / (earth_radius * 1000)

        def laplace_noise(b):
            u = np.random.uniform(-0.5, 0.5)
            return -b * np.sign(u) * np.log(1 - 2 * abs(u))
        b = 1 / epsilon
        r = laplace_noise(b)
        r = np.clip(r, min_radians, max_radians)
        theta = np.random.uniform(0, 2 * np.pi)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat_delta = r * np.cos(theta)
        lon_delta = r * np.sin(theta) / np.cos(lat_rad)
        lat_star = lat_rad + lat_delta
        lon_star = lon_rad + lon_delta
        lat_star_deg = np.degrees(lat_star)
        lon_star_deg = np.degrees(lon_star)

        return lat_star_deg, lon_star_deg

    def encrypt_location(data_df, epsilon, min_dist, max_dist):
        encrypted_data = data_df.copy()

        for idx, row in encrypted_data.iterrows():
            lat = row['Latitude']
            lon = row['Longitude']
            lat_star, lon_star = planar_laplace_mechanism(epsilon, lat, lon, min_dist, max_dist)
            encrypted_data.at[idx, 'Latitude'] = lat_star
            encrypted_data.at[idx, 'Longitude'] = lon_star

        return encrypted_data

    users_df = pd.read_csv(users_file)
    encrypted_users_df = encrypt_location(users_df, epsilon, min_dist, max_dist)
    encrypted_users_df.to_csv(laplace_encrypted_users_file, index=False)

    # print(f"加密后的用户数据已保存至 {laplace_encrypted_users_file}")


def geo_gaussian_encryption(users_file, encrypted_users_file, epsilon=0.5, delta=1e-5, min_dist=10, max_dist=60):
    def planar_gaussian_mechanism(epsilon, delta, lat, lon, min_dist, max_dist):
        earth_radius = 6371.0
        min_radians = min_dist / (earth_radius * 1000)
        max_radians = max_dist / (earth_radius * 1000)
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        r = np.random.normal(0, sigma)
        r = np.clip(r, min_radians, max_radians)
        theta = np.random.uniform(0, 2 * np.pi)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat_delta = r * np.cos(theta)
        lon_delta = r * np.sin(theta) / np.cos(lat_rad)
        lat_star = lat_rad + lat_delta
        lon_star = lon_rad + lon_delta
        lat_star_deg = np.degrees(lat_star)
        lon_star_deg = np.degrees(lon_star)

        return lat_star_deg, lon_star_deg

    def encrypt_location(data_df, epsilon, delta, min_dist, max_dist):
        encrypted_data = data_df.copy()

        for idx, row in encrypted_data.iterrows():
            lat = row['Latitude']
            lon = row['Longitude']
            lat_star, lon_star = planar_gaussian_mechanism(epsilon, delta, lat, lon, min_dist, max_dist)
            encrypted_data.at[idx, 'Latitude'] = lat_star
            encrypted_data.at[idx, 'Longitude'] = lon_star

        return encrypted_data
    users_df = pd.read_csv(users_file)
    encrypted_users_df = encrypt_location(users_df, epsilon, delta, min_dist, max_dist)
    encrypted_users_df.to_csv(encrypted_users_file, index=False)

    # print(f"加密后的用户数据已保存至 {encrypted_users_file}")


def calculate_position_error_laplace(selected_file='selected_users.csv', encrypted_file='laplace_encrypted_users.csv'):
    selected_users = pd.read_csv(selected_file).values
    encrypted_users = pd.read_csv(encrypted_file).values
    distance_differences = []
    for selected_user, encrypted_user in zip(selected_users, encrypted_users):
        original_location = selected_user[:2]
        encrypted_location = encrypted_user[:2]
        distance_difference = calculate_distance(original_location, encrypted_location)
        distance_differences.append(distance_difference)

    return distance_differences


def calculate_position_error_gaussian(selected_file='selected_users.csv', encrypted_file='gaussian_encrypted_users.csv'):
    selected_users = pd.read_csv(selected_file).values
    encrypted_users = pd.read_csv(encrypted_file).values
    distance_differences = []
    for selected_user, encrypted_user in zip(selected_users, encrypted_users):
        original_location = selected_user[:2]
        encrypted_location = encrypted_user[:2]
        distance_difference = calculate_distance(original_location, encrypted_location)
        distance_differences.append(distance_difference)

    return distance_differences


def Solver_initial_allocation(U, S, c_i_j, c_i_cloud, L_j, L_i_j, cov_noise, r, omega, d):
    prob = pl.LpProblem("Edge_User_Allocation_P2", pl.LpMaximize)
    x = pl.LpVariable.dicts("x", (range(len(U)), range(len(S) + 1)), cat='Binary')
    F_u = pl.lpSum([x[i][j] for i in range(len(U)) for j in range(len(S))]) / len(U)
    F_g = pl.lpSum([d[i] * x[i][j] / (len(U) * r[j]) for i in range(len(U)) for j in range(len(S))])
    C_max = pl.lpSum([c_i_cloud[i] for i in range(len(U))])
    F_c = (pl.lpSum([c_i_cloud[i] * x[i][len(S)] for i in range(len(U))]) +
           pl.lpSum([c_i_j[i][j] * x[i][j] for i in range(len(U)) for j in range(len(S))])) / C_max
    prob += omega[0] * F_u + omega[1] * F_g - omega[2] * F_c, "Objective"

    for i in range(len(U)):
        prob += pl.lpSum([x[i][j] for j in range(len(S)) if cov_noise[i, j]]) + x[i][len(S)] == 1, f"user_{i}_allocation"

    for i in range(len(U)):
        for j in range(len(S)):
            distance_to_server = calculate_distance(U[i][:2], S[j][:2])  # 假设 U 和 S 存储了经纬度信息
            if distance_to_server >= r[j]:
                prob += x[i][j] == 0, f"distance_limit_user_{i}_server_{j}"

    for j in range(len(S)):
        prob += pl.lpSum([x[i][j] * L_i_j[i][j] for i in range(len(U))]) <= L_j[j], f"capacity_server_{j}"

    start_time = time.time()
    prob.solve()
    end_time = time.time()
    initial_runtime = end_time - start_time

    x_initial_i_j = np.array([[pl.value(x[i][j]) for j in range(len(S))] for i in range(len(U))])
    x_initial_i_cloud = np.array([pl.value(x[i][len(S)]) for i in range(len(U))])
    print('F_u',F_u)
    print('F_g', F_g)
    print('F_c', F_c)
    print('x_final_i_cloud',x_initial_i_cloud)
    initial_f = pl.value(prob.objective)

    return initial_f, initial_runtime, x_initial_i_j, x_initial_i_cloud



def LR_EUA_initial_allocation(U, S, c_i_j, c_i_cloud, L_j, L_i_j, cov_noise, omega, d, r):
    num_users = len(U)
    num_servers = len(S)
    x_initial_i_j = np.zeros((num_users, num_servers))
    x_initial_i_cloud = np.zeros(num_users)
    L_current = np.zeros(num_servers)
    start_time = time.time()
    for i in range(num_users):
        S_cov = [j for j in range(num_servers) if cov_noise[i][j] == 1]
        S_ava = [j for j in S_cov if L_current[j] + L_i_j[i][j] <= L_j[j]]
        if len(S_ava) == 0:
            x_initial_i_cloud[i] = 1
        else:
            utility_values = []
            for j in S_ava:
                f_Sj = (omega[0] * (1 / num_users) +
                        omega[1] * (d[i] / (num_users * r[j])) -
                        omega[2] * c_i_j[i][j])
                utility_values.append(f_Sj)
            best_server_index = S_ava[np.argmax(utility_values)]
            x_initial_i_j[i][best_server_index] = 1
            L_current[best_server_index] += L_i_j[i][best_server_index]

    F_u = np.sum(x_initial_i_j) / num_users
    F_g = 0
    for i in range(num_users):
        for j in range(num_servers):
            F_g += (d[i] * x_initial_i_j[i][j]) / (num_users * r[j])

    C_max = np.sum(c_i_cloud)
    F_c = 0
    for i in range(num_users):
        F_c += c_i_cloud[i] * x_initial_i_cloud[i]
        for j in range(num_servers):
            F_c += c_i_j[i][j] * x_initial_i_j[i][j]
    F_c = F_c / C_max
    initial_f = omega[0] * F_u + omega[1] * F_g - omega[2] * F_c
    initial_runtime = time.time() - start_time

    return initial_f, initial_runtime, x_initial_i_j, x_initial_i_cloud


def generate_default_solution(U, S, cov_noise, L_current, L_i_j, L_j):
    default_solution = []
    for i in range(len(U)):
        valid_servers = [j for j in range(len(S)) if cov_noise[i][j] == 1 and L_current[j] + L_i_j[i][j] <= L_j[j]]
        if valid_servers:
            default_solution.append(random.choice(valid_servers))
        else:
            default_solution.append('cloud')
    return default_solution


def GA_initial_allocation(U, S, c_i_j, c_i_cloud, L_j, L_i_j, cov_noise, omega, d, r,pop_size=100, generations=500, mutation_rate=0.1, crossover_rate=0.8, early_stop_gen=50):
    num_users = len(U)
    num_servers = len(S)
    d = np.array(d)
    r = np.array(r)
    best_solution = None
    best_fitness = -float('inf')
    L_current = np.zeros(num_servers)
    no_improvement_count = 0

    def initialize_population():
        population = []
        for _ in range(pop_size):
            individual = []
            for i in range(num_users):
                valid_servers = [j for j in range(num_servers) if
                                 cov_noise[i][j] == 1 and L_current[j] + L_i_j[i][j] <= L_j[j]]
                if valid_servers:
                    individual.append(random.choice(valid_servers))
                else:
                    individual.append('cloud')
            population.append(individual)
        return population

    def fitness_function(individual):
        x_initial_i_j = np.zeros((num_users, num_servers))
        x_initial_i_cloud = np.zeros(num_users)
        for i in range(num_users):
            if individual[i] == 'cloud':
                x_initial_i_cloud[i] = 1
            else:
                x_initial_i_j[i][individual[i]] = 1
        for j in range(num_servers):
            if np.sum(x_initial_i_j[:, j] * L_i_j[:, j]) > L_j[j]:
                return None

        F_u = np.sum(x_initial_i_j) / num_users
        F_g = np.sum((d[:, None] * x_initial_i_j) / (num_users * r[None, :]))
        C_max = np.sum(c_i_cloud)
        F_c = (np.sum(c_i_cloud * x_initial_i_cloud) + np.sum(c_i_j * x_initial_i_j)) / C_max
        fitness = omega[0] * F_u + omega[1] * F_g - omega[2] * F_c

        return fitness

    def selection(population, fitnesses):
        tournament_size = 3
        selected = []
        for _ in range(2):
            candidate_indices = np.random.choice(len(population), tournament_size, replace=False)
            candidates = [population[i] for i in candidate_indices]
            fitness_candidates = [fitnesses[i] for i in candidate_indices]
            selected.append(candidates[np.argmax(fitness_candidates)])
        return selected

    def crossover(parent1, parent2):
        if random.random() < crossover_rate:
            child1 = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(num_users)]
            child2 = [parent2[i] if random.random() < 0.5 else parent1[i] for i in range(num_users)]
            return child1, child2
        return parent1, parent2

    def mutate(individual):
        if random.random() < mutation_rate:
            idx = random.randint(0, num_users - 1)
            valid_servers = [j for j in range(num_servers) if
                             cov_noise[idx][j] == 1 and L_current[j] + L_i_j[idx][j] <= L_j[j]]
            if valid_servers:
                individual[idx] = random.choice(valid_servers)
            else:
                individual[idx] = 'cloud'
        return individual

    population = initialize_population()
    start_time = time.time()
    for generation in range(generations):
        fitnesses = []
        for individual in population:
            fitness = fitness_function(individual)
            if fitness is not None:
                fitnesses.append(fitness)
            else:
                fitnesses.append(-float('inf'))

        fitnesses = np.array(fitnesses)
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= early_stop_gen:
            # print(f"提前终止于第 {generation} 代")
            break

        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population[:pop_size]

    runtime = time.time() - start_time
    best_fitness = best_fitness
    x_initial_i_j = np.zeros((num_users, num_servers))
    x_initial_i_cloud = np.zeros(num_users)

    if best_solution is None:
        # print("未找到最佳解，返回改进的默认解。")
        best_solution = generate_default_solution(U, S, cov_noise, L_current, L_i_j, L_j)
    for i in range(num_users):
        if best_solution[i] == 'cloud':
            x_initial_i_cloud[i] = 1
        else:
            x_initial_i_j[i][best_solution[i]] = 1
    return best_fitness, runtime, x_initial_i_j, x_initial_i_cloud


def reallocate_failed_users(U, S, L_j, L_i_j, c_i_j, c_i_cloud, x_initial_i_j, x_initial_i_cloud, cov_initial, r, omega, d):
    start_time = time.time()
    x_final_i_j = np.copy(x_initial_i_j)
    x_final_i_cloud = np.copy(x_initial_i_cloud)
    failed_users = []
    for i, user in enumerate(U):
        cover_Ui_server = [j for j, server in enumerate(S) if cov_initial[i][j] == 1]
        initial_server = np.argmax(x_initial_i_j[i]) if np.sum(x_initial_i_j[i]) > 0 else -1
        if initial_server != -1 and initial_server not in cover_Ui_server:
            failed_users.append(i)
    print('failed_users',failed_users)
    successful_users = [i for i in range(len(U)) if i not in failed_users]
    for i in failed_users:
        available_servers = []
        for j, server in enumerate(S):
            distance = calculate_distance(U[i][:2], S[j][:2])
            if distance <= r[j]:
                current_capacity_used = np.sum([x_final_i_j[k][j] * L_i_j[k][j] for k in range(len(U))])
                if current_capacity_used + L_i_j[i][j] <= L_j[j]:
                    available_servers.append(j)

        if not available_servers:
            x_final_i_cloud[i] = 1
            x_final_i_j[i] = np.zeros(len(S))
        else:
            max_utility = float('-inf')
            best_server = -1
            for j in available_servers:
                utility = (omega[0] / len(U)) + (omega[1] * d[i] / (len(U) * r[j])) - (omega[2] * c_i_j[i][j])
                if utility > max_utility:
                    max_utility = utility
                    best_server = j
            if best_server != -1:
                x_final_i_j[i] = np.zeros(len(S))
                x_final_i_j[i][best_server] = 1
                x_final_i_cloud[i] = 0

    F_u = np.sum(x_final_i_j) / len(U)
    F_g = np.sum([d[i] * x_final_i_j[i][j] / (len(U) * r[j]) for i in range(len(U)) for j in range(len(S))])
    C_max = np.sum(c_i_cloud)
    F_c = (np.sum([c_i_cloud[i] * x_final_i_cloud[i] for i in range(len(U))]) +
           np.sum([c_i_j[i][j] * x_final_i_j[i][j] for i in range(len(U)) for j in range(len(S))])) / C_max

    final_cost = omega[0] * F_u + omega[1] * F_g - omega[2] * F_c
    if successful_users:
        F_u_initial = np.sum(x_initial_i_j[successful_users]) / len(successful_users)  
        F_g_initial = np.sum([d[i] * x_initial_i_j[i][j] / (len(successful_users) * r[j]) for i in successful_users for j in range(len(S))])
        F_c_initial = (np.sum([c_i_cloud[i] * x_initial_i_cloud[i] for i in successful_users]) +
                       np.sum([c_i_j[i][j] * x_initial_i_j[i][j] for i in successful_users for j in range(len(S))])) / C_max
        f_true_initial = omega[0] * F_u_initial + omega[1] * F_g_initial - omega[2] * F_c_initial
    else:
        f_true_initial = 0

    end_time = time.time()
    runtime = end_time - start_time

    return F_u,F_g,F_c,final_cost, runtime, x_final_i_j, x_final_i_cloud,f_true_initial


def filter_results(method_results, num_repeats):
    filtered_fu = []
    filtered_fp = []
    filtered_fc = []
    filtered_f_initial = []
    filtered_f_final = []
    filtered_time_total = []

    for i in range(len(method_results['f_initial']) // num_repeats):
        current_fu = method_results['fu'][i * num_repeats:(i + 1) * num_repeats]
        current_fp = method_results['fp'][i * num_repeats:(i + 1) * num_repeats]
        current_fc = method_results['fc'][i * num_repeats:(i + 1) * num_repeats]
        current_f_initial = method_results['f_initial'][i * num_repeats:(i + 1) * num_repeats]
        current_f_final = method_results['f_final'][i * num_repeats:(i + 1) * num_repeats]
        current_time_total = method_results['time_total'][i * num_repeats:(i + 1) * num_repeats]

        mean_fu = np.mean(current_fu)
        mean_fp = np.mean(current_fp)
        mean_fc = np.mean(current_fc)
        mean_f_initial = np.mean(current_f_initial)
        mean_f_final = np.mean(current_f_final)
        mean_time_total = np.mean(current_time_total)

        filtered_current_fu = [fu for fu in current_fu
                                      if (abs(fu - mean_fu) <= 0.6 * mean_fu)
                                      or (abs(fu - mean_fu) <= 1e-5)]
        filtered_current_fp = [fp for fp in current_fp
                               if (abs(fp - mean_fp) <= 0.6 * mean_fp)
                               or (abs(fp - mean_fp) <= 1e-5)]
        filtered_current_fc = [fc for fc in current_fc
                               if (abs(fc - mean_fc) <= 0.6 * mean_fc)
                               or (abs(fc - mean_fc) <= 1e-5)]
        filtered_current_f_initial = [f_initial for f_initial in current_f_initial
                                      if (abs(f_initial - mean_f_initial) <= 0.6 * mean_f_initial)
                                      or (abs(f_initial - mean_f_initial) <= 1e-5)]
        filtered_current_f_final = [f_final for f_final in current_f_final
                                    if (abs(f_final - mean_f_final) <= 0.6 * mean_f_final)
                                    or (abs(f_final - mean_f_final) <= 1e-5)]
        filtered_current_time_total = [time_total for time_total in current_time_total
                                       if (abs(time_total - mean_time_total) <= 0.6 * mean_time_total)
                                       or (abs(time_total - mean_time_total) <= 1e-5)]

        if len(filtered_current_fu) == 0:
            filtered_current_fu = [mean_fu]
        if len(filtered_current_fp) == 0:
            filtered_current_fp = [mean_fp]
        if len(filtered_current_fc) == 0:
            filtered_current_fc = [mean_fc]
        if len(filtered_current_f_initial) == 0:
            filtered_current_f_initial = [mean_f_initial]
        if len(filtered_current_f_final) == 0:
            filtered_current_f_final = [mean_f_final]
        if len(filtered_current_time_total) == 0:
            filtered_current_time_total = [mean_time_total]

        filtered_fu.extend(filtered_current_fu)
        filtered_fp.extend(filtered_current_fp)
        filtered_fc.extend(filtered_current_fc)
        filtered_f_initial.extend(filtered_current_f_initial)
        filtered_f_final.extend(filtered_current_f_final)
        filtered_time_total.extend(filtered_current_time_total)

    return filtered_fu, filtered_fp, filtered_fc, filtered_f_initial, filtered_f_final, filtered_time_total


mean = 8
num_servers = 15
num_users = 100
omega = [1/3,1/3,1/3]
U, S ,r ,cov_initial= read_data(num_servers, num_users)
# print('r',r)
# print('cov_initial',cov_initial)
users_file = 'selected_users.csv'
laplace_encrypted_users_file = 'laplace_encrypted_users.csv'
gaussian_encrypted_users_file = 'gaussian_encrypted_users.csv'
epsilon = math.log(2)
min_dist = 10
max_dist = 60
delta=1e-5
geo_laplace_encryption(users_file,laplace_encrypted_users_file,epsilon,min_dist, max_dist)
geo_gaussian_encryption(users_file, gaussian_encrypted_users_file, epsilon, delta, min_dist, max_dist)
L_j = np.random.normal(mean, 1, size=len(S))
L_i_j = np.ones((len(U), len(S)), dtype=int)
c_i_j = np.random.rand(len(U), len(S))
c_i_cloud = np.random.rand(len(U)) * 3
d_laplace = calculate_position_error_laplace()
d_gaussian = calculate_position_error_gaussian()

cov_laplace = np.zeros((len(U), len(S)), dtype=int)
encrypted_users = pd.read_csv('laplace_encrypted_users.csv').values
for i in range(len(U)):
    user_location = encrypted_users[i][:2]
    for j in range(len(S)):
        server_location = S[j][:2]
        distance_to_server = calculate_distance(user_location, server_location)
        if distance_to_server < r[j]:
            cov_laplace[i, j] = 1
# print('cov_laplace',cov_laplace)

cov_gaussian = np.zeros((len(U), len(S)), dtype=int)
encrypted_users = pd.read_csv('gaussian_encrypted_users.csv').values
for i in range(len(U)):
    user_location = encrypted_users[i][:2]
    for j in range(len(S)):
        server_location = S[j][:2]
        distance_to_server = calculate_distance(user_location, server_location)
        if distance_to_server < r[j]:
            cov_gaussian[i, j] = 1
# print('cov_gaussian',cov_gaussian)

f_initial_solver_L, time_initial_solver_L, x_initial_i_j_solver_L, x_initial_i_cloud_solver_L = Solver_initial_allocation(U, S, c_i_j, c_i_cloud, L_j, L_i_j, cov_laplace, r, omega, d_laplace)
fu_final_solver_L,fp_final_solver_L,fc_final_solver_L,f_final_solver_L, time_final_solver_L, x_final_i_j_solver_L, x_final_i_cloud_solver_L,f_true_initial__solver_L = reallocate_failed_users(U, S, L_j, L_i_j, c_i_j, c_i_cloud, x_initial_i_j_solver_L, x_initial_i_cloud_solver_L, cov_initial, r, omega, d_laplace)
print(f"Solver_L final Cost: {f_final_solver_L}")

f_initial_LR_L, time_initial_LR_L, x_initial_i_j_LR_L, x_initial_i_cloud_LR_L = LR_EUA_initial_allocation(U, S, c_i_j, c_i_cloud, L_j, L_i_j, cov_laplace, omega, d_laplace, r)
fu_final_LR_L,fp_final_LR_L,fc_final_LR_L,f_final_LR_L, time_final_LR_L, x_final_i_j_LR_L, x_final_i_cloud_LR_L,f_true_initial_LR_L = reallocate_failed_users(U, S, L_j, L_i_j, c_i_j, c_i_cloud, x_initial_i_j_LR_L, x_initial_i_cloud_LR_L, cov_initial, r, omega, d_laplace)
print(f"LR_L final Cost: {f_final_LR_L}")

f_initial_ga_L, time_initial_ga_L, x_initial_i_j_ga_L, x_initial_i_cloud_ga_L = GA_initial_allocation(U, S, c_i_j, c_i_cloud, L_j, L_i_j, cov_laplace, omega, d_laplace, r)
fu_final_ga_L,fp_final_ga_L,fc_final_ga_L,f_final_ga_L, time_final_ga_L, x_final_i_j_ga_L, x_final_i_cloud_ga_L,f_true_initial_ga_L = reallocate_failed_users(U, S, L_j, L_i_j, c_i_j, c_i_cloud, x_initial_i_j_ga_L, x_initial_i_cloud_ga_L, cov_initial, r, omega, d_laplace)
print(f"GA_L final Cost: {f_final_ga_L}")

f_initial_solver_G, time_initial_solver_G, x_initial_i_j_solver_G, x_initial_i_cloud_solver_G = Solver_initial_allocation(U, S, c_i_j, c_i_cloud, L_j, L_i_j, cov_gaussian, r, omega, d_gaussian)
fu_final_solver_G,fp_final_solver_G,fc_final_solver_G,f_final_solver_G, time_final_solver_G, x_final_i_j_solver_G, x_final_i_cloud_solver_G,f_true_initial_solver_G = reallocate_failed_users(U, S, L_j, L_i_j, c_i_j, c_i_cloud, x_initial_i_j_solver_G, x_initial_i_cloud_solver_G, cov_initial, r, omega, d_gaussian)
print(f"Solver_G final Cost: {f_final_solver_G}")

f_initial_LR_G, time_initial_LR_G, x_initial_i_j_LR_G, x_initial_i_cloud_LR_G = LR_EUA_initial_allocation(U, S, c_i_j, c_i_cloud, L_j, L_i_j, cov_gaussian, omega, d_gaussian, r)
fu_final_LR_G,fp_final_LR_G,fc_final_LR_G,f_final_LR_G, time_final_LR_G, x_final_i_j_LR_G, x_final_i_cloud_LR_G,f_true_initial_LR_G = reallocate_failed_users(U, S, L_j, L_i_j, c_i_j, c_i_cloud, x_initial_i_j_LR_G, x_initial_i_cloud_LR_G, cov_initial, r, omega, d_gaussian)
print(f"LR_G final Cost: {f_final_LR_G}")

f_initial_ga_G, time_initial_ga_G, x_initial_i_j_ga_G, x_initial_i_cloud_ga_G = GA_initial_allocation(U, S, c_i_j, c_i_cloud, L_j, L_i_j, cov_gaussian, omega, d_gaussian, r)
fu_final_ga_G,fp_final_ga_G,fc_final_ga_G,f_final_ga_G, time_final_ga_G, x_final_i_j_ga_G, x_final_i_cloud_ga_G,f_true_initial_ga_G = reallocate_failed_users(U, S, L_j, L_i_j, c_i_j, c_i_cloud, x_initial_i_j_ga_G, x_initial_i_cloud_ga_G, cov_initial, r, omega, d_gaussian)
print(f"GA_G final Cost: {f_final_ga_G}")
