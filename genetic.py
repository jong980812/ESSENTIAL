import numpy as np
import os
import os
import numpy as np
import torch, random, json, utils
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
import copy
import itertools
from datasets_cil import build_dataset
def create_initial_population(size, num_frames):
    return [np.random.permutation(num_frames) for _ in range(size)]



# def select_parents(population, fitnesses, num_parents):
#     """
#     이 함수는 주어진 적합도에 따라 부모를 선택합니다.
#     룰렛 휠 선택 방식을 사용하여, 높은 적합도를 가진 개체가 부모로 선택될 확률이 높습니다.
    
#     :param population: 전체 개체군, 각 개체는 프레임 순서 배열을 나타냄
#     :param fitnesses: 각 개체의 적합도, 1차원 배열
#     :param num_parents: 선택할 부모의 수
#     :return: 선택된 부모 개체의 리스트
#     """
#     # 적합도에 음수가 있거나 합이 0인 경우를 처리
#     if np.any(fitnesses < 0) or np.sum(fitnesses) == 0:
#         fitnesses = fitnesses - np.min(fitnesses) + 0.0001  # 적합도 최소값을 0 이상으로 조정하고, 모든 값에 작은 양수를 더함

#     # 적합도에 비례하여 부모의 인덱스를 선택
#     parents_indices = np.random.choice(len(population), size=num_parents, replace=False, p=fitnesses/np.sum(fitnesses))
#     # 인덱스를 사용하여 부모 개체를 선택
#     parents = [population[index] for index in parents_indices]
#     return parents

# def crossover(parent1, parent2):
#     idx = np.random.randint(1, len(parent1))
#     first_part = parent1[:idx]
#     second_part = [x for x in parent2 if x not in first_part]
#     new_order = np.concatenate([first_part, second_part])
#     return new_order


# def mutate(order, mutation_rate=0.01):
#     if np.random.rand() < mutation_rate:
#         idx1, idx2 = np.random.randint(0, len(order), 2)
#         order[idx1], order[idx2] = order[idx2], order[idx1]
#     return order

# def genetic_algorithm(dataest,path, label,model, num_generations, population_size, num_parents,args):
#     frames,_ = dataest.get_video(path,label)
#     frames = frames.permute(1,0,2,3).to(args.device)
#     model.eval()
#     population = create_initial_population(population_size, 8)
#     for _ in range(num_generations):
#         fitnesses = np.array([calculate_fitness(model,p,frames).squeeze(0)[label].cpu() for p in population])
#         parents = select_parents(population, fitnesses, num_parents)
#         new_population = []
#         for i in range(0, num_parents, 2):
#             for _ in range(population_size // num_parents):
#                 child = crossover(parents[i], parents[i+1])
#                 child = mutate(child)
#                 new_population.append(child)
#         population = new_population
#     return max(population, key=lambda x: calculate_fitness(model,x,frames))

# 비디오 프레임과 모델, 세대 수, 인구 크기, 부모 수 설정)


def select_parents(population, fitnesses, num_parents):
    fitnesses = np.maximum(fitnesses, 0)  # 음수 적합도 처리
    if np.sum(fitnesses) == 0:
        fitnesses += 1.0 / len(fitnesses)  # 모든 적합도가 0인 경우 처리
    parents_indices = np.random.choice(len(population), size=num_parents, replace=False, p=fitnesses/np.sum(fitnesses))
    parents = [population[index] for index in parents_indices]
    return parents

# 교차 함수
def crossover(parent1, parent2):
    idx = np.random.randint(1, len(parent1))
    first_part = parent1[:idx]
    second_part = [x for x in parent2 if x not in first_part]
    return np.concatenate([first_part, second_part])

# 돌연변이 함수
def mutate(order, mutation_rate=0.01):
    for _ in range(int(len(order) * mutation_rate)):
        idx1, idx2 = np.random.randint(0, len(order), 2)
        order[idx1], order[idx2] = order[idx2], order[idx1]
    return order

# 유전 알고리즘
@torch.no_grad()
def genetic_algorithm(dataest,path, label,model, num_generations=100, population_size=50, num_parents=10, mutation_rate=0.01,args=None):
    frames,_ = dataest.get_video(path,label)
    with torch.cuda.amp.autocast():original = model(frames.unsqueeze(0).to(args.device))
    print(original[0,].argmax())
    frames = frames.permute(1,0,2,3).to(args.device)
    model.eval()
    population = generate_all_combinations()#create_initial_population(population_size, 8)
    best_order = None
    best_fitness = -np.inf

    for generation in range(num_generations):
        # fitnesses = np.array([calculate_fitness(individual, model, video_data) for individual in population])
        fitnesses = get_fitnesses(model,frames,population,label)
        top_10_indices = np.argpartition(fitnesses, -10)[-10:]
        # 상위 10개 값 중에서 다시 정렬
        top_10_indices = top_10_indices[np.argsort(fitnesses[top_10_indices])[::-1]]

        # 상위 10개 인덱스의 fitnesses 값
        top_10_fitnesses = fitnesses[top_10_indices]

        # 상위 10개 인덱스에 해당하는 population의 값
        top_10_population = population[top_10_indices]

        print("Top 10 indices:", top_10_indices)
        print("Top 10 fitnesses:", top_10_fitnesses)
        print("Top 10 population values:\n", top_10_population)
    #     parents = select_parents(population, fitnesses, num_parents)
    #     new_population = []
    #     num_children_per_pair = population_size // (num_parents // 2)
    #     for i in range(0, num_parents, 2):
    #         for _ in range(num_children_per_pair):
    #             if len(new_population) < population_size:
    #                 child = crossover(parents[i], parents[i+1])
    #                 child = mutate(child, mutation_rate)
    #                 new_population.append(child)
    #     population = new_population
    #     # Update best solution found
    #     current_best = population[np.argmax(fitnesses)]
    #     current_best_fitness = np.max(fitnesses)
    #     if current_best_fitness > best_fitness:
    #         best_fitness = current_best_fitness
    #         best_order = current_best
    # return best_order, best_fitness


@torch.no_grad()
def get_fitnesses(model,frames,population,label):
    with torch.cuda.amp.autocast():
        batch = []
        fitnesses = []
        for i,p in enumerate(population):
            batch.append(frames[p].permute(1,0,2,3).unsqueeze(0))
            if (i+1) % 32 ==0:
                batch = torch.cat(batch,0)
                fitnesses.append((model(batch)[:,label].cpu()))
                batch = []

        fitnesses = np.array(torch.cat(fitnesses,0))
    return fitnesses
    # np.array([calculate_fitness(model,p,frames).squeeze(0)[label].cpu() for p in population])
@torch.no_grad()
def calculate_fitness(model,order,frames):
    # 예: 특정 순서에서 모델의 예측 확률을 반환하는 함수
    with torch.cuda.amp.autocast():
        shuffeld_frames = frames[order].permute(1,0,2,3).unsqueeze(0)
        output = model(shuffeld_frames)
    return output

def generate_all_combinations():
    # 쌍을 이루는 그룹 정의
    groups = [
        [(0, 1), (1, 0)], 
        [(2, 3), (3, 2)], 
        [(4, 5), (5, 4)], 
        [(6, 7), (7, 6)]
    ]
    
    # 모든 그룹의 가능한 조합 생성
    group_combinations = list(itertools.product(*groups))
    
    # 조합들의 모든 가능한 순열 생성
    all_permutations = []
    for combo in group_combinations:
        all_permutations.extend(itertools.permutations(combo))
    
    # 최종 리스트 저장용 배열
    all_orders = np.array([np.concatenate(perm) for perm in all_permutations])
    np.random.shuffle(all_orders)
    return all_orders