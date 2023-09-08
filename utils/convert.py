import random

import yaml

basename = "house"

with open(f"{basename}.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    obstacle_list = data["obstacle_coord_list"]
    position_list = data["position_coord_list"]

dimensions = [32, 32]
starts = []
goals = []

num_of_agents = 50
count = 1
for j in range(count):
    # select 100 instances from position_list randomly but do not duplicate
    for i in range(num_of_agents):
        print(i)
        while True:
            start = random.choice(position_list)
            goal = random.choice(position_list)
            if start not in starts and goal not in goals:
                starts.append(start)
                goals.append(goal)
                break

    # change obstacles to a list of tuples
    obstacles = []
    for i in range(len(obstacle_list)):
        obstacles.append((obstacle_list[i][0], obstacle_list[i][1]))
    output = {
        "starts": starts,
        "goals": goals,
        "map": {"dimensions": dimensions, "obstacles": obstacles},
    }
    with open(f"{basename}_input_{j}.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False)
