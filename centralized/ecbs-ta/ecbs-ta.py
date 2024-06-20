"""

Python implementation of Conflict-based search

author: Ashwin Bose (@atb033)

"""
import os
import sys
import networkx as nx
import numpy as np
import time

sys.path.insert(0, "../")
import argparse
import yaml
from math import fabs
from itertools import combinations
from copy import deepcopy
from tqdm import tqdm
from hungarian import hungarian, constrained_hungarian
from a_star_focal import AStarEpsilon


class Location(object):
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return str((self.x, self.y))


class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time) + str(self.location.x) + str(self.location.y))

    def is_equal_except_time(self, state):
        return self.location == state.location

    def __str__(self):
        return str((self.time, self.location.x, self.location.y))


class Conflict(object):
    VERTEX = 1
    EDGE = 2

    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ""
        self.agent_2 = ""

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return (
            "("
            + str(self.time)
            + ", "
            + self.agent_1
            + ", "
            + self.agent_2
            + ", "
            + str(self.location_1)
            + ", "
            + str(self.location_2)
            + ")"
        )


class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time) + str(self.location))

    def __str__(self):
        return "(" + str(self.time) + ", " + str(self.location) + ")"


class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2

    def __eq__(self, other):
        return (
            self.time == other.time
            and self.location_1 == other.location_1
            and self.location_2 == other.location_2
        )

    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))

    def __str__(self):
        return (
            "("
            + str(self.time)
            + ", "
            + str(self.location_1)
            + ", "
            + str(self.location_2)
            + ")"
        )


class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return (
            "VC: "
            + str([str(vc) for vc in self.vertex_constraints])
            + "EC: "
            + str([str(ec) for ec in self.edge_constraints])
        )


class Environment(object):
    def __init__(self, dimension, starts, goals, agents, obstacles, w):
        self.dimension = dimension
        self.obstacles = obstacles
        self.w = w

        self.starts = starts
        self.goals = goals
        self.cost_matrix = self.generate_cost_matrix(
            self.dimension, self.obstacles, self.starts, self.goals
        )

        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.reservation_table = {}

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star_epsilon = AStarEpsilon(self, self.w)

    def generate_cost_matrix(self, dimension, obstacles, starts, goals):
        valid_grid_list = []
        dim_x, dim_y = dimension

        # Initialize graph
        G = nx.Graph()

        # Add nodes
        for i in range(dim_x):
            for j in range(dim_y):
                if (i, j) not in obstacles:
                    G.add_node(len(valid_grid_list), pos=(i, j))
                    valid_grid_list.append((i, j))

        # Add edges
        for i in range(len(valid_grid_list)):
            for j in range(i + 1, len(valid_grid_list)):
                x1, y1 = valid_grid_list[i]
                x2, y2 = valid_grid_list[j]
                if abs(x1 - x2) + abs(y1 - y2) == 1:
                    G.add_edge(i, j, weight=1)

        # Initialize list to store the cost lists for each robot
        all_cost_lists = []

        for sp in starts:
            # Convert start position to corresponding node index
            start_node = valid_grid_list.index(tuple(sp))

            # Calculate shortest path lengths from start node to all other nodes
            length_dict = nx.single_source_shortest_path_length(G, start_node)

            # Initialize cost list for this start position
            cost_list = []

            for idx, gp in enumerate(goals):
                # Convert goal position to corresponding node index
                goal_node = valid_grid_list.index(tuple(gp))

                # Get the shortest path length to this goal
                if goal_node in length_dict:
                    cost_list.append(length_dict[goal_node])
                else:
                    cost_list.append(float("inf"))  # If the goal is not reachable

            # Append the cost list to the list
            all_cost_lists.append(cost_list)
        all_cost_lists = np.array(all_cost_lists, dtype=np.float64)
        return all_cost_lists

    def get_neighbors(self, state):
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y + 1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y - 1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x - 1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x + 1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        return neighbors

    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t + 1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t + 1)

                if state_1a.is_equal_except_time(
                    state_2b
                ) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False

    def focal_heuristic(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        num_of_conflicts = 0
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    num_of_conflicts += 1

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t + 1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t + 1)

                if state_1a.is_equal_except_time(
                    state_2b
                ) and state_1b.is_equal_except_time(state_2a):
                    num_of_conflicts += 1
        return num_of_conflicts

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(
                conflict.time, conflict.location_1, conflict.location_2
            )
            e_constraint2 = EdgeConstraint(
                conflict.time, conflict.location_2, conflict.location_1
            )

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        return (
            state.location.x >= 0
            and state.location.x < self.dimension[0]
            and state.location.y >= 0
            and state.location.y < self.dimension[1]
            and VertexConstraint(state.time, state.location)
            not in self.constraints.vertex_constraints
            and (state.location.x, state.location.y) not in self.obstacles
        )

    def transition_valid(self, state_1, state_2):
        return (
            EdgeConstraint(state_1.time, state_1.location, state_2.location)
            not in self.constraints.edge_constraints
        )

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        goal = self.agent_dict[agent_name]["goal"]
        return fabs(state.location.x - goal.location.x) + fabs(
            state.location.y - goal.location.y
        )

    def focal_vertex_heuristic(self, state):
        num_of_conflicts = 0
        for agent_name in self.reservation_table.keys():
            agent_location = self.reservation_table[agent_name].get(state.time, None)
            if agent_location and agent_location == state.location:
                num_of_conflicts += 1
        return num_of_conflicts

    def focal_edge_heuristic(self, prev_state, next_state):
        num_of_conflicts = 0
        for agent_name in self.reservation_table.keys():
            agent_prev_location = self.reservation_table[agent_name].get(
                prev_state.time, None
            )
            agent_next_location = self.reservation_table[agent_name].get(
                next_state.time, None
            )
            if agent_prev_location and agent_next_location:
                if (
                    agent_prev_location == next_state.location
                    and agent_next_location == prev_state.location
                ):
                    num_of_conflicts += 1
        return num_of_conflicts

    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(0, Location(agent["start"][0], agent["start"][1]))
            goal_state = State(0, Location(agent["goal"][0], agent["goal"][1]))

            self.agent_dict.update(
                {agent["name"]: {"start": start_state, "goal": goal_state}}
            )

    def update_agent_dict(self, agent_name, start_point, goal_point):
        start_state = State(0, Location(start_point[0], start_point[1]))
        goal_state = State(0, Location(goal_point[0], goal_point[1]))

        self.agent_dict.update({agent_name: {"start": start_state, "goal": goal_state}})

    def compute_path(self, agent_id):
        self.constraints = self.constraint_dict.setdefault(agent_id, Constraints())
        path, min_f_score = self.a_star_epsilon.search(agent_id)
        if not path:
            return False, 0
        return path, min_f_score

    def compute_solution_cost(self, solution):
        return sum([len(path) - 1 for path in solution.values()])


class HighLevelNode(object):
    def __init__(self):
        self.root = False
        self.solution = {}
        self.constraint_dict = {}
        self.min_f_scores = {}
        self.assignment = []
        self.reservation_table = {}
        self.cost = 0
        self.lb = 0
        self.focal_heuristic = 0
        self.tree_id = -1

    def __hash__(self):
        return hash(str(self.solution) + str(self.assignment) + str(self.cost))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.solution == other.solution
            and self.cost == other.cost
            and self.assignment == other.assignment
        )

    def __lt__(self, other):
        # first compare the focal heuristic
        if self.focal_heuristic != other.focal_heuristic:
            return self.focal_heuristic < other.focal_heuristic
        return self.cost < other.cost


class ASGNode(object):
    def __init__(self):
        self.includedAssignments = set()
        self.excludedAssignments = set()
        self.assignment = []
        self.cost = 0

    def __hash__(self):
        return hash(
            str(self.includedAssignments)
            + str(self.excludedAssignments)
            + str(self.assignment)
            + str(self.cost)
        )

    def __lt__(self, other):
        return self.cost < other.cost


class ECBS(object):
    def __init__(self, environment):
        self.env = environment
        self.open_set = set()
        self.focal_set = set()
        self.asg_open_set = set()
        self.high_level_split_list = list()

    def search(self):
        start = HighLevelNode()
        start.constraint_dict = {}
        for agent_id in self.env.agent_dict.keys():
            start.constraint_dict[agent_id] = Constraints()

        # first assignment
        start_asg = ASGNode()
        start_asg.includedAssignments = set()
        start_asg.excludedAssignments = set()
        start_asg.assignment, start_asg.cost = hungarian(self.env.cost_matrix)
        self.asg_open_set |= {start_asg}

        start.assignment = start_asg.assignment
        # update agent_dict
        for i in range(len(self.env.agents)):
            self.env.update_agent_dict(
                self.env.agents[i]["name"],
                self.env.starts[start.assignment[i][0]],
                self.env.goals[start.assignment[i][1]],
            )
        start.root = True
        for agent_id in self.env.agent_dict.keys():
            self.env.reservation_table = start.reservation_table
            path, min_f_score = self.env.compute_path(agent_id)
            if not path:
                return {}
            start.solution.update({agent_id: path})
            start.min_f_scores.update({agent_id: min_f_score})
            start.reservation_table.update(
                {agent_id: {state.time: state.location for state in path}}
            )
        self.high_level_split_list.append(0)
        start.tree_id = 0
        start.lb = sum(start.min_f_scores.values())
        start.cost = self.env.compute_solution_cost(start.solution)
        start.focal_heuristic = self.env.focal_heuristic(start.solution)

        self.open_set |= {start}
        self.focal_set |= {start}
        min_lb = start.lb

        iter = 0
        while self.open_set:
            iter += 1
            # update FoCal if min_f_score has increased
            new_min_lb = min([CTNode.lb for CTNode in self.open_set])
            if min_lb < new_min_lb:
                self.focal_set.clear()
                for CTNode in self.open_set:
                    if CTNode.cost <= self.env.w * new_min_lb:
                        self.focal_set |= {CTNode}
                min_lb = new_min_lb

            P = min(self.focal_set)
            self.open_set -= {P}
            self.focal_set -= {P}
            conflict_dict = self.env.get_first_conflict(P.solution)
            if not conflict_dict:
                print(*self.high_level_split_list, sep=", ")
                print("Tree ID: ", P.tree_id)
                return (
                    self.high_level_split_list[P.tree_id],
                    sum([len(path) - 1 for path in P.solution.values()]),
                    max([len(path) - 1 for path in P.solution.values()]),
                    self.generate_plan(P.solution),
                )

            if P.root:
                new_node = HighLevelNode()
                new_node.constraint_dict = {}
                for agent_id in self.env.agent_dict.keys():
                    new_node.constraint_dict[agent_id] = Constraints()
                new_node.assignment = self.next_assignment()
                self.env.constraint_dict = new_node.constraint_dict
                # update agent_dict
                for i in range(len(self.env.agents)):
                    self.env.update_agent_dict(
                        self.env.agents[i]["name"],
                        self.env.starts[new_node.assignment[i][0]],
                        self.env.goals[new_node.assignment[i][1]],
                    )
                new_node.root = True
                continue_flag = True
                # find solution for new node
                for agent_id in self.env.agent_dict.keys():
                    self.env.reservation_table = new_node.reservation_table
                    path, min_f_score = self.env.compute_path(agent_id)
                    if not path:
                        continue_flag = False
                        break
                    new_node.solution.update({agent_id: path})
                    new_node.min_f_scores.update({agent_id: min_f_score})
                    new_node.reservation_table.update(
                        {agent_id: {state.time: state.location for state in path}}
                    )
                if continue_flag:
                    self.high_level_split_list.append(0)
                    new_node.tree_id = len(self.high_level_split_list) - 1
                    new_node.lb = sum(new_node.min_f_scores.values())
                    new_node.cost = self.env.compute_solution_cost(new_node.solution)
                    new_node.focal_heuristic = self.env.focal_heuristic(
                        new_node.solution
                    )
                    self.open_set |= {new_node}
                    if new_node.cost <= self.env.w * min_lb:
                        self.focal_set |= {new_node}

            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)
            self.high_level_split_list[P.tree_id] += 1
            for agent_id in constraint_dict.keys():
                if conflict_dict.time >= len(P.solution[agent_id]):
                    continue

                new_node = deepcopy(P)
                new_node.constraint_dict[agent_id].add_constraint(
                    constraint_dict[agent_id]
                )
                # update agent_dict
                for i in range(len(self.env.agents)):
                    self.env.update_agent_dict(
                        self.env.agents[i]["name"],
                        self.env.starts[new_node.assignment[i][0]],
                        self.env.goals[new_node.assignment[i][1]],
                    )
                new_node.root = False
                new_node.reservation_table[agent_id].clear()
                self.env.reservation_table = new_node.reservation_table
                self.env.constraint_dict = new_node.constraint_dict
                path, min_f_score = self.env.compute_path(agent_id)
                if not path:
                    continue
                new_node.solution.update({agent_id: path})
                new_node.min_f_scores.update({agent_id: min_f_score})
                new_node.reservation_table.update(
                    {agent_id: {state.time: state.location for state in path}}
                )
                new_node.lb = sum(new_node.min_f_scores.values())
                new_node.cost = self.env.compute_solution_cost(new_node.solution)
                self.open_set |= {new_node}
                if new_node.cost <= self.env.w * min_lb:
                    self.focal_set |= {new_node}

        return {}

    def next_assignment(self):
        P = min(self.asg_open_set)
        self.asg_open_set -= {P}

        for i in range(len(self.env.agents)):
            continue_flag = False
            for assignment in P.includedAssignments:
                if assignment[0] == i:
                    continue_flag = True
                    break
            if continue_flag:
                continue

            new_asg_node = deepcopy(P)
            new_asg_node.excludedAssignments |= {P.assignment[i]}
            for j in range(i):
                new_asg_node.includedAssignments |= {P.assignment[j]}
            new_asg_node.assignment, new_asg_node.cost = constrained_hungarian(
                self.env.cost_matrix,
                new_asg_node.includedAssignments,
                new_asg_node.excludedAssignments,
            )

            self.asg_open_set |= {new_asg_node}
        best_node = min(self.asg_open_set)
        return best_node.assignment

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [
                {"t": state.time, "x": state.location.x, "y": state.location.y}
                for state in path
            ]
            plan[agent] = path_dict_list
        return plan


def main():
    # Read from input file
    parser = argparse.ArgumentParser()
    parser.add_argument("map_name", type=str, help="Input filename")
    parser.add_argument("robot_num", type=int, help="Input filename")
    parser.add_argument("count", type=int, help="Input filename")
    parser.add_argument("separate_flag", type=int, help="Input filename")
    args = parser.parse_args()
    map_name = args.map_name
    robot_num = args.robot_num
    count = args.count
    separate_flag = args.separate_flag

    basename = f"mini_{map_name}_{robot_num}"
    basefolder = os.path.dirname(os.path.abspath(__file__)) + "/../../"
    input_folder = f"{basefolder}ECBS-TA-testset/YAML_{basename}/"
    modified_input_folder = f"{basefolder}ECBS-TA-mtestset/YAML_{basename}/"
    output_folder = f"{basefolder}ECBS-TA-testresult/YAML_{basename}/"

    if separate_flag == 1:
        basename = f"mini_{map_name}_Separate_{robot_num}"
        basefolder = os.path.dirname(os.path.abspath(__file__)) + "/../../"
        input_folder = f"{basefolder}ECBS-TA-testset/YAML_{basename}/"
        modified_input_folder = f"{basefolder}ECBS-TA-mtestset/YAML_{basename}/"
        output_folder = f"{basefolder}ECBS-TA-testresult/YAML_{basename}/"

    input_filename = input_folder + f"{basename}_{count}.yaml"
    modified_input_filename = (
            modified_input_folder + f"m_{basename}_{count}.yaml"
    )
    output_filename = output_folder + f"{basename}_{count}_output.yaml"
    with open(
            input_filename,
            "r",
    ) as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    starts = param["starts"]
    goals = param["goals"]
    agents = [
        dict(name=f"agent{i}", start=(0, 0), goal=(0, 0))
        for i in range(len(starts))
    ]
    env = Environment(dimension, starts, goals, agents, obstacles, 1.1)
    start_time = time.time()
    try:
        ecbs = ECBS(env)
        num_of_split, sum_of_cost, makespan, solution = ecbs.search()
    except Exception as e:
        print("Error occured: ", e)
        return

    computation_time = time.time() - start_time
    if not solution:
        print(" Solution not found")
        return
    print("Number of split: ", num_of_split)
    # Write to modified input file
    with open(modified_input_filename, "w") as param_file:
        param["agents"] = []
        for agent in env.agent_dict:
            param["agents"].append(
                {
                    "name": agent,
                    "start": [
                        env.agent_dict[agent]["start"].location.x,
                        env.agent_dict[agent]["start"].location.y,
                    ],
                    "goal": [
                        env.agent_dict[agent]["goal"].location.x,
                        env.agent_dict[agent]["goal"].location.y,
                    ],
                }
            )
        # remove starts
        param.pop("starts", None)
        # remove goals
        param.pop("goals", None)
        yaml.dump(param, param_file)

    # Write to output file
    output = dict()
    output["schedule"] = solution
    output["sum_of_cost"] = sum_of_cost
    output["makespan"] = makespan
    output["time"] = computation_time
    output["num_of_split"] = num_of_split
    with open(output_filename, "w") as output_yaml:
        yaml.safe_dump(output, output_yaml)


if __name__ == "__main__":
    main()
