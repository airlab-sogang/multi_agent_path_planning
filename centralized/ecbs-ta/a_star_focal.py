"""

AStar search

author: Ashwin Bose (@atb033)

"""


class AStarEpsilon:
    def __init__(self, env, w):
        self.w = w
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.focal_vertex_heuristic = env.focal_vertex_heuristic
        self.focal_edge_heuristic = env.focal_edge_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name):
        """
        low level search
        """
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1

        closed_set = set()
        open_set = {initial_state}
        focal_set = {initial_state}

        came_from = {}

        g_score = {}
        g_score[initial_state] = 0

        # f_score is the sum of g_score and h_score
        f_score = {}
        # d_score is the number of conflicts
        d_score = {}

        f_score[initial_state] = self.admissible_heuristic(initial_state, agent_name)
        d_score[initial_state] = self.focal_vertex_heuristic(initial_state)
        min_f_score = f_score[initial_state]

        while open_set:
            # update FOCAL if min_f_score has increased
            new_min_f_score = min([f_score[open_node] for open_node in open_set])
            if min_f_score < new_min_f_score:
                focal_set.clear()
                for open_node in open_set:
                    if f_score[open_node] <= self.w * new_min_f_score:
                        focal_set |= {open_node}
                min_f_score = new_min_f_score

            # select current node from FOCAL
            temp_dict = {
                focal_item: (
                    d_score.setdefault(focal_item, float("inf")),
                    f_score.setdefault(focal_item, float("inf")),
                )
                for focal_item in focal_set
            }
            # get first item with minimum d_score, then minimum f_score
            current = min(temp_dict, key=lambda x: (temp_dict[x][0], temp_dict[x][1]))

            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current), min_f_score

            open_set -= {current}
            focal_set -= {current}
            closed_set |= {current}

            neighbor_list = self.get_neighbors(current)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue

                tentative_g_score = (
                    g_score.setdefault(current, float("inf")) + step_cost
                )

                if neighbor not in open_set:
                    open_set |= {neighbor}
                    # add to FOCAL if f score is lower than w * min_f_score
                    if (
                        tentative_g_score
                        + self.admissible_heuristic(neighbor, agent_name)
                        <= self.w * min_f_score
                    ):
                        focal_set |= {neighbor}
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(
                    neighbor, agent_name
                )
                d_score[neighbor] = (
                    d_score[current]
                    + self.focal_vertex_heuristic(neighbor)
                    + self.focal_edge_heuristic(current, neighbor)
                )
        return False, min_f_score
