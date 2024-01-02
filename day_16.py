import numpy as np
import sys

from notebooks.aoc_2023.utils import load_stripped_lines, Grid, COORDINATE

sys.setrecursionlimit(100000)
MOVES = {
    'left': (0, -1),
    'right': (0, +1),
    'up': (-1, 0),
    'down': (1, 0)
}
DIRECTION_MAP = {
    (0, -1): 'left',
    (0, 1): 'right',
    (-1, 0): 'up',
    (1, 0): 'down'
}


def get_direction(current_point, previvious):
    return current_point[0] - previvious[0], current_point[1] - previvious[1]


def make_move(current_point, direction):
    return current_point[0] + direction[0], current_point[1] + direction[1]


def get_mirror_direction(direction, mirror):
    if DIRECTION_MAP[direction] == 'right' and mirror == '/':
        new_direction = MOVES['up']
    elif DIRECTION_MAP[direction] == 'right' and mirror == '\\':
        new_direction = MOVES['down']
    elif DIRECTION_MAP[direction] == 'left' and mirror == '/':
        new_direction = MOVES['down']
    elif DIRECTION_MAP[direction] == 'left' and mirror == '\\':
        new_direction = MOVES['up']
    elif DIRECTION_MAP[direction] == 'down' and mirror == '/':
        new_direction = MOVES['left']
    elif DIRECTION_MAP[direction] == 'down' and mirror == '\\':
        new_direction = MOVES['right']
    elif DIRECTION_MAP[direction] == 'up' and mirror == '/':
        new_direction = MOVES['right']
    elif DIRECTION_MAP[direction] == 'up' and mirror == '\\':
        new_direction = MOVES['left']
    else:
        raise NotImplemented
    return new_direction


def is_loop(point, _trajectory) -> bool:
    if len(_trajectory) < 3:
        return False

    for p_1, p_2, p_3 in zip(_trajectory[:-3], _trajectory[1:-3], _trajectory[2:-3]):
        if p_3 == _trajectory[-1] and p_2 == _trajectory[-2] and p_1 == _trajectory[-3]:
            return True
    return False


class Beam:
    def __init__(self):
        self.vertices: set[COORDINATE] = set()
        self.edges: set[tuple[COORDINATE, COORDINATE]] = set()

    def update(self, trajectory: list[COORDINATE]) -> None:
        for p_1, p_2 in zip(trajectory, trajectory[1:]):
            self.edges.add((p_1, p_2))
        self.vertices.update(trajectory)

    def __len__(self) -> int:
        return len(self.vertices)

    def clear(self) -> None:
        self.vertices.clear()
        self.edges.clear()


BEAM = Beam()


def trajectory_already_found(_trajectory) -> bool:
    # for found_trajectory in FOUND_TRAJECTORIES:
    #     for p_1, p_2 in zip(found_trajectory[:], found_trajectory[1:]):
    #         if p_2 == _trajectory[-1] and p_1 == _trajectory[-2]:
    #             return True
    # return False
    return (_trajectory[-2], _trajectory[-1]) in BEAM.edges


FOUND_TRAJECTORIES = []
_VISITED = Grid([['.' for _ in range(110)] for _ in range(110)])
# _VISITED = Grid([['.' for _ in range(10)] for _ in range(10)])


def update_visited_grid(visited_grid, _tra):
    for coordinate in _tra:
        if visited_grid.is_inside_grid(coordinate):
            visited_grid[coordinate] = '#'


def nb_visited(_tra):
    for coordinate in _tra:
        if _VISITED.is_inside_grid(coordinate):
            _VISITED[coordinate] = '#'

    visited = _VISITED.get_coordinates('#')
    return len(visited)


def find_beam_trajectories(grid: Grid, trajectory: list[COORDINATE], visited_grid: Grid) -> bool:

        current_point = trajectory[-1]
        if not grid.is_inside_grid(current_point):
            #FOUND_TRAJECTORIES.append(trajectory)
            BEAM.update(trajectory)
            update_visited_grid(visited_grid, trajectory)
            # print(f'new_trajectory found, len: {len(trajectory)}, visited: {len(visited_grid.get_coordinates("#"))}')
            return True
            # vis = np.zeros(shape=(110, 110))
            # for coor in FOUND_TRAJECTORIES[-1]:
            #     vis[coor] = 255

        if is_loop(current_point, trajectory):
            #FOUND_TRAJECTORIES.append(trajectory)
            BEAM.update(trajectory)
            update_visited_grid(visited_grid, trajectory)
            # print(f'loop trajectory found, len: {len(trajectory)}, visited: {len(visited_grid.get_coordinates("#"))}')
            return True

        if len(trajectory) > 110 and trajectory_already_found(trajectory):
            # FOUND_TRAJECTORIES.append(trajectory)
            BEAM.update(trajectory)
            update_visited_grid(visited_grid, trajectory)
            # print(f'stopping computation, already on visited path; len: {len(trajectory)}, visited: {len(visited_grid.get_coordinates("#"))}')
            return True

        previous_point = trajectory[-2]
        direction = get_direction(current_point, previous_point)
        symbol_on_grid = grid[current_point]

        if symbol_on_grid == '.':
            # continue in the same direction
            next_point = make_move(current_point, direction)
            # trajectory.append(next_point)
            find_beam_trajectories(grid, trajectory + [next_point], visited_grid)

        elif symbol_on_grid in ('/', '\\'):
            next_direction = get_mirror_direction(direction, symbol_on_grid)
            next_point = make_move(current_point, next_direction)
            # trajectory.append(next_point)
            find_beam_trajectories(grid, trajectory + [next_point], visited_grid)

        elif symbol_on_grid in ('|', '-'):
            if (DIRECTION_MAP[direction] in ('left', 'right') and symbol_on_grid == '-') or (DIRECTION_MAP[direction] in ('up', 'down') and symbol_on_grid == '|'):
                # continue in the same direction
                next_point = make_move(current_point, direction)
                # trajectory.append(next_point)
                find_beam_trajectories(grid, trajectory + [next_point], visited_grid)
            else:
                # beam splitted:
                if symbol_on_grid == '-':
                    for new_direction in (MOVES['left'], MOVES['right']):
                        next_point = make_move(current_point, new_direction)
                        find_beam_trajectories(grid, trajectory + [next_point], visited_grid)

                elif symbol_on_grid == '|':
                    for new_direction in (MOVES['up'], MOVES['down']):
                        next_point = make_move(current_point, new_direction)
                        find_beam_trajectories(grid, trajectory + [next_point], visited_grid)
                else:
                    raise NotImplemented()
        else:
            raise NotImplemented()


def solve_grid(grid, starting_points) -> int:
    _visited_grid = Grid([['.' for _ in range(grid.width)] for _ in range(grid.height)])
    BEAM.clear()
    assert BEAM.edges == BEAM.vertices == set()
    a = find_beam_trajectories(grid, starting_points, _visited_grid)
    return len(_visited_grid.get_coordinates('#'))


def solve_2(grid: Grid) -> int:
    resulting_values = []

    # from right
    for j in range(_grid.height):
        starting_points = [(j, -1), (j, 0)]
        total_visited = solve_grid(grid, starting_points)
        resulting_values.append(total_visited)
        print(f'for {starting_points}: {total_visited}, maximum is {max(resulting_values)}')

    # from top
    for i in range(_grid.width):
        starting_points = [(-1, i), (0, i)]
        total_visited = solve_grid(grid, starting_points)
        resulting_values.append(total_visited)
        print(f'for {starting_points}: {total_visited}, maximum is {max(resulting_values)}')

    # from left
    for j in range(_grid.height):
        starting_points = [(j, _grid.width), (j, _grid.width-1)]
        total_visited = solve_grid(grid, starting_points)
        resulting_values.append(total_visited)
        print(f'for {starting_points}: {total_visited}, maximum is {max(resulting_values)}')

    # from bottom
    for i in range(_grid.width):
        starting_points = [(i, _grid.height), (i, _grid.height - 1)]
        total_visited = solve_grid(grid, starting_points)
        resulting_values.append(total_visited)

    return max(resulting_values)


if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/16.txt')
    _grid = Grid([list(item) for item in raw_data])

    # _test_trajectory = [(0, 0), (0, 1)]
    _solve_trajectory = [(0, -1), (0, 0)]
    #print(solve_grid(_grid, _solve_trajectory))
    _visited = Grid([['.' for _ in range(_grid.width)] for _ in range(_grid.height)])

    a = find_beam_trajectories(_grid, _solve_trajectory, _visited)
    #solve_1 = nb_visited(FOUND_TRAJECTORIES, _grid)
    #print(solve_1)
    print(solve_2(_grid))
    pass

# guesses:
# 7615 : wrong
# 7931: ???
# 8036: ???
# 8036: wrong
# 8112: correct