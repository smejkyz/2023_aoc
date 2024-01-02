import itertools
from tqdm import tqdm

from notebooks.aoc_2023.utils import load_stripped_lines, Grid, COORDINATE, bfs, multiple_bfs, dijkstra, dijkstra_optimize


def load_maze(_raw_data: list[str]) -> Grid:
    return Grid([list(line) for line in _raw_data])


def expand_universe(_grid: Grid) -> Grid:
    new_rows = []
    for i in range(_grid.height):
        grid_values = _grid.get_row(i)
        if '#' in grid_values:
            new_rows.append(grid_values)
        else:
            new_rows.extend([grid_values,grid_values])
    _tmp_grid = Grid(new_rows)
    new_cols = []
    for j in range(_tmp_grid.width):
        grid_values = _tmp_grid.get_column(j)
        if '#' in grid_values:
            new_cols.append(grid_values)
        else:
            new_cols.extend([grid_values, grid_values])
    return Grid(list(zip(*new_cols)))


def find_shortest_path(start_point: COORDINATE, end_point: COORDINATE, maze: Grid):
    shortest_path = bfs(start_point, end_point, maze, set())
    return shortest_path


def solve_1(_universe: Grid) -> int:
    galaxies = _universe.get_coordinates('#')
    shortests_paths = {}
    print(f'galaxies: {len(galaxies)}')
    combinations = list(itertools.combinations(galaxies, 2))
    for galaxy in galaxies:
        other_galaxies_to_explore = []
        for one, two in combinations:
            if (one, two) in shortests_paths or (two, one) in shortests_paths:
                continue
            if one == galaxy:
                other_galaxies_to_explore.append(two)
            if two == galaxy:
                other_galaxies_to_explore.append(one)

        computed_paths = multiple_bfs(galaxy, other_galaxies_to_explore, _universe, border_values=set())
        for path in computed_paths:
            shortests_paths[(galaxy, path[-1])] = path

    steps = [len(value) - 1 for value in shortests_paths.values()]
    return sum(steps)


def build_distance_matrix(vertices: list[COORDINATE], edges: dict[COORDINATE, int]) -> list[list[float]]:
    distances = [[float("inf") for _ in range(len(vertices))] for _ in range(len(vertices))]
    # print(distances)
    for i, v in enumerate(vertices):
        for j, u in enumerate(vertices):
            if i == j:
                distances[i][j] = 0.0
            if (i, j) in edges:
                distances[i][j] = float(edges[(i, j)])

    return distances


def build_distance_matrix_optimize(vertices: list[COORDINATE], edges: dict[COORDINATE, int]) -> list[dict[float]]:
    distances = [{i: 0.0} for i in range(len(vertices))]
    # print(distances)
    for i, v in enumerate(vertices):
        for j, u in enumerate(vertices):
            if (i, j) in edges:
                distances[i][j] = float(edges[(i, j)])

    # dist_deprecated = build_distance_matrix(vertices, edges)
    # for i, (_one, _two) in enumerate(zip(distances, dist_deprecated)):
    #     if list(_one.values()) != _two:
    #         raise ValueError()
    return distances


def solve_1_using_diiksra(_universe: Grid) -> int:
    galaxies = _universe.get_coordinates('#')
    shortests_paths = {}
    print(f'galaxies: {len(galaxies)}')
    combinations = list(itertools.combinations(galaxies, 2))

    vertices = [(j, i) for i in range(_universe.width) for j in range(_universe.height)]
    edges: dict[tuple[int, int], int] = {}
    for vertex in vertices:
        vertex_id = vertices.index(vertex)
        for neighbour in _universe.four_neighbours(vertex):
            neighbour_id = vertices.index(neighbour)
            edges[(vertex_id, neighbour_id)] = 1

    distance_matrix = build_distance_matrix(vertices, edges)
    for galaxy_1, galaxy_2 in combinations:

        start_id = vertices.index(galaxy_1)
        end_id = vertices.index(galaxy_2)
        shortest_path = dijkstra(vertices=vertices, distance_matrix=distance_matrix, start=start_id, goal=end_id)

        shortests_paths[(galaxy_1, galaxy_2)] = [vertices[idx] for idx in shortest_path]
        # computed_paths = multiple_bfs(galaxy, other_galaxies_to_explore, _universe, border_values=set())
        # for path in computed_paths:
        #     shortests_paths[(galaxy, path[-1])] = path

    return shortests_paths
    steps = [len(value) - 1 for value in shortests_paths.values()]
    return sum(steps)


def solve_1_using_diiksra_base_universe(_base_universe: Grid): #, shortest_paths_CORRECT: dict, _base_universe_expanded: Grid) -> int:
    # shortest_paths_CORRECT_list = list(shortest_paths_CORRECT.values())

    scale_factor = 2
    galaxies = _base_universe.get_coordinates('#')
    empty_columns = [j for j in range(_base_universe.width) if '#' not in _base_universe.get_column(j)]

    empty_rows = [i for i in range(_base_universe.height) if '#' not in _base_universe.get_row(i)]
    shortests_paths = {}
    print(f'galaxies: {len(galaxies)}')
    combinations = list(itertools.combinations(galaxies, 2))

    vertices = [(j, i) for i in range(_base_universe.width) for j in range(_base_universe.height)]

    vertices_on_empty_columns = [(j, i) for (j, i) in vertices if i in empty_columns]
    vertices_on_empty_rows = [(j, i) for (j, i) in vertices if j in empty_rows]

    all_on_empty = vertices_on_empty_columns + vertices_on_empty_rows

    edges: dict[tuple[int, int], float] = {}
    for vertex in tqdm(vertices):
        if vertex == (7, 8):
            print(0)
        vertex_id = vertices.index(vertex)
        for neighbour in _base_universe.four_neighbours(vertex):
            neighbour_id = vertices.index(neighbour)
            if (vertex in all_on_empty and neighbour not in all_on_empty) or (vertex in vertices_on_empty_columns and vertex in vertices_on_empty_rows):
                edges[(vertex_id, neighbour_id)] = 2
            else:
                edges[(vertex_id, neighbour_id)] = 1

    distance_matrix = build_distance_matrix(vertices, edges)
    print('starting')
    for i, (galaxy_1, galaxy_2) in tqdm(enumerate(combinations)):

        start_id = vertices.index(galaxy_1)
        end_id = vertices.index(galaxy_2)
        shortest_path = dijkstra(vertices=vertices, distance_matrix=distance_matrix, start=start_id, goal=end_id)

        distances = [distance_matrix[_start][_end] for _start, _end in zip(shortest_path, shortest_path[1:])]
        path_true = [vertices[idx] for idx in shortest_path]
        #path_expected = shortest_paths_CORRECT_list[i]
        shortests_paths[(galaxy_1, galaxy_2)] = path_true, distances

        true_value = sum(distances)
        #exp_value = len(shortest_paths_CORRECT_list[i]) - 1
        #if true_value != exp_value:
        #    print('problem')
        #    # raise ValueError()
        pass
        # computed_paths = multiple_bfs(galaxy, other_galaxies_to_explore, _universe, border_values=set())
        # for path in computed_paths:
        #     shortests_paths[(galaxy, path[-1])] = path

    steps = [sum(values) for path, values in shortests_paths.values()]
    return sum(steps)


def solve_1_using_diiksra_base_universe_optimize(_base_universe: Grid): #  shortest_paths_CORRECT: dict, _base_universe_expanded: Grid) -> int:
    # shortest_paths_CORRECT_list = list(shortest_paths_CORRECT.values())

    galaxies = _base_universe.get_coordinates('#')
    empty_columns = [j for j in range(_base_universe.width) if '#' not in _base_universe.get_column(j)]

    empty_rows = [i for i in range(_base_universe.height) if '#' not in _base_universe.get_row(i)]
    shortests_paths = {}
    print(f'galaxies: {len(galaxies)}')
    combinations = list(itertools.combinations(galaxies, 2))

    vertices = [(j, i) for i in range(_base_universe.width) for j in range(_base_universe.height)]

    vertices_on_empty_columns = [(j, i) for (j, i) in vertices if i in empty_columns]
    vertices_on_empty_rows = [(j, i) for (j, i) in vertices if j in empty_rows]

    all_on_empty = vertices_on_empty_columns + vertices_on_empty_rows

    edges: dict[tuple[int, int], float] = {}
    for vertex in tqdm(vertices):
        if vertex == (7, 8):
            print(0)
        vertex_id = vertices.index(vertex)
        for neighbour in _base_universe.four_neighbours(vertex):
            neighbour_id = vertices.index(neighbour)
            if (vertex in all_on_empty and neighbour not in all_on_empty) or (vertex in vertices_on_empty_columns and vertex in vertices_on_empty_rows):
                edges[(vertex_id, neighbour_id)] = 1000000
            elif (vertex[1] == 113 and neighbour[1] == 114) or (vertex[1] == 114 and neighbour[1] == 113):
                edges[(vertex_id, neighbour_id)] = 1000000
            else:
                edges[(vertex_id, neighbour_id)] = 1

    print('building matrix')
    distance_matrix = build_distance_matrix_optimize(vertices, edges)
    print('starting')
    for i, (galaxy_1, galaxy_2) in tqdm(enumerate(combinations)):

        start_id = vertices.index(galaxy_1)
        end_id = vertices.index(galaxy_2)
        shortest_path = dijkstra_optimize(vertices=vertices, distance_matrix=distance_matrix, start=start_id, goal=end_id)

        distances = [distance_matrix[_start][_end] for _start, _end in zip(shortest_path, shortest_path[1:])]
        path_true = [vertices[idx] for idx in shortest_path]
        # path_expected = shortest_paths_CORRECT_list[i]
        shortests_paths[(galaxy_1, galaxy_2)] = path_true, distances

        true_value = sum(distances)
        # exp_value = len(shortest_paths_CORRECT_list[i]) - 1
        #if true_value != exp_value:
        #    print('problem')
        #    # raise ValueError()
        # pass
        # computed_paths = multiple_bfs(galaxy, other_galaxies_to_explore, _universe, border_values=set())
        # for path in computed_paths:
        #     shortests_paths[(galaxy, path[-1])] = path

    steps = [sum(values) for path, values in shortests_paths.values()]
    return sum(steps)



# def solve_2(_base_universe: Grid) -> int:
#     scale_factor = 2
#     # from the grid I need to create a  of weights
#     weights = {}
#     for i, j in itertools.product(range(_base_universe.width), range(_base_universe.height)):
#         for neighbour in _base_universe.four_neighbours((j, i)):
#
#             weights[((i, j), neighbour)] = value
#

if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/11.txt')
    base_universe = load_maze(data)
    universe = expand_universe(base_universe)

    # exp_result = solve_1_using_diiksra(universe)
    # s1 = solve_1_using_diiksra_base_universe(base_universe)# , exp_result, universe)
    # assert s1 == 8410
    s1_new = solve_1_using_diiksra_base_universe_optimize(base_universe)# , exp_result, universe)
    print(s1_new)
    assert s1_new == 8410
    # s1 = solve_1(universe)
    # assert  s1 == 9545480
    #print(s1)

    # s2 = solve_2(base_universe)