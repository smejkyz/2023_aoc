import cv2
import numpy as np
from tqdm import tqdm

from notebooks.aoc_2023.utils import load_stripped_lines, Grid, COORDINATE, ParametrizedLineCurve, Complex, JordanCurve


def load_maze(_raw_data: list[str]) -> Grid:
    return Grid([list(line) for line in _raw_data])


def next_moves(current: COORDINATE, grid: Grid) -> tuple[COORDINATE, COORDINATE]:
    def north(_tmp: COORDINATE) -> COORDINATE:
        return _tmp[0] - 1, _tmp[1]

    def south(_tmp: COORDINATE) -> COORDINATE:
        return _tmp[0] + 1, _tmp[1]

    def east(_tmp: COORDINATE) -> COORDINATE:
        return _tmp[0], _tmp[1] + 1

    def west(_tmp: COORDINATE) -> COORDINATE:
        return _tmp[0], _tmp[1] - 1

    current_pipe = grid[current]
    match current_pipe:
        case '|': return north(current), south(current)
        case '-': return east(current), west(current)
        case 'L': return north(current), east(current)
        case 'J': return north(current), west(current)
        case '7': return south(current), west(current)
        case 'F': return south(current), east(current)
    raise NotImplemented()


def select_next_position(current_position, previous_position, _maze) -> COORDINATE:
    current_pipe = _maze[current_position]
    if current_pipe == 'S':
        pos_neighbours = [neig for neig in _maze.four_neighbours(current_position) if _maze[neig] != '.']
        return current_position[0] + 1, current_position[1]  # todo: this only works for my input, fix to make general
        assert len(pos_neighbours) == 2
        return pos_neighbours[0]

    possible_moves = next_moves(current_position, _maze)
    assert previous_position in possible_moves

    if previous_position == possible_moves[0]:
        return possible_moves[1]
    if previous_position == possible_moves[1]:
        return possible_moves[0]


def loop_finder(_maze: Grid) -> list[COORDINATE]:

    start_position = _maze.get_coordinates('S')[0]
    empty_grid = Grid.empty(_maze.height, _maze.width, '.')
    empty_grid[start_position] = 0

    current_position = start_position
    previous_position = start_position
    all_positions = [current_position]
    while True:
        next_position = select_next_position(current_position, previous_position, _maze)
        if next_position == start_position:
            return all_positions, empty_grid
        _tmp = current_position
        current_position = next_position
        previous_position = _tmp

        empty_grid[current_position] = len(all_positions)
        all_positions.append(next_position)


def is_connected_to_border(ind_y: np.array, ind_x: np.array, width, height) -> bool:
    for iy, ix in zip(ind_y, ind_x, strict=True):
        if iy == 0 or iy == height - 1 or ix == 0 or ix == width -1:
            return True
    return False


def solve_2(find_loop: list[COORDINATE], _maze: Grid, _path_help: Grid) -> int:

    test = _path_help.as_numpy
    test[test != '.'] = 0
    test[test == '.'] = 1

    output = cv2.connectedComponentsWithStats(test.astype(np.uint8), 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    possible_points: list[list[COORDINATE]] = []
    for label in range(1, numLabels):  # 0 is background
        ind_y, ind_x = np.where(labels == label)
        if not is_connected_to_border(ind_y, ind_x, _maze.width, _maze.height):
            possible_points.append([(iy, ix) for iy, ix in zip(ind_y, ind_x, strict=True)])

    curves = [ParametrizedLineCurve(Complex(point_start[1], point_start[0]), Complex(point_end[1], point_end[0])) for point_start, point_end in zip(find_loop, find_loop[1:])]
    curves.append(ParametrizedLineCurve(Complex(find_loop[-1][1], find_loop[-1][0]), Complex(find_loop[0][1], find_loop[0][0])))

    jordan_curve = JordanCurve(curves)
    print(f'possible points: {len(possible_points)}')
    points_inside = []
    for _set_of_points in tqdm(possible_points):
        # choose representant
        _repre = _set_of_points[0]
        z_0 = Complex(_repre[1], _repre[0])
        winding_number = jordan_curve.winding_number(z_0)
        if winding_number != 0:
            points_inside.extend(_set_of_points)
    return len(points_inside)


if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/10.txt')
    maze = load_maze(data)
    start_coor = maze.get_coordinates('S')[0]

    _find_loop, path = loop_finder(maze)

    solve_1 = len(_find_loop) // 2

    solution_2 = solve_2(_find_loop, maze, path)
    assert solution_2 == 303
