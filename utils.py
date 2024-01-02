import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field

import itertools
from typing import TypeVar, Generic, Iterable
import scipy.integrate as integrate
import numpy as np
from typing_extensions import Self


def load_raw_lines(path: str) -> list[str]:
    with open(path, "r") as fp:
        return [line for line in fp.readlines()]


def load_stripped_lines(path: str) -> list[str]:
    return [line.strip() for line in load_raw_lines(path)]


def load_lines_as_integers(path: str) -> list[int]:
    return [int(line) for line in load_stripped_lines(path)]


def _parse_int_or_none(x: str) -> int | None:
    try:
        return int(x)
    except ValueError:
        return None


def load_lines_as_optional_integers(path: str) -> list[int | None]:
    return [_parse_int_or_none(line) for line in load_stripped_lines(path)]


_T = TypeVar("_T", str, int, float)
COORDINATE = tuple[int, int]


class Grid(Generic[_T]):
    def __init__(self, raw_data: list[list[_T]]) -> None:
        self.data = raw_data
        self.height = len(raw_data)
        self.width = len(raw_data[0])

    @property
    def as_numpy(self) -> np.ndarray:
        return np.array(self.data)

    def __getitem__(self, item: COORDINATE) -> _T:
        return self.data[item[0]][item[1]]

    def __setitem__(self, key: COORDINATE, value: _T) -> None:
        self.data[key[0]][key[1]] = value

    def get_coordinates(self, value: _T) -> list[COORDINATE]:
        return [(i, j) for i in range(self.height) for j in range(self.width) if self.data[i][j] == value]

    def four_neighbours(self, pos: COORDINATE) -> list[COORDINATE]:
        _raw_neighbours = ((pos[0] - 1), pos[1]), ((pos[0] + 1), pos[1]), ((pos[0]), pos[1] - 1), ((pos[0]), pos[1] + 1)
        return [neighb for neighb in _raw_neighbours if 0 <= neighb[0] < self.height and 0 <= neighb[1] < self.width]

    def infinite_four_neighbour(self, pos: COORDINATE) -> list[COORDINATE]:
        # the grid repeats infinitely
        y, x = pos
        _candidates = [(y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x)]
        neighbours = []
        for _candidate in _candidates:
            _y, _x = _candidate
            x_shifted = _x % self.width
            y_shifted = _y % self.height
            if self.__getitem__((y_shifted, x_shifted)) != '#':
                neighbours.append((_y, _x))
        return neighbours

    def infinite_get_item(self, coor: COORDINATE) -> _T:
        _y, _x = coor
        x_shifted = _x % self.width
        y_shifted = _y % self.height
        return self.__getitem__((y_shifted, x_shifted))

    def get_row(self, i: int) -> list[_T]:
        return [self.data[i][j] for j in range(self.width)]

    def get_column(self, j: int) -> list[_T]:
        return [self.data[i][j] for i in range(self.height)]

    @classmethod
    def empty(cls, height: int, width: int, fill_value: _T) -> Self:
        raw_data = [[fill_value for _ in range(width)] for _ in range(height)]
        return cls(raw_data)

    def is_inside_grid(self, coor: COORDINATE) -> bool:
        _y, _x = coor
        return 0 <= _y < self.height and 0 <= _x < self.width


_NUMERIC = TypeVar("_NUMERIC", bound=float)


class Complex(Generic[_NUMERIC]):
    def __init__(self, x: _NUMERIC, y: _NUMERIC) -> None:
        self.x = x
        self.y = y

    def __add__(self, other):
        return Complex(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Complex(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if not isinstance(other, Complex) and isinstance(other, float | int):
            return Complex(self.x * other, self.y * other)
        real = self.x * other.x - self.y * other.y
        imag = self.x * other.y + self.y * other.x
        return Complex(real, imag)

    def __repr__(self) -> str:
        return f'Complex{self.x, self.y}'

    def __eq__(self, other) -> bool:
        if not isinstance(other, Complex):
            return False
        return self.y == other.y and self.x == other.x

    @property
    def is_strictly_real(self) -> bool:
        return self.y == 0


IMAG_UNIT = Complex(0, 1)


class ParametrizedLineCurve:
    def __init__(self, start_point: Complex[int], end_point: Complex[int]):
        self.start_point = start_point
        self.end_point = end_point

    def value(self, t: int) -> Complex:
        return Complex(self.value_real_part(t), self.value_imag_part(t))

    def value_real_part(self, t: int) -> int:
        return self.start_point.x + t * (self.end_point.x - self.start_point.x)

    def value_imag_part(self, t: int) -> int:
        return self.start_point.y + t * (self.end_point.y - self.start_point.y)

    def derivative(self) -> Complex[int]:
        return Complex(self.derivative_real_part(), self.derivative_imag_part())

    def derivative_real_part(self) -> int:
        return self.end_point.x - self.start_point.x

    def derivative_imag_part(self) -> int:
        return self.end_point.y - self.start_point.y

    def integral_over_curve(self, z_0: Complex[int]) -> Complex[float]:
        # compute integral curve over xi from 1 / (xi - z_0) which is
        # integral 0 to 1 from xi'(t) / (xi(t) - z_0)dt
        # in case of line-curve the derivative does not depend on t:

        x_0 = z_0.x
        y_0 = z_0.y

        def real_part_to_integrate(t: int):
            return (self.value_real_part(t) - x_0) / ((self.value_real_part(t) - x_0) ** 2 + (self.value_imag_part(t) - y_0) ** 2)

        def imag_part_to_integrate(t: int):
            return (self.value_imag_part(t) - y_0) / ((self.value_real_part(t) - x_0) ** 2 + (self.value_imag_part(t) - y_0) ** 2)

        real_part, error = integrate.quad(real_part_to_integrate, 0, 1, limit=100)
        if abs(error) > 1e-5:
            raise ValueError()

        imag_part, error = integrate.quad(imag_part_to_integrate, 0, 1)
        if abs(error) > 1e-5:
            raise ValueError()

        c = self.derivative()
        result = c * Complex(real_part, -1 * imag_part)
        return result

    def all_coordinates(self) -> list[COORDINATE]:
        if self.start_point.x == self.end_point.x:
            if self.start_point.y > self.end_point.y:
                return [(self.start_point.x, y) for y in range(self.end_point.y, self.start_point.y + 1)]
            else:
                return [(self.start_point.x, y) for y in range(self.start_point.y, self.end_point.y + 1)]
        elif self.start_point.y == self.end_point.y:
            if self.start_point.x > self.end_point.x:
                return [(x, self.start_point.y) for x in range(self.end_point.x, self.start_point.x + 1)]
            else:
                return [(x, self.start_point.y) for x in range(self.start_point.x, self.end_point.x + 1)]
        else:
            raise NotImplemented()

    def __len__(self) -> int:
        if self.start_point.x == self.end_point.x:
            return abs(self.start_point.y - self.end_point.y)
        elif self.start_point.y == self.end_point.y:
            return abs(self.start_point.x - self.end_point.x)
        else:
            raise NotImplemented()


class JordanCurve:
    def __init__(self, curves: list[ParametrizedLineCurve] | None = None) -> None:
        self.curves = curves if curves is not None else []

    def add_curve(self, curve: ParametrizedLineCurve) -> None:
        self.curves.append(curve)

    def winding_number(self, z_0: Complex) -> int:
        _sum = sum([curve.integral_over_curve(z_0) for curve in self.curves], Complex(0,0))
        number_complex = IMAG_UNIT * _sum * (-1/(2 * np.pi))
        assert abs(number_complex.y) < 1e-5
        w_number: int = round(number_complex.x)
        return w_number

    def draw(self) -> np.ndarray:
        min_y = min([min(curve.start_point.y, curve.end_point.y) for curve in self.curves])
        max_y = max([max(curve.start_point.y, curve.end_point.y) for curve in self.curves])
        min_x = min([min(curve.start_point.x, curve.end_point.x) for curve in self.curves])
        max_x = max([max(curve.start_point.x, curve.end_point.x) for curve in self.curves])
        canvas = np.ones(dtype=np.uint8, shape=(max_x - min_x + 1, max_y-min_y + 1))
        for curve in self.curves:
            for coor in curve.all_coordinates():
                coor_shifted = coor[0] - min_x, coor[1] - min_y
                # assert coor_shifted[0] >= 0
                # assert coor_shifted[1] < max_x
                # assert coor_shifted[1] >= 0
                # assert coor_shifted[1] < max_y

                canvas[coor_shifted] = 0
        return canvas

    def __len__(self) -> int:
        return sum([len(curve) for curve in self.curves])

    def get_vertices(self) -> list[COORDINATE]:
        return [(curve.start_point.y, curve.start_point.x) for curve in self.curves]

    def area(self) -> int:
        # Shoelace formula
        all_vertices = self.get_vertices()
        all_vertices_looped = [all_vertices[-1]] + all_vertices + [all_vertices[0]]
        # A &= \frac 1 2 \sum_{i=1}^n (x_iy_{i+1}-x_{i+1}y_i)

        x = np.array([0, 2, 2, 0])
        y = np.array([0, 0, 2, 2])
        poly_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        _sum = 0
        for i in range(len(all_vertices)):
            _sum += (all_vertices_looped[i][1]*all_vertices_looped[i+1][0] - all_vertices_looped[i+1][1]*all_vertices_looped[i][0])
        return abs(_sum / 2)


def reconstruct_path(goal: COORDINATE, predecessors: dict[COORDINATE, COORDINATE | None]) -> list[COORDINATE]:
    path = []
    current: COORDINATE | None = goal
    while current is not None:
        path.append(current)
        current = predecessors[current]
    return list(reversed(path))


def bfs(start, goal, grid, border_values: set[str] = {"#"}):
    visited = {start}
    predecessors = {start: None}
    open = [start]

    while open:
        current = open.pop(0)

        if current == goal:
            break

        for adjacent in grid.four_neighbours(current):
            if adjacent not in visited and grid[adjacent] not in border_values:
                visited.add(adjacent)
                open.append(adjacent)
                predecessors[adjacent] = current

    return reconstruct_path(goal, predecessors)


def dfs(start, goal, grid):
    visited = {start}
    predecessors = {start: None}
    open = [start]

    while open:
        current = open.pop()

        if current == goal:
            break

        for adjacent in grid.four_neighbors(current):
            if adjacent not in visited and grid[adjacent] != "#":
                visited.add(adjacent)
                open.append(adjacent)
                predecessors[adjacent] = current

    return reconstruct_path(goal, predecessors)

def multiple_bfs(start: COORDINATE, goals: list[COORDINATE], grid: Grid, border_values: set[str] = {"#"}):
    visited = {start}
    predecessors = {start: None}
    open = [start]

    left_to_visit = [goal for goal in goals]
    while open:
        current = open.pop(0)

        if current in left_to_visit:
            left_to_visit.remove(current)

        if not left_to_visit:
            break

        for adjacent in grid.four_neighbours(current):
            if adjacent not in visited and grid[adjacent] not in border_values:
                visited.add(adjacent)
                open.append(adjacent)
                predecessors[adjacent] = current

    return [reconstruct_path(goal, predecessors) for goal in goals]


#
# def _viz_open(screen: curses.window, open: list[PrioritizedState]) -> None:
#     screen.addstr(0, 0, ", ".join([f"({ALPHABET[ps.vertex_idx]}, {ps.priority:.0f}, {ps.time_added})" for ps in open]))
#     screen.clrtoeol()


def _reconstruct_path(predecessors: dict[int, int | None], goal: int) -> list[int]:
    current = predecessors[goal]
    path = [goal]
    while current is not None:
        path.append(current)
        current = predecessors[current]
    return list(reversed(path))


ALPHABET = "ABCDEFGHIJK"


@dataclass(order=True)
class PrioritizedState:
    priority: float
    time_added: int
    vertex_idx: int = field(compare=False)

    def __str__(self) -> str:
        return f"{ALPHABET[self.vertex_idx]} p={self.priority:.0f}, t={self.time_added}"


#def dijkstra(vertices: list[Point2], distance_matrix: list[list[float]], start: int, goal: int) -> list[int]:
def dijkstra(vertices: list, distance_matrix: list[list[float]], start: int, goal: int) -> list[int]:
    counter = itertools.count()

    predecessors: dict[int, int | None] = {start: None}
    open = [PrioritizedState(0, next(counter), start)]
    distances_from_start: dict[int, float] = defaultdict(lambda: float("inf"))
    distances_from_start[start] = 0

    visited: set[int] = set()

    while open:
        current_state = heapq.heappop(open)
        current = current_state.vertex_idx
        visited.add(current)
        current_vertex = vertices[current]

        if current == goal:
            break

        for adjacent, distance in enumerate(distance_matrix[current]):
            if math.isinf(distance):
                continue

            new_distance = distances_from_start[current] + distance
            if new_distance < distances_from_start[adjacent]:
                predecessors[adjacent] = current
                distances_from_start[adjacent] = new_distance
                heapq.heappush(open, PrioritizedState(new_distance, next(counter), adjacent))

    path = _reconstruct_path(predecessors, goal)
    #for v in path:
    #    screen.addstr(vertices[v].y, vertices[v].x, ALPHABET[v], curses.color_pair(Color.MAGENTA))
    #screen.addstr(1, 0, "->".join([ALPHABET[v] for v in path]), curses.color_pair(Color.MAGENTA))
    # animate(screen)
    #screen.nodelay(False)
    #screen.getkey()
    return path


def dijkstra_optimize(vertices: list, distance_matrix: list[dict[float]], start: int, goal: int) -> list[int]:
    counter = itertools.count()

    predecessors: dict[int, int | None] = {start: None}
    open = [PrioritizedState(0, next(counter), start)]
    distances_from_start: dict[int, float] = defaultdict(lambda: float("inf"))
    distances_from_start[start] = 0

    visited: set[int] = set()

    while open:
        current_state = heapq.heappop(open)
        current = current_state.vertex_idx
        visited.add(current)
        current_vertex = vertices[current]

        if current == goal:
            break

        for adjacent, distance in distance_matrix[current].items():
            if math.isinf(distance):
                continue

            new_distance = distances_from_start[current] + distance
            if new_distance < distances_from_start[adjacent]:
                predecessors[adjacent] = current
                distances_from_start[adjacent] = new_distance
                heapq.heappush(open, PrioritizedState(new_distance, next(counter), adjacent))

    path = _reconstruct_path(predecessors, goal)
    #for v in path:
    #    screen.addstr(vertices[v].y, vertices[v].x, ALPHABET[v], curses.color_pair(Color.MAGENTA))
    #screen.addstr(1, 0, "->".join([ALPHABET[v] for v in path]), curses.color_pair(Color.MAGENTA))
    # animate(screen)
    #screen.nodelay(False)
    #screen.getkey()
    return path


import heapq


def dijkstra_maze(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    # Priority queue to store the vertices to visit
    pq = [(0, start)]
    heapq.heapify(pq)

    # Dictionary to store the shortest distance to each vertex
    distance = {start: 0}

    while pq:
        current_dist, current_pos = heapq.heappop(pq)

        # Check if we reached the destination
        if current_pos == end:
            return distance[end]

        # Explore neighbors
        for drow, dcol in directions:
            new_row, new_col = current_pos[0] + drow, current_pos[1] + dcol

            # Check if the neighbor is within bounds
            if 0 <= new_row < rows and 0 <= new_col < cols:
                neighbor = (new_row, new_col)
                weight = maze[new_row][new_col]

                # Calculate the new distance
                new_dist = current_dist + weight

                # Update the distance if it's shorter
                if neighbor not in distance or new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

    # If no path is found
    return float('inf')


if __name__ == "__main__":
    maze = [
        [1, 3, 1, 2],
        [6, 2, 5, 9],
        [8, 7, 4, 5],
        [3, 7, 2, 8]
    ]
    start_position = (0, 0)
    end_position = (3, 3)

    result = dijkstra_maze(maze, start_position, end_position)
    print("Shortest path length:", result)

952408144115
952404941483