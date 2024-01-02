import itertools
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from notebooks.aoc_2023.day_10 import is_connected_to_border
from notebooks.aoc_2023.utils import load_stripped_lines, ParametrizedLineCurve, JordanCurve, Complex, COORDINATE


class Direction(Enum):
    R = 'R'
    L = 'L'
    U = 'U'
    D = 'D'
    NONE = 'none'  # starting point


@dataclass
class Instruction:
    direction: Direction
    meters: int
    color: str  # in hex


def parse_dig_plan(_data: list[str]) -> list[Instruction]:
    plan = []
    for line in _data:
        splitted = line.split(' ')
        plan.append(Instruction(Direction(splitted[0]), int(splitted[1]), splitted[2]))
    return plan


def parse_dig_plan_part_two(_data: list[str]) -> list[Instruction]:
    plan = []
    for line in _data:
        hex_number = line.split(' ')[2].strip('()#')
        assert len(hex_number) == 6
        nb_meters = int(hex_number[:5], 16)
        last_digit = int(hex_number[-1])
        direction = None
        # The last hexadecimal digit encodes the direction to dig: 0 means R, 1 means D, 2 means L, and 3 means U.
        match last_digit:
            case 0: direction = Direction.R
            case 1: direction = Direction.D
            case 2: direction = Direction.L
            case 3: direction = Direction.U
        assert direction is not None
        plan.append(Instruction(direction, nb_meters, 'color'))
    return plan


def move_in_direction(start_coordinate: Complex, direction: Direction, meters: int) -> Complex:
    if direction == Direction.R:
        return start_coordinate + Complex(0, meters)
    if direction == Direction.L:
        return start_coordinate + Complex(0, -meters)
    if direction == Direction.U:
        return start_coordinate + Complex(-meters, 0)
    if direction == Direction.D:
        return start_coordinate + Complex(meters, 0)


def create_edge_of_a_ditch(plan: list[Instruction]) -> JordanCurve:
    start_coordinate = Complex(0, 0)
    edge_of_ditch = JordanCurve()
    for instruction in plan:
        end_coordinate = move_in_direction(start_coordinate, instruction.direction, instruction.meters)
        curve = ParametrizedLineCurve(start_coordinate, end_coordinate)
        edge_of_ditch.add_curve(curve)
        start_coordinate = end_coordinate
    assert start_coordinate == Complex(0, 0)  # we made and close curve
    return edge_of_ditch


def find_endge_and_inner_points(edge_of_ditch: JordanCurve) -> tuple[int, int]:
    ditch = np.pad(edge_of_ditch.draw(), 1, 'constant', constant_values=(1, ))
    print(f'ditch_len: {ditch.shape}')
    output = cv2.connectedComponentsWithStats(ditch, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    possible_points: list[list[COORDINATE]] = []
    for label in range(1, numLabels):  # 0 is background
        ind_y, ind_x = np.where(labels == label)
        if not is_connected_to_border(ind_y, ind_x, ditch.shape[1], ditch.shape[0]):
            # possible_points.append([(iy, ix) for iy, ix in zip(ind_y, ind_x, strict=True)])
            return len(edge_of_ditch), len(ind_y)

def working_solve_1(plan: list[Instruction]) -> int:
    edge_of_ditch = create_edge_of_a_ditch(plan)
    print(f'len of the ditch: {len(edge_of_ditch)}')
    print(f'area of the ditch: {edge_of_ditch.area()}')
    trench_enge, trench_inner = find_endge_and_inner_points(edge_of_ditch)

    total = trench_enge + trench_inner
    return total

class VertexIdentity(Enum):
    RIGHT = 'right'
    LEFT = 'left'


MAP_FROM_UP_DIRECTION_TO_VERTEX_IDENTITY = {
    (Direction.U, Direction.R): VertexIdentity.RIGHT,
    (Direction.U, Direction.L): VertexIdentity.LEFT,

    (Direction.R, Direction.D): VertexIdentity.RIGHT,
    (Direction.R, Direction.U): VertexIdentity.LEFT,

    (Direction.D, Direction.L): VertexIdentity.RIGHT,
    (Direction.D, Direction.R): VertexIdentity.LEFT,

    (Direction.L, Direction.D): VertexIdentity.LEFT,
    (Direction.L, Direction.U): VertexIdentity.RIGHT,
}


def solve_1(plan: list[Instruction]) -> int:
    edge_of_ditch = create_edge_of_a_ditch(plan)
    vertices = edge_of_ditch.get_vertices()
    vertex_identities = []
    for i, vertex in enumerate(vertices):
        direction = plan[i].direction
        direction_before = plan[i-1].direction
        vertex_identities.append((vertex, MAP_FROM_UP_DIRECTION_TO_VERTEX_IDENTITY[(direction_before, direction)]))
        pass

    inner_area = edge_of_ditch.area()
    len_of_edge = len(edge_of_ditch)
    nb_right_vertex = sum([identity == VertexIdentity.RIGHT for _, identity in vertex_identities])
    nb_left_vertex = sum([identity == VertexIdentity.LEFT for _, identity in vertex_identities])

    total = inner_area + (len_of_edge - (nb_right_vertex+nb_left_vertex))*0.5 + 0.75 * nb_right_vertex + 0.25 * nb_left_vertex
    return total


if __name__ == "__main__":
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/18.txt')
    dig_plan = parse_dig_plan(raw_data)
    s1 = solve_1(dig_plan)
    assert s1 in (58550, 62)
    print(s1)

    dig_plan_2 = parse_dig_plan_part_two(raw_data)
    s2 = solve_1(dig_plan_2)
    print(s2)
    pass
# part one: 4292 - too low

952408144115