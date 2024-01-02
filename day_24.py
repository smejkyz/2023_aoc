import itertools
from dataclasses import dataclass

import numpy as np

from notebooks.aoc_2023.utils import load_stripped_lines


@dataclass
class Hailstone:
    x: int
    y: int
    z: int
    v_x: int
    v_y: int
    v_z: int

    @property
    def line_2d_coordites(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x+self.v_x, self.y+self.v_y

    def correct_side(self, p_x, p_y) -> bool:
        if (self.x < p_x and self.v_x < 0) or (self.x > p_x and self.v_x > 0):
            # problem, does not intersect
            return False
        if (self.y < p_y and self.v_y < 0) or (self.y > p_y and self.v_y > 0):
            return False
        return True

    @property
    def direction_vector(self) -> tuple[int, int, int]:
        return self.v_x, self.v_y, self.v_z

    @property
    def start_point(self) -> tuple[int, int, int]:
        return self.x, self.y, self.z

    def pos_in_time(self, t) -> tuple[int, int, int]:
        return self.x + t*self.v_x, self.y + t*self.v_y, self.z + t*self.v_z


def parse_hailstones(data: list[str]) -> list[Hailstone]:
    hail = []
    for line in data:
        x, y, z = [int(item.strip()) for item in line.split('@')[0].split(', ')]
        vx, vy, vz = [int(item.strip()) for item in line.split('@')[1].split(', ')]
        hail.append(Hailstone(x,y,z,vx,vy,vz))
    return hail


def compute_intersection_2d(h_1: Hailstone, h_2: Hailstone) -> tuple[float, float] | None:

    x_1, y_1, x_2, y_2 = h_1.line_2d_coordites
    x_3, y_3, x_4, y_4 = h_2.line_2d_coordites

    denom = (x_1 - x_2) * (y_3 - y_4) - (y_1 - y_2) * (x_3 - x_4)
    if denom == 0:
        return None
    p_x = ((x_1*y_2 - y_1*x_2) * (x_3 - x_4) - (x_1 - x_2) * (x_3*y_4 - y_3*x_4)) / denom
    p_y = ((x_1*y_2 - y_1*x_2) * (y_3 - y_4) - (y_1 - y_2) * (x_3*y_4 - y_3*x_4)) / denom

    # last check whether the computed intersection is in the righ side of the semi-infinite line segment
    # h1:
    h_1_is_on_right_side = h_1.correct_side(p_x, p_y)
    h_2_is_on_right_side = h_2.correct_side(p_x, p_y)
    if h_1_is_on_right_side and h_2_is_on_right_side:
        return p_x, p_y
    return None


def solve_1(hailstones: list[Hailstone]) -> int:
    # _min, _max = 7, 27
    _min, _max = 200000000000000, 400000000000000
    crossed = []
    for h_1, h_2 in itertools.combinations(hailstones, 2):
        intersection = compute_intersection_2d(h_1, h_2)
        if intersection is not None:
            p_x, p_y = intersection
            if (_min <= p_x <= _max) and (_min <= p_y <= _max):
                crossed.append((h_1, h_2))
    return len(crossed)


def equations_linear(p: tuple[int, int, int, int, int, int], h_1: Hailstone, h_2: Hailstone):
    x, y, z, v_x, v_y, v_z = p
    # x vs y
    _first = x * (h_1.v_y - h_2.v_y) - y*(h_1.v_x - h_2.v_x) - v_x * ( h_1.y - h_2.y) + v_y * (h_1.x - h_2.x)  \
            - h_1.x*h_1.v_y + h_2.x*h_2.v_y + h_1.y * h_1.v_x - h_2.y * h_2.v_x
    # x vs z
    _second = x * (h_1.v_z - h_2.v_z) - z * (h_1.v_x - h_2.v_x) - v_x * (h_1.z - h_2.z) + v_z * (h_1.x - h_2.x) \
             - h_1.x * h_1.v_z + h_2.x * h_2.v_z + h_1.z * h_1.v_x - h_2.z * h_2.v_x

    # y vs z
    _third = y * (h_1.v_z - h_2.v_z) - z * (h_1.v_y - h_2.v_y) - v_y * (h_1.z - h_2.z) + v_z * (h_1.y - h_2.y) \
              - h_1.y * h_1.v_z + h_2.y * h_2.v_z + h_1.z * h_1.v_y - h_2.z * h_2.v_y

    return _first, _second, _third


def solve_2_linear(hailstones: list[Hailstone]) -> int:
    h_1 = hailstones[0]
    h_2 = hailstones[1]

    # x vs y
    _first = [(h_1.v_y - h_2.v_y), -(h_1.v_x - h_2.v_x), 0 , -( h_1.y - h_2.y), (h_1.x - h_2.x), 0]
    _first_b = (-1) * ( - h_1.x*h_1.v_y + h_2.x*h_2.v_y + h_1.y * h_1.v_x - h_2.y * h_2.v_x)
    # x vs z
    _second = [ (h_1.v_z - h_2.v_z), 0 , - (h_1.v_x - h_2.v_x), - (h_1.z - h_2.z), 0,  (h_1.x - h_2.x)]
    _second_b = (-1) * (- h_1.x * h_1.v_z + h_2.x * h_2.v_z + h_1.z * h_1.v_x - h_2.z * h_2.v_x)

    # y vs z
    _third = [0, (h_1.v_z - h_2.v_z),  -(h_1.v_y - h_2.v_y), 0,  - (h_1.z - h_2.z), (h_1.y - h_2.y)]
    _thrid_b = (-1) * ( - h_1.y * h_1.v_z + h_2.y * h_2.v_z + h_1.z * h_1.v_y - h_2.z * h_2.v_y)

    _tmp_A_i = [_first, _second, _third]
    _tm_b_i = [_first_b, _second_b, _thrid_b]

    h_1 = hailstones[10]
    h_2 = hailstones[11]

    # x vs y
    _first = [(h_1.v_y - h_2.v_y), -(h_1.v_x - h_2.v_x), 0, -(h_1.y - h_2.y), (h_1.x - h_2.x), 0]
    _first_b = (-1) * (- h_1.x * h_1.v_y + h_2.x * h_2.v_y + h_1.y * h_1.v_x - h_2.y * h_2.v_x)
    # x vs z
    _second = [(h_1.v_z - h_2.v_z), 0, - (h_1.v_x - h_2.v_x), - (h_1.z - h_2.z), 0, (h_1.x - h_2.x)]
    _second_b = (-1) * (- h_1.x * h_1.v_z + h_2.x * h_2.v_z + h_1.z * h_1.v_x - h_2.z * h_2.v_x)

    # y vs z
    _third = [0, (h_1.v_z - h_2.v_z), -(h_1.v_y - h_2.v_y), 0, - (h_1.z - h_2.z), (h_1.y - h_2.y)]
    _thrid_b = (-1) * (- h_1.y * h_1.v_z + h_2.y * h_2.v_z + h_1.z * h_1.v_y - h_2.z * h_2.v_y)

    _tmp_A_ii = [_first, _second, _third]
    _tm_b_ii = [_first_b, _second_b, _thrid_b]

    A = np.array(_tmp_A_i + _tmp_A_ii, dtype=int)
    b = np.array(_tm_b_i + _tm_b_ii)
    solution = np.linalg.solve(A, b)
    print(solution[0] + solution[1] + solution[2])
    x, y, z = solution[0], solution[1], solution[2]
    vx, vy, vz = int(solution[3]), int(solution[4]), int(solution[5])
    return Hailstone(x,y,z,vx,vy,vz)


def solve_2_linear_only_start(hailstones: list[Hailstone]) -> int:
    h_1 = hailstones[0]
    h_2 = hailstones[1]
    h_3 = hailstones[2]
    rock_v_x, rock_v_y, rock_v_z = -261, 15, 233
    rock_v_x, rock_v_y, rock_v_z = -3, 1, 2
    # x vs y
    _first = [(h_1.v_y - h_2.v_y), -(h_1.v_x - h_2.v_x), 0]
    _first_b = (-1) * ( - h_1.x*h_1.v_y + h_2.x*h_2.v_y + h_1.y * h_1.v_x - h_2.y * h_2.v_x -( h_1.y - h_2.y)*rock_v_x + (h_1.x - h_2.x)*rock_v_y)
    # x vs z
    _second = [ (h_1.v_z - h_2.v_z), 0 , - (h_1.v_x - h_2.v_x)]
    _second_b = (-1) * (- h_1.x * h_1.v_z + h_2.x * h_2.v_z + h_1.z * h_1.v_x - h_2.z * h_2.v_x  - (h_1.z - h_2.z)*rock_v_x + (h_1.x - h_2.x) * rock_v_z      )

    # y vs z
    _third = [0, (h_1.v_z - h_3.v_z),  -(h_1.v_y - h_3.v_y)]
    _thrid_b = (-1) * ( - h_1.y * h_1.v_z + h_3.y * h_3.v_z + h_1.z * h_1.v_y - h_3.z * h_3.v_y - (h_1.z - h_3.z)*rock_v_y +(h_1.y - h_3.y)*rock_v_z)

    _tmp_A_i = [_first, _second, _third]
    _tm_b_i = [_first_b, _second_b, _thrid_b]

    solution = np.linalg.solve(np.array(_tmp_A_i), np.array(_tm_b_i))

    pass
    h_1 = hailstones[2]
    h_2 = hailstones[4]

    # x vs y
    _first = [(h_1.v_y - h_2.v_y), -(h_1.v_x - h_2.v_x), 0, -(h_1.y - h_2.y), (h_1.x - h_2.x), 0]
    _first_b = (-1) * (- h_1.x * h_1.v_y + h_2.x * h_2.v_y + h_1.y * h_1.v_x - h_2.y * h_2.v_x)
    # x vs z
    _second = [(h_1.v_z - h_2.v_z), 0, - (h_1.v_x - h_2.v_x), - (h_1.z - h_2.z), 0, (h_1.x - h_2.x)]
    _second_b = (-1) * (- h_1.x * h_1.v_z + h_2.x * h_2.v_z + h_1.z * h_1.v_x - h_2.z * h_2.v_x)

    # y vs z
    _third = [0, (h_1.v_z - h_2.v_z), -(h_1.v_y - h_2.v_y), 0, - (h_1.z - h_2.z), (h_1.y - h_2.y)]
    _thrid_b = (-1) * (- h_1.y * h_1.v_z + h_2.y * h_2.v_z + h_1.z * h_1.v_y - h_2.z * h_2.v_y)

    _tmp_A_ii = [_first, _second, _third]
    _tm_b_ii = [_first_b, _second_b, _thrid_b]

    A = np.array(_tmp_A_i + _tmp_A_ii, dtype=np.float64)
    b = np.array(_tm_b_i + _tm_b_ii)
    solution = np.linalg.solve(A, b)
    print(solution[0] + solution[1] + solution[2])
    return Hailstone(*list(solution))



def solve_2(hailstones: list[Hailstone]) -> int:
    from scipy.optimize import fsolve
    import math

    def equations(p):
        x_r, y_r, z_r, v_rx, v_ry, v_rz = p
        eqs = []
        eqs.extend(equations_linear(p, hailstones[0], hailstones[1]))
        eqs.extend(equations_linear(p, hailstones[1], hailstones[2]))
        # for i, h in enumerate(hailstones):
        #     eqs.append(_distance(np.array((v_rx, v_ry, v_rz)), np.array((x_r, y_r, z_r)), np.array(h.direction_vector), np.array(h.start_point)))

        return eqs[:len(p)]
    init_pos = hailstones[1].start_point
    init_vel = hailstones[1].direction_vector
    init_solution = init_pos + init_vel
    solution = fsolve(equations, init_solution, full_output=True)
    could_be_solution = solution[0][0] + solution[0][1] + solution[0][2]
    return could_be_solution


def solve_2_only_start(hailstones: list[Hailstone]) -> int:
    from scipy.optimize import fsolve
    import math

    def equations(p):
        x_r, y_r, z_r = p
        eqs = []
        eqs.extend(_equal_functions_only_start(p, hailstones[1]))
        # eqs.extend(_equal_functions(p, hailstones[1]))
        # for i, h in enumerate(hailstones):
        #     eqs.append(_distance(np.array((v_rx, v_ry, v_rz)), np.array((x_r, y_r, z_r)), np.array(h.direction_vector), np.array(h.start_point)))

        return eqs[:len(p)]
    init_pos = hailstones[50].start_point
    init_vel = (hailstones[199].v_x, hailstones[1].v_y, hailstones[1].v_z)
    init_solution = init_pos # + init_vel
    solution = fsolve(equations, init_solution , full_output=True)
    can_be = solution[0][0] + solution[0][1] + solution[0][2]
    return can_be
    pass


def _distance(d_1, r_1, d_2, r_2) -> int:
    n = np.cross(d_1, d_2)
    dot_product = np.dot(n, r_1 - r_2)
    d = dot_product / np.linalg.norm(n)
    return d


def distance(h_1: Hailstone, h_2: Hailstone) -> int:
    d_1 = h_1.direction_vector
    d_2 = h_2.direction_vector
    r_1 = np.array(h_1.start_point)
    r_2 = np.array(h_2.start_point)
    d = _distance(d_1, r_1, d_2, r_2)

    n = np.cross(d_1, d_2)
    t_1 = np.dot(np.cross(d_2, n), (r_2-r_1) / np.dot(n, n))
    t_2 = np.dot(np.cross(d_1, n), (r_2-r_1) / np.dot(n, n))

    pos_1 = h_1.pos_in_time(t_1)
    pos_2 = h_2.pos_in_time(t_2)
    return d


def _equal_functions_only_start(p: tuple[int, int, int], h: Hailstone) -> tuple[int, int, int]:
    x, y, z, = p
    _first = (h.x - x) * (20 - h.v_y) - (h.y - y) * (35 - h.v_x)
    _second = (h.x - x) * (33 - h.v_z) - (h.z - z) * (35 - h.v_x)
    _third = (h.y - y) * (33 - h.v_z) - (h.z - z) * (20 - h.v_y)
    return _first, _second, _third


def _equal_functions(p: tuple[int,int,int,int,int,int], h: Hailstone) -> tuple[int, int, int]:
    x, y, z, v_x, v_y, v_z = p
    _first = (h.x - x)  * (v_y - h.v_y) - (h.y - y) * (v_x - h.v_x)
    _second = (h.x - x) * (v_z - h.v_z) - (h.z - z) * (v_x - h.v_x)
    _third = (h.y - y)  * (v_z - h.v_z) - (h.z - z) * (v_y - h.v_y)
    return _first, _second, _third


def _tmp_test(rock: Hailstone, h: Hailstone):
    _first = (h.x - rock.x)*(rock.v_y - h.v_y) - (h.y - rock.y)*(rock.v_x - h.v_x)
    _second = (h.x - rock.x)*(rock.v_z - h.v_z) - (h.z - rock.z) * (rock.v_x - h.v_x)
    _third = (h.y - rock.y)*(rock.v_z - h.v_z) - (h.z - rock.z) * (rock.v_y - h.v_y)
    pass


def _tmp_test_ii(rock: Hailstone, h_1: Hailstone, h_2: Hailstone):
    x, y, z = rock.start_point
    v_x, v_y, v_z = rock.direction_vector
    # x vs y
    _first = x * (h_1.v_y - h_2.v_y) - y*(h_1.v_x - h_2.v_x) - v_x * ( h_1.y - h_2.y) + v_y * (h_1.x - h_2.x)  \
            - h_1.x*h_1.v_y + h_2.x*h_2.v_y + h_1.y * h_1.v_x - h_2.y * h_2.v_x
    # x vs z
    _second = x * (h_1.v_z - h_2.v_z) - z * (h_1.v_x - h_2.v_x) - v_x * (h_1.z - h_2.z) + v_z * (h_1.x - h_2.x) \
             - h_1.x * h_1.v_z + h_2.x * h_2.v_z + h_1.z * h_1.v_x - h_2.z * h_2.v_x

    # y vs z
    _third = y * (h_1.v_z - h_2.v_z) - z * (h_1.v_y - h_2.v_y) - v_y * (h_1.z - h_2.z) + v_z * (h_1.y - h_2.y) \
              - h_1.y * h_1.v_z + h_2.y * h_2.v_z + h_1.z * h_1.v_y - h_2.z * h_2.v_y

    pass

if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/24.txt')
    _hailstones = parse_hailstones(raw_data)
    # s1 = solve_1(_hailstones) # correct is 15318
    # print(s1)
    #s2 = solve_2_only_start(_hailstones)
    rock = solve_2_linear(_hailstones)

    for h_1 in _hailstones:
        print(distance(rock, h_1))

    # rock = solve_2_linear_only_start(_hailstones)
    # rock = Hailstone(24, 13, 10, -3, 1, 2)
    # rock_2 = Hailstone(262253147094843, 220705240641723, 274249842555989, 35, 20, 33)
    #print(rock_2.x + rock_2.y + rock_2.z)
    # [3.13593461e+14 3.18994946e+14 2.75400620e+14] 3.18994946e+14 2.75400620e+14]
    for h_1, h_2 in itertools.combinations(_hailstones, 2):
        print(_tmp_test_ii(rock, h_1, h_2))
    # print(s2)

    _h_1 = Hailstone(0, 0, 0, 1, 0, 0)
    _h_2 = Hailstone(5, 6, 7, 0, 1, 0)

    _h_3 = Hailstone(5, 6, 7, 1, 1, 0)

    distance(_h_1, _h_3)

    A = np.array([[-2, -1, -2, -1, 1], [1, -1, -2, -2, -5], [-2, -2, -4, -1, 3]])
    rank = np.linalg.matrix_rank(A)
    q, r = np.linalg.qr(A)

# 909216073709151 : too high
# 907989027654894: now
# # 870379016024859 is correct
# 803241397576453: too low

# 958650524729902.0: too high