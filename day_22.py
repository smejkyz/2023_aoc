from collections import defaultdict

import itertools
from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from notebooks.aoc_2023.utils import load_stripped_lines


class Brick:
    def __init__(self, x, y, z, x_size, y_size, z_size, name) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

        self.name = name

        self.is_stable: bool | None = None
        self.has_fall: bool = False

    def fall(self, z_decrement: int = 1) -> None:

        self.z = self.z - z_decrement
        # print(f'{self.__repr__()} falls to {self.z=}')
        self.has_fall = True

    def __repr__(self) -> str:
        return f'Brick({self.name}, {self.z=} )'

    @property
    def volume(self) -> int:
        return (self.x_size + 1) * (self.y_size + 1) * (self.z_size +1)

    @property
    def range_x(self) -> range:
        return range(self.x, self.x + self.x_size + 1)

    @property
    def range_y(self) -> range:
        return range(self.y, self.y + self.y_size + 1)

    @property
    def range_z(self) -> range:
        return range(self.z, self.z + self.z_size + 1)


_base_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NAMES = list(_base_names)
NAMES += [_i+_j for _i, _j in itertools.product(list(_base_names), list(_base_names))]
NAMES += [_i+_j+_k for _i, _j,_k in itertools.product(list(_base_names), list(_base_names), list(_base_names))]


def parse_bricks(_data: list[str]) -> list[Brick]:
    bricks = []
    for i, line in enumerate(_data):
        _tmp = line.split('~')
        x_1, y_1, z_1 = [int(item) for item in _tmp[0].strip().split(',')]
        x_2, y_2, z_2 = [int(item) for item in _tmp[1].strip().split(',')]
        name = str(i) # if i < len(NAMES) else 'Unknown'
        bricks.append(Brick(x=x_1, y=y_1, z=z_1, x_size=x_2-x_1, y_size=y_2-y_1, z_size=z_2-z_1, name=name))
    return bricks


class System:
    def __init__(self, bricks: list[Brick]) -> None:
        self.bricks = sorted(bricks, key=lambda b: b.z)
        self.zero_level = 0
        self.empty_space = '.'
        self.bricks_volumes = {}

        self.brick_in_space = self.init_space()
        self.update_space()

    @property
    def system_is_stable(self) -> bool:
        return all([br.is_stable for br in self.bricks])

    def init_space(self) -> np.ndarray:
        max_x = max([brick.x + brick.x_size for brick in self.bricks])
        max_y = max([brick.y + brick.y_size for brick in self.bricks])
        max_z = max([brick.z + brick.z_size for brick in self.bricks])
        space = np.full(shape=(max_x+1, max_y+1, max_z+1), dtype='<U8', fill_value=self.empty_space)
        return space

    def remove_brick_from_space(self, brick_name_remove: str) -> None:
        _xs, _ys, _zs = np.where(self.brick_in_space == brick_name_remove)
        for _x, _y, _z in zip(_xs, _ys, _zs):
            self.brick_in_space[_x, _y, _z] = self.empty_space

    def update_space(self, id_brick_change: int | None = None) -> None:
        if id_brick_change is None:
            # everything is recomputed
            self.brick_in_space[:, :, :] = self.empty_space
            for brick in self.bricks:
                for _x, _y, _z in itertools.product(brick.range_x, brick.range_y, brick.range_z):
                    self.brick_in_space[_x, _y, _z] = brick.name

                self.bricks_volumes[brick.name] = sum(self.brick_in_space.flatten() == brick.name)
        else:
            _xs, _ys, _zs = np.where(self.brick_in_space == self.bricks[id_brick_change].name)
            for _x, _y, _z in zip(_xs, _ys, _zs):
                self.brick_in_space[_x, _y, _z] = self.empty_space
            for _x, _y, _z in itertools.product(self.bricks[id_brick_change].range_x, self.bricks[id_brick_change].range_y, self.bricks[id_brick_change].range_z):
                self.brick_in_space[_x, _y, _z] = self.bricks[id_brick_change].name

            self.bricks_volumes[self.bricks[id_brick_change].name] = sum(self.brick_in_space.flatten() == self.bricks[id_brick_change].name)

        # check consistency:
        for brick in self.bricks:
            assert brick.volume == self.bricks_volumes[brick.name]

    def make_stable(self) -> None:
        # evolve the system that no brick can fall lower
        while not self.system_is_stable:
            for i, brick in enumerate(self.bricks):
                while not brick.is_stable:
                    if self.check_if_brick_is_stable(i):
                        brick.is_stable = True
                    else:
                        brick.fall()
                        self.update_space(i)
                pass
        pass

    def check_if_brick_is_stable(self, id_brick: int) -> bool:
        given_brick = self.bricks[id_brick]
        if given_brick.z == self.zero_level + 1:
            # cannot fall
            return True
        # check whether is something below the brick
        values_below = []
        for _x, _y in itertools.product(given_brick.range_x, given_brick.range_y):
            _val = self.brick_in_space[_x, _y, given_brick.z - 1]
            values_below.append(_val)
        pass
        if any([val != self.empty_space for val in values_below]):
            # there is something below brick - is stable
            # print(f'{given_brick} is stable at z={given_brick.z}')
            return True

        return False

    def remove_brick(self, id_brick: int) -> None:
        self.remove_brick_from_space(self.bricks[id_brick].name)
        self.bricks.pop(id_brick)
        for br in self.bricks:
            br.is_stable = False
            br.has_fall = False

    def is_stable(self) -> bool:
        for i, brick in enumerate(self.bricks):
            if not self.check_if_brick_is_stable(i):
                return False
        return True


def solve_1(bricks: list[Brick]) -> int:
    system = System(bricks)
    system.make_stable()
    can_be_disintegrated = []
    for i, brick in tqdm(enumerate(system.bricks)):
        print(f'testing system without brick {brick.name}')
        # create second system without this bric
        system_2 = deepcopy(system)
        assert system_2 is not system
        system_2.remove_brick(i)
        # system_2.make_stable()
        # nb_brick_fals[brick.name] = sum([br.has_fall for br in system_2.bricks])
        if system_2.is_stable():
            print('brick can be disintegrated')
            can_be_disintegrated.append(brick.name)
    print(len(can_be_disintegrated))



def solve_2(bricks: list[Brick]) -> int:
    system = System(bricks)
    system.make_stable()
    nb_brick_fals = defaultdict(int)
    for i, brick in tqdm(enumerate(system.bricks)):
        print(f'testing system without brick {brick.name}')
        if brick.name in ['199', '289', '472', '553', '899', '961', '1011', '463', '870', '929', '991', '1077', '52', '53', '349', '689', '603', '18', '882', '1069', '532', '55', '662', '884', '323', '999', '1064', '1105', '255', '17', '387', '450', '1099', '404', '299', '620', '650', '993', '346', '724', '750', '455', '666', '124', '1070', '520', '1052', '1029', '177', '80', '448', '312', '1117', '210', '549', '218', '490', '205', '1124', '589', '170', '182', '706', '1118', '335', '536', '935', '1040', '1197', '83', '611', '962', '703', '975', '1110', '420', '552', '739', '1139', '237', '861', '925', '564', '813', '939', '1084', '599', '145', '230', '264', '562', '700', '705', '394', '835', '579', '736', '1060', '837', '181', '201', '94', '1154', '1114', '100', '324', '451', '59', '409', '730', '1062', '1186', '331', '369', '940', '981', '1203', '42', '142', '687', '256', '799', '751', '880', '221', '1072', '252', '120', '471', '276', '566', '593', '189', '811', '1003', '712', '111', '375', '483', '507', '987', '143', '459', '521', '460', '902', '295', '427', '986', '1047', '1133', '1150', '421', '1149', '690', '711', '1013', '525', '796', '1183', '290', '534', '1051', '238', '947', '382', '327', '1079', '741', '179', '1103', '1156', '1006', '192', '542', '645', '896', '1078', '293', '485', '606', '640', '989', '63', '1113', '316', '642', '235', '45', '155', '147', '333', '405', '787', '1168', '1', '47', '191', '11', '296', '285', '417', '216', '1061', '1192', '196', '108', '167', '972', '359', '881', '227', '597', '749', '87', '1121', '173', '396', '443', '1049', '408', '397', '771', '653', '860', '27', '913', '757', '225', '250', '381', '592', '557', '649', '656', '81', '522', '1039', '379', '600', '900', '942', '1066', '1089', '931', '911', '1180', '629', '1044', '259', '570', '165', '207', '1184', '1042', '954', '580', '621', '856', '260', '475', '478', '807', '72', '808', '895', '1019', '1056', '291', '551', '1104', '1004', '663', '555', '668', '776', '571', '674', '389', '848', '958', '1102', '308', '469', '109', '843', '1123', '608', '1033', '121', '411', '982', '58', '1147', '995', '355', '416', '1037', '269', '454', '734', '1138', '1027', '1045', '77', '817', '818', '150', '297', '178', '266', '868', '1175', '122', '468', '529', '572', '526', '847', '816', '374', '437', '462', '745', '585', '979', '30', '358', '1198', '204', '1146', '366', '853', '76', '588', '14', '137', '441', '424', '85', '740', '548', '774', '190', '464', '185', '798', '393', '477', '183', '779', '326', '573', '990', '1108', '36', '673', '1169', '602', '171', '213', '1190', '440', '487', '1088', '797', '1160', '129', '1076', '432', '966', '148', '865', '119', '827', '878', '928', '1031', '164', '509', '563', '676', '426', '439', '744', '773', '1174', '948', '971', '510', '582', '249', '8', '546', '302', '132', '452', '665', '710', '789', '84', '24', '9', '828', '733', '701', '923', '239', '769']:
            # these does not make any move
            continue
        # create second system without this bric
        system_2 = deepcopy(system)
        assert system_2 is not system
        system_2.remove_brick(i)
        system_2.make_stable()
        nb_brick_fals[brick.name] = sum([br.has_fall for br in system_2.bricks])
        # if is_stable:
        #     print('brick can be disintegrated')
        #     can_be_disintegrated.append(brick.name)
    print(nb_brick_fals)
    return sum(nb_brick_fals.values())


if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/22.txt')
    _bricks = parse_bricks(raw_data)
    s1 = solve_1(_bricks)
    print(s1)
    pass


# s2:  1133 - wrong