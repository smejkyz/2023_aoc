from typing import TypeVar, Generic
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

    @classmethod
    def empty(cls, height: int, width: int, fill_value: _T) -> Self:
        raw_data = [[fill_value for _ in range(width)] for _ in range(height)]
        return cls(raw_data)


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


if __name__ == "__main__":
    point_1 = Complex(0, 0)
    point_2 = Complex(2, 0)
    point_3 = Complex(2, 2)
    point_4 = Complex(0, 2)
    curves = [
        ParametrizedLineCurve(point_1, point_2),
        ParametrizedLineCurve(point_2, point_3),
        ParametrizedLineCurve(point_3, point_4),
        ParametrizedLineCurve(point_4, point_1),
    ]
    point = Complex(5, 5)
    jordan = JordanCurve(curves)
    jordan.winding_number(point)
