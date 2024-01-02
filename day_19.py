import json
import operator
from copy import copy
from dataclasses import dataclass
import re
from enum import Enum

from notebooks.aoc_2023.utils import load_stripped_lines, load_raw_lines


@dataclass
class Rating:
    x: int
    m: int
    a: int
    s: int

    accepted: bool | None = None

    @property
    def value(self) -> int:
        return self.x + self.m + self.a + self.s


@dataclass
class IntervalRating:
    x: tuple[int, int]  # min/max values
    m: tuple[int, int]  # min/max values
    a: tuple[int, int]  # min/max values
    s: tuple[int, int]  # min/max values

    @property
    def possible_states(self) -> int:
        return (self.x[1] - self.x[0] + 1) * (self.m[1] - self.m[0] + 1) * (self.a[1] - self.a[0] + 1) * (self.s[1] - self.s[0] + 1)

def parse_ratings(data: list[str]) -> list[Rating]:
    _str_ratings = ''.join(data).split('\n\n')[1].split('\n')
    all_ratings = []
    for plain_str in _str_ratings:
        all_ratings.append(Rating(**{key: int(val) for key, val in re.findall(r'(\w+)=(\w+)', plain_str)}))

    return all_ratings


class Rule:
    def __init__(self, raw_rule: str) -> None:
        if ':' in raw_rule:
            _splitted = raw_rule.split(':')
            self.escape_state = _splitted[1]
            self.always_true = False

            if '<' in _splitted[0]:
                self.comparison_operator = operator.lt
                self.attribute = _splitted[0].split('<')[0]
                self.threshold = int(_splitted[0].split('<')[1])
            elif '>' in _splitted[0]:
                self.comparison_operator = operator.gt
                self.attribute = _splitted[0].split('>')[0]
                self.threshold = int(_splitted[0].split('>')[1])
            else:
                raise NotImplemented()

        else:
            self.always_true = True
            self.escape_state = raw_rule

    def is_accepted(self, input: Rating) -> bool:
        if self.always_true:
            return True
        return self.comparison_operator(getattr(input, self.attribute), self.threshold)

    def __repr__(self) -> str:
        if self.always_true:
            return f'Rule({self.escape_state})'
        return f'Rule({self.attribute} {self.comparison_operator} {self.threshold}, {self.escape_state}'

    def transform_interval(self, input: IntervalRating) -> tuple[IntervalRating | None, IntervalRating | None]:
        if self.always_true:
            return input, None
        # split interval into satisfied and not satisfied condition:
        if self.comparison_operator == operator.lt:
            return self.transform_interval_lt(input)
        if self.comparison_operator == operator.gt:
            return self.transform_interval_gt(input)

    def transform_interval_lt(self, input: IntervalRating) -> tuple[IntervalRating | None, IntervalRating | None]:
        # split interval into satisfied and not satisfied condition:
        given_attributes_limits = getattr(input, self.attribute)
        if given_attributes_limits[0] < self.threshold <= given_attributes_limits[1]:
            satisfied_condition = copy(input)
            setattr(satisfied_condition, self.attribute, (given_attributes_limits[0], self.threshold - 1))
            not_satisfied_condition = copy(input)
            setattr(not_satisfied_condition, self.attribute, (self.threshold, given_attributes_limits[1]))
        elif given_attributes_limits[1] < self.threshold:
            # everything is satisfied:
            satisfied_condition = copy(input)
            not_satisfied_condition = None
        elif self.threshold <= given_attributes_limits[0]:
            # nothing  is satisfied:
            satisfied_condition = None
            not_satisfied_condition = copy(input)
        # elif self.threshold == given_attributes_limits[1]:
            # everything accept the last point is satisfied - this is identical to the first condition

            # satisfied_condition = copy(input)
            # setattr(satisfied_condition, self.attribute, (given_attributes_limits[0], self.threshold - 1))
            # not_satisfied_condition = copy(input)
            # setattr(not_satisfied_condition, self.attribute, (self.threshold, given_attributes_limits[1]))
        else:
            raise NotImplemented()

        return satisfied_condition, not_satisfied_condition

    def transform_interval_gt(self, input: IntervalRating) -> tuple[IntervalRating | None, IntervalRating | None]:
        # split interval into satisfied and not satisfied condition:
        given_attributes_limits = getattr(input, self.attribute)
        if given_attributes_limits[0] <= self.threshold < given_attributes_limits[1]:
            satisfied_condition = copy(input)
            setattr(satisfied_condition, self.attribute, (self.threshold + 1, given_attributes_limits[1]))
            not_satisfied_condition = copy(input)
            setattr(not_satisfied_condition, self.attribute, (given_attributes_limits[0], self.threshold))
        elif given_attributes_limits[0] > self.threshold:
            # everything is satisfied:
            satisfied_condition = copy(input)
            not_satisfied_condition = None
        elif self.threshold >= given_attributes_limits[1]:
            # nothing  is satisfied:
            satisfied_condition = None
            not_satisfied_condition = copy(input)
        else:
            raise NotImplemented()
        return satisfied_condition, not_satisfied_condition


class WorkFlow:
    def __init__(self, raw_rules: str) -> None:
        _splitted = raw_rules.split(',')
        self.rules = [Rule(_raw) for _raw in _splitted]

    def transform(self, input: Rating) -> str:
        for rule in self.rules:
            if rule.is_accepted(input):
                return rule.escape_state

    def transform_interval(self, input: IntervalRating) -> list[tuple[IntervalRating, str]]:
        result = []
        _start_value = input
        for rule in self.rules:
            _pass, _fail = rule.transform_interval(_start_value)
            if _pass is not None:
                result.append((_pass, rule.escape_state))

            if _fail is None:
                # nothing more to compute
                break
            _start_value = _fail
        return result


def parse_workflows(data: list[str]) -> dict[str, WorkFlow]:
    _str_workflows = ''.join(data).split('\n\n')[0].split('\n')
    all_workflows = {}
    for plain_str in _str_workflows:
        workflow_name = plain_str.split('{')[0]
        rules = plain_str.split('{')[1].strip('}')
        all_workflows[workflow_name] = WorkFlow(rules)

    return all_workflows


class Resolution(Enum):
    ACCEPTED = 'A'
    REJECTED = 'R'


def compute_rating_resulution(rating: Rating, workflows: dict[str, WorkFlow]) -> Resolution:
    work_flow_id = 'in'
    _testing_path = [work_flow_id]
    while work_flow_id not in ('A', 'R'):
        work_flow_id = workflows[work_flow_id].transform(rating)
        _testing_path.append(work_flow_id)
        pass
    return Resolution(work_flow_id)


def solve_1(ratings: list[Rating], workflows: dict[str, WorkFlow]) -> int:
    values_of_accepted = []
    for rating in ratings:
        rating_resolution = compute_rating_resulution(rating, workflows)
        if rating_resolution == Resolution.ACCEPTED:
            values_of_accepted.append(rating.value)

    return sum(values_of_accepted)


def compute_interval_rating_resolution(rating: IntervalRating, workflows: dict[str, WorkFlow]) -> int:
    work_flow_id = 'in'
    accepted_states = []
    rejected_states = []
    open_states = [(rating, 'in')]
    while open_states:
        interval, work_flow_id = open_states.pop()
        _result = workflows[work_flow_id].transform_interval(interval)
        for new_val, flow_id in _result:
            if flow_id == 'A':
                accepted_states.append(new_val)
            elif flow_id == 'R':
                rejected_states.append(new_val)
            else:
                open_states.append((new_val,  flow_id))
        pass
    total_combinations_accepted = [state.possible_states for state in accepted_states]
    total_combinations_rejected = [state.possible_states for state in rejected_states]
    return sum(total_combinations_accepted)


def solve_2(workflows: dict[str, WorkFlow]) -> int:
    interval_rating = IntervalRating(x=(1, 4000), m=(1, 4000), a=(1, 4000), s=(1, 4000))
    return compute_interval_rating_resolution(interval_rating, workflows)


if __name__ == '__main__':
    raw_data = load_raw_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/19.txt')
    _ratings = parse_ratings(raw_data)
    _workflows = parse_workflows(raw_data)
    #s1 = solve_1(_ratings, _workflows)
    #print(s1)

    s2 = solve_2(_workflows)
    print(s2)