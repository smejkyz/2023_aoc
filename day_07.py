from collections import Counter
from enum import Enum

from typing_extensions import Self

from notebooks.aoc_2023.utils import load_stripped_lines

RANK_MAP = {
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'T': 10,
    'J': 11,
    'Q': 12,
    'K': 13,
    'A': 14,
}

RANK_MAP_WITH_JOKER = {
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'T': 10,
    'J': 1,
    'Q': 12,
    'K': 13,
    'A': 14,
}


class Type(Enum):
    FIVE_OF_A_KIND = 6
    FOUR_OF_A_KIND = 5
    FULL_HOUSE = 4
    THREE_OF_A_KIND = 3
    TWO_PAIRS = 2
    ONE_PAIR = 1
    HIGH_CARD = 0


class Hand:
    def __init__(self, raw_str: str, rank_map: dict[str, int], apply_joker: bool = False) -> None:
        cards, bid = raw_str.split(' ')
        self.bid = int(bid)
        self.cards = list(cards)
        self.rank_map = rank_map
        self.apply_joker = apply_joker

    @property
    def type(self) -> Type:
        if not self.apply_joker or "J" not in self.cards:
            return self.compute_type(self.cards)
        # define strategy based on number of jokers:
        _counter = Counter(self.cards)
        nb_jokers = _counter["J"]
        if (
            nb_jokers in (4, 5)
            or (nb_jokers == 3 and len(_counter) == 2)
            or (nb_jokers == 2 and set(_counter.values()) == {2, 3})
            or (nb_jokers == 1 and set(_counter.values()) == {4, 1})
        ):
            return Type.FIVE_OF_A_KIND
        if (nb_jokers == 3 and len(_counter) == 3) or (nb_jokers == 2 and len(_counter) == 3) or (nb_jokers == 1 and set(_counter.values()) == {1, 3}):
            return Type.FOUR_OF_A_KIND
        if (nb_jokers == 2 and len(_counter) == 4) or (nb_jokers == 1 and len(_counter) == 4):
            return Type.THREE_OF_A_KIND
        if nb_jokers == 1 and set(_counter.values()) == {1, 2} and len(_counter) == 3:
            return Type.FULL_HOUSE
        if nb_jokers == 1 and len(_counter) == 5:
            return Type.ONE_PAIR
        raise NotImplemented()

    @staticmethod
    def compute_type(cards: list[str]) -> Type:
        _counter = Counter(cards)
        if len(_counter) == 1:
            return Type.FIVE_OF_A_KIND
        if len(_counter) == 2 and set(_counter.values()) == {4, 1}:
            return Type.FOUR_OF_A_KIND
        if len(_counter) == 2 and set(_counter.values()) == {2, 3}:
            return Type.FULL_HOUSE
        if len(_counter) == 3 and set(_counter.values()) == {1, 3}:
            return Type.THREE_OF_A_KIND
        if len(_counter) == 3 and set(_counter.values()) == {1, 2}:
            return Type.TWO_PAIRS
        if len(_counter) == 4:
            return Type.ONE_PAIR
        if len(_counter) == 5:
            return Type.HIGH_CARD
        raise NotImplemented()

    def __repr__(self) -> str:
        return f'Hand({self.cards}, type: {self.type})'

    def __lt__(self, other: Self):
        if self.type != other.type:
            return self.type.value < other.type.value
        for card_self, card_other in zip(self.cards, other.cards):
            if card_self != card_other:
                return self.rank_map[card_self] < self.rank_map[card_other]


def solve_1(_hands: list[Hand]) -> int:
    sorted_hands = sorted(_hands)
    winnings = [(i+1)*hand.bid for i, hand in enumerate(sorted_hands)]

    return sum(winnings)


if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/07.txt')
    hands = [Hand(line, RANK_MAP) for line in data]
    assert solve_1(hands) == 246795406
    hands_2 = [Hand(line, RANK_MAP_WITH_JOKER, apply_joker=True) for line in data]
    assert solve_1(hands_2) == 249356515

    #792