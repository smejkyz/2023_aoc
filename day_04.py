from collections import defaultdict

from notebooks.aoc_2023.utils import load_stripped_lines


def load_card(line: str):
    splitted = line.split(' | ')
    winnings_numbers = [int(item) for item in splitted[0].split(':')[-1].split( )]
    my_numbers = [int(item) for item in splitted[1].split( )]
    return winnings_numbers, my_numbers


def solve_1(cards: list[tuple[list[int], list[int]]]):
    wininnings = []
    for winnings_numbers, my_numbers in cards:
        nb_winning_mine = [number for number in my_numbers if number in winnings_numbers]
        points = 2**(len(nb_winning_mine) - 1) if nb_winning_mine else 0
        wininnings.append(points)

    return sum(wininnings)


def solve_2(cards: list[tuple[list[int], list[int]]]):
    nb_cards = {i: 1 for i in range(len(cards))}

    for i, (winnings_numbers, my_numbers) in enumerate(cards):
        nb_winning_mine = len([number for number in my_numbers if number in winnings_numbers])
        for id_card in range(i+1, i+1+nb_winning_mine):
            nb_cards[id_card] += nb_cards[i]

    return sum(nb_cards.values())


if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/04.txt')
    cards = [load_card(line) for line in data]
    print(solve_1(cards))

    print(solve_2(cards))
    pass