from collections import defaultdict

from notebooks.aoc_2023.utils import load_stripped_lines


def load_game(line: str):
    pass
    game_id = line.split(':')[0].split(' ')[1]
    games = line.split(':')[1].split(';')
    values = defaultdict(list)
    for game in games:
        every_value = [value.strip().split(' ') for value in game.split(',')]
        for value in every_value:
            values[value[1]].append(int(value[0]))

    return game_id, values


def game_is_possible(game, max_values) -> bool:
    for colour, value in max_values.items():
        if max(game[1][colour]) > value:
            return False
    return True


def solve_1(games):
    max_values = {'green': 13, 'red': 12, 'blue': 14}
    is_possible = [game for game in games if game_is_possible(game, max_values)]
    print(sum([int(game[0]) for game in is_possible]))


def compute_min_values(game):
    game_state = game[1]
    min_state = {key: max(values) for key,values in game_state.items()}
    return min_state


def solve_2(games):
    min_values = [compute_min_values(game) for game in games]
    result = 0
    for one_game in min_values:
        a = one_game['green']*one_game['blue']*one_game['red']
        result += a
    print(result)


if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/input_2.txt')
    games = [load_game(line) for line in data]
    solve_1(games)
    solve_2(games)
    pass