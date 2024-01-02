import abc
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum

from notebooks.aoc_2023.utils import load_stripped_lines


class Pulse(Enum):
    LOW = 'low'
    HIGH = 'high'


class State(Enum):
    OFF = 'off'
    ON = 'on'


@dataclass
class Signal:
    pulse: Pulse


class Module(ABC):
    def __init__(self, _raw: str) -> None:
        _splitted = _raw.split('->')[1].split(',')
        self.destinations_modules = [val.strip() for val in _splitted]
        self.name = _raw.split(' ')[0][1:]

    def __repr__(self) -> str:
        return f'{Module.__name__}(name={self.name}, destinations={self.destinations_modules})'

    @abc.abstractmethod
    def init_state_module(self, *args) -> None:
        pass

    @abc.abstractmethod
    def receive_and_broadcast(self, signal: Signal, source: str, nb_pushed: int) -> Signal | None:
        pass


class FlipFlopModule(Module):

    def init_state_module(self, *args) -> None:
        self.state = State.OFF

    def receive_and_broadcast(self, input_signal: Signal, source: str, nb_pushed: int) -> Signal | None:

        if input_signal.pulse == Pulse.HIGH:
            # nothing happened
            return None

        elif input_signal.pulse == Pulse.LOW:
            if self.state == State.ON:
                self.state = State.OFF
                return Signal(pulse=Pulse.LOW)
            elif self.state == State.OFF:
                self.state = State.ON
                return Signal(pulse=Pulse.HIGH)
            else:
                raise NotImplemented()
        else:
            raise NotImplemented()


class ConjunctionModule(Module):

    def receive_and_broadcast(self, signal: Signal, source: str, nb_pushed: int) -> Signal | None:
        # update memory for input:
        self.input_modules_most_recent_pulse[source] = signal.pulse
        if self.name == 'gf' and not all(value == Pulse.LOW for value in self.input_modules_most_recent_pulse.values()):
            print(f'{nb_pushed=} : {self.input_modules_most_recent_pulse}')
        if all(value == Pulse.HIGH for value in self.input_modules_most_recent_pulse.values()):
            return Signal(pulse=Pulse.LOW)
        else:
            return Signal(pulse=Pulse.HIGH)

    def init_state_module(self, *args) -> None:
        self.input_modules_most_recent_pulse: dict[str, Pulse] = {_input_module: Pulse.LOW for _input_module in args}


class BroadCastModule(Module):

    def init_state_module(self, *args) -> None:
        pass

    def receive_and_broadcast(self, signal: Signal, source: str, nb_pushed: int) -> Signal | None:
        return signal


def parse_connections(input_data: list[str]) -> dict[str, list[str]]:
    connections = {}
    for line in input_data:
        _splitted = line.split('->')[1].split(',')
        name = line.split(' ')[0][1:] if 'broadcaster' not in line else 'broadcaster'
        destinations_modules = [val.strip() for val in _splitted]
        connections[name] = destinations_modules
    return connections


def parse_modules(input_data: list[str]) -> dict[str, Module]:
    _modules = {}
    for line in input_data:
        if 'broadcaster' in line:
            _modules['broadcaster'] = BroadCastModule(line)
        else:
            if line[0] == '%':
                name = line.split(' ')[0][1:]
                _modules[name] = FlipFlopModule(line)
            elif line[0] == '&':
                name = line.split(' ')[0][1:]
                _modules[name] = ConjunctionModule(line)
            else:
                raise NotImplemented()

    return _modules


class System:

    def __init__(self, modules: dict[str, Module], connections: dict[str, list[str]]):
        self.connections = connections
        self.modules = modules
        self.number_of_send_signals: dict[Pulse, int] = {Pulse.LOW: 0, Pulse.HIGH: 0}
        self.nb_pushes = 0
        for module in self.modules.values():
            if not isinstance(module, ConjunctionModule):
                module.init_state_module()
            else:
                # find all input modules for Conjunction module
                input_modules = [_md.name for _md in self.modules.values() if module.name in _md.destinations_modules]
                module.init_state_module(*input_modules)

    def run(self):
        nb_push_button = 1000
        while True:
            self.number_of_send_signals[Pulse.LOW] += 1  # push is counted as low
            self.nb_pushes += 1
            self.push_button()
            if self.nb_pushes % 100000 == 0:
                print(f'button pushed {self.nb_pushes} times')

        #
        # print('first push: ')
        # self.push_button()
        # print(f'second push: ')
        # self.push_button()
        # print(f'third push: ')
        # self.push_button()
        # print(f'fourth push: ')
        # self.push_button()

    def push_button(self):
        # self.modules['broadcaster'].receive(signal)
        _front_open_to_solve = [('broadcaster', Signal(pulse=Pulse.LOW))]
        while _front_open_to_solve:
            id_module, signal_to_send = _front_open_to_solve.pop(0)
            for destination_module_id in self.connections[id_module]:
                # print(f'{id_module} -{signal_to_send.pulse} -> {destination_module_id}')
                self.number_of_send_signals[signal_to_send.pulse] += 1
                if destination_module_id not in self.modules:
                    #print(f'warning: module id {destination_module_id} does not exists - signal with {signal_to_send.pulse} sinks')
                    if signal_to_send.pulse == Pulse.LOW:
                        print(f'End of computation, received low pulse after: {self.nb_pushes}')
                        raise ValueError()
                    continue
                output_signal = self.modules[destination_module_id].receive_and_broadcast(signal_to_send, id_module, self.nb_pushes)
                if output_signal is not None:
                    _front_open_to_solve.append((destination_module_id, output_signal))

            pass


if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/20.txt')
    _mod = parse_modules(raw_data)
    _connections = parse_connections(raw_data)
    print(3761 * 3767 * 4001 * 4091)
    system = System(_mod, _connections)
    system.run()

#    s1 = system.number_of_send_signals[Pulse.LOW] * system.number_of_send_signals[Pulse.HIGH]
    print(s1)
