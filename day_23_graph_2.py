from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def dfs_util(self, vertex, visited, depth):
        visited[vertex] = True
        current_depth = depth

        for neighbor in self.graph[vertex]:
            if not visited[neighbor]:
                current_depth = max(
                    current_depth, self.dfs_util(neighbor, visited, depth + 1)
                )

        visited[vertex] = False  # Reset visited for backtracking
        return current_depth

    def longest_path(self):
        num_vertices = len(self.graph)
        visited = [False] * num_vertices
        max_path_length = 0

        for vertex in range(num_vertices):
            if not visited[vertex]:
                max_path_length = max(
                    max_path_length, self.dfs_util(vertex, visited, 0)
                )

        return max_path_length


# Example usage:
g = Graph()
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(1, 3)
g.add_edge(3, 4)
g.add_edge(4, 1)

print("Longest Path Length:", g.longest_path())
