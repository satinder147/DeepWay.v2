import numpy as np
import heapq


def heuristic(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def a_star(start, end, grid):
    neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    heap = []
    closed_list = set()
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    parent = {}
    heapq.heappush(heap, (f_score[start], start))
    while heap:
        current = heapq.heappop(heap)[1]
        closed_list.add(current)
        if current == end:
            data = []
            while current in parent:
                data.append(tuple(current))
                current = parent[current]
            return data
        for i, j in neighbours:
            neighbour = current[0] + i, current[1] + j
            tg_score = g_score[current] + heuristic(current, neighbour)
            if 0 <= neighbour[0] < grid.shape[0]:
                if 0 <= neighbour[1] < grid.shape[1]:
                    if grid[neighbour[0]][neighbour[1]]:
                        continue
                else:
                    continue
            else:
                continue
            if neighbour in closed_list and tg_score >= g_score.get(neighbour, 0):
                continue
            if tg_score < g_score.get(neighbour, 0) or neighbour not in [i[1] for i in heap]:
                parent[neighbour] = current
                g_score[neighbour] = tg_score
                f_score[neighbour] = tg_score + heuristic(neighbour, end)
                heapq.heappush(heap, (f_score[neighbour], neighbour))

    return False
