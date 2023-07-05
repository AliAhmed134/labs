# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': [],
#     'E': ['F'],
#     'F': [],
# }
#
# visited = set()
#
#
# def dfs(visited, graph, node,goal):
#     if node not in visited:
#         print(node)
#         visited.add(node)
#
#         if node!=goal:
#             for neighbor in graph[node]:
#                 print("Neighbors is :" + neighbor)
#                 dfs(visited,graph,neighbor,goal)
#
#
#         else:
#             print("Find Goal state "+ node)
#
#
# dfs(visited,graph,'A',"E")


graph={
    '5':['3','7'],
    '3':['2','4'],
    '7':['8'],
    '2':[],
    '4':['8'],
    '8':[],
}
print(graph)
visited1=[]
queue=[]

def bfs(visited,graph,node):
    visited.append(node)
    queue.append(node)


    while queue:

        first_node=queue.pop(0)
        print(first_node,end=" ")

        for neighbors in graph[first_node]:
            if neighbors not in visited:
                visited.append(neighbors)
                queue.append(neighbors)


print("Following Example is BFS")
bfs(visited1,graph,'5')
