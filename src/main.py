import matplotlib.pyplot as plt
import numpy as np
import heapq

plt.ioff()
plt.ion()

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        return heapq.heappop(self.heap)

    def peek(self):
        return self.heap[0]
        return self.heap[0][1]

    def is_empty(self):
        return len(self.heap) == 0


def read_grid_from_file(file_path):
    grid = []
    starts_goals=[]
    lineNumber =0
    with open(file_path, 'r') as file:
        for line in file:
            lineNumber+=1
            if 2<lineNumber<6:
                line_split = line.strip().split()
                start = line_split[3]
                start = ( int(start[1:-1].split(",")[1]), int(start[1:-1].split(",")[0]))
                goal = line_split[6]
                goal = ( int(goal[1:-1].split(",")[1]), int(goal[1:-1].split(",")[0]))
                starts_goals.append([start,goal])
            if lineNumber>5:
                line_split = list(line.strip())
                if len(line_split)==0:
                    continue
                # Strip the newline character and split by spaces
                grid.append(line_split)
    return grid,starts_goals

def readGrid():
    # Example usage
    file_path = './src/maps/map1.txt'
    grid, start_goals = read_grid_from_file(file_path)

def plot_grid(ax,grid, path=None, start=None, goal=None):

    
    # Create a color map for the grid
    cmap = plt.cm.get_cmap('Greys').copy()
    cmap.set_under(color='white') # Free space color
    cmap.set_over(color='black') # Obstacle color
    grid_array = np.asarray(grid)
    #fig, ax = plt.subplots()
    # Plot the grid with respect to the upper left-hand corner
    ax.matshow(grid_array, cmap=cmap, vmin=0.1, vmax=1.0, origin='lower')
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1))
    ax.set_yticks(np.arange(-0.5, len(grid), 1))
    ax.set_xticklabels(range(0, len(grid[0])+1))
    ax.set_yticklabels(range(0, len(grid)+1))
    # Plot the path with direction arrows
    if path:
        for i in range(len(path) - 1):
            start_y,start_x = path[i]
            end_y, end_x  = path[i + 1]
            ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                head_width=0.3, head_length=0.3, fc='blue', ec='blue')
        # Plot the last point in the path
        ax.plot(path[-1][1], path[-1][0], 'b.')
        # Plot the start and goal points
        if start:
            ax.plot(start[1], start[0], 'go') # Start point in green
        if goal:
            ax.plot(goal[1], goal[0], 'ro') # Goal point in red
        
    #return fig


def plotTest():
    # Example usage
    grid = [
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', 'X', '.', '.', '.', '.', '.', 'X', '.', '.'],
    ['.', '.', '.', 'X', '.', '.', '.', 'X', '.', '.'],
    ['.', '.', '.', 'X', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', 'X', 'X', '.', 'X', 'X', 'X', '.'],
    ['.', '.', '.', '.', '.', '.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.', '.', 'X', '.', 'X', '.'],
    ['.', 'X', '.', 'X', '.', '.', 'X', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.', '.', 'X', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
    ]
    # Convert grid to numerical values for plotting
    # Free space = 0, Obstacle = 1
    grid_numerical = [[1 if cell == 'X' else 0 for cell in row] for row in grid]
    grid_numerical = np.flipud(grid_numerical)
    # Define start and goal positions
    start = (0, 0)
    goal = (9, 9)
    # Example path
    path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 3), (6, 4), (7, 5), (
    8, 5), (9, 6), (9, 7), (9, 8), (9, 9)]
    # Plot the grid and path
    f = plot_grid(grid_numerical, path=path, start=start, goal=goal)

def dijkstra(start, goal, grid_numerical):
    visited = set()
    #queue_to_visit = [start]
    queue_to_visit:PriorityQueue = PriorityQueue()
    costs = {}
    costs[start]=0
    queue_to_visit.push(start,0)
    for i in range(0,len(grid_numerical)):
        for j in range(0,len(grid_numerical[0])):
            state = (i,j)
            if state!=start:
                costs[state]=float('inf')
    visited.add(start)
    predecessor_map={}
    path = []
    curr = None
    counter = 0
    
    while queue_to_visit:
        cost_to_come, curr = queue_to_visit.pop()
        counter+=1
        if curr == goal:
            while True:
                path.append(curr)
                if curr == start:
                    path = path[::-1]
                    return path, counter
                curr = predecessor_map[curr]
        neighbors = get_neighbors(curr, grid_numerical)
        for n in neighbors:
            tentative_cost = cost_to_come+1
            if tentative_cost<costs[n]:
                queue_to_visit.push(n,tentative_cost)
                costs[n]=tentative_cost
                predecessor_map[n]=curr
            # if n not in visited:
            #     queue_to_visit.push(n,tentative_cost)
            #     costs[n]=tentative_cost
            #     visited.add(n)
            #     predecessor_map[n]=curr
            # else:
            #     if cost_to_come+1<costs[n]:
            #         queue_to_visit.push(n,tentative_cost)
            #         costs[n]=tentative_cost
            #         predecessor_map[n]=curr
    return None, counter

def uniform_cost_search(start, goal, grid_numerical):
    visited = set()
    queue_to_visit = [start]
    queue_to_visit = PriorityQueue()
    queue_to_visit.push(start,0)
    visited.add(start)
    pathway={}
    path = []
    costs = {}
    costs[start]=0

    curr = None
    counter = 0
    while queue_to_visit:
        cost_to_come, curr = queue_to_visit.pop()
        counter+=1
        if curr == goal:
            while True:
                path.append(curr)
                if curr == start:
                    path = path[::-1]
                    return path, counter
                curr = pathway[curr]
        neighbors = get_neighbors(curr, grid_numerical)
        for n in neighbors:
            if n not in visited:
                queue_to_visit.push(n,cost_to_come+1)
                costs[n]=cost_to_come+1
                visited.add(n)
                pathway[n]=curr
            else:
                if cost_to_come+1<costs[n]:
                    queue_to_visit.push(n,cost_to_come+1)
                    costs[n]=cost_to_come+1
                    pathway[n]=curr
    return None, counter

def breath_first_search(start, goal, grid_numerical):
    visited = set()
    queue_to_visit = [start]
    visited.add(start)
    pathway={}
    path = []
    curr = None
    counter = 0
    while queue_to_visit:
        curr = queue_to_visit.pop(0)
        counter+=1
        if curr == goal:
            while True:
                path.append(curr)
                if curr == start:
                    path = path[::-1]
                    return path, counter
                curr = pathway[curr]
        neighbors = get_neighbors(curr, grid_numerical)
        for n in neighbors:
            if n not in visited:
                queue_to_visit.append(n)
                visited.add(n)
                pathway[n]=curr
    return None, counter

def depth_first_search(start, goal, grid_numerical):
    visited = set()
    queue_to_visit = [start]
    visited.add(start)
    pathway={} # dictionary to store the path to previous node
    path = [] #final list of path
    curr = None
    counter = 0
    while queue_to_visit:
        curr = queue_to_visit.pop()
        counter+=1
        if curr == goal:
            while True:
                path.append(curr)
                if curr == start:
                    path = path[::-1]
                    return path, counter
                curr = pathway[curr]
                
        neighbors = get_neighbors(curr, grid_numerical)
        for n in neighbors:
            if n not in visited:
                queue_to_visit.append(n)
                visited.add(n)
                pathway[n]=curr
    return None,counter

def neighbors_four():
    return [[-1,0], #up
            [0,1], #right
            [1,0], #down
            [0,-1] #left
            ]

def neighbors_8():
    return [[-1,0,0], #up
            [0,1,0],  #right
            [1,0,0],  #down
            [0,-1,0], #left
            
            [-1,1,1], #right-up
            [1,1,1],  #right-down
            [1,-1,1], #left-down
            [-1,-1,1] #left-up
            ]


def get_neighbors(curr, grid):
    possible_neighbors = neighbors_8()
    
    #possible_neighbors = neighbors_four()
    #possible_neighbors.reverse()
    neighbors = []
    for pn in possible_neighbors:
        row = curr[0] + pn[0]
        col = curr[1] + pn[1]
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == 0:
            if pn[2]==1:
                if grid[curr[0]+pn[0]][curr[1]]==0 and grid[curr[0]][curr[1]+pn[1]]==0:
                    neighbors.append((row, col))
            else:
                neighbors.append((row, col))
    return neighbors

def run_algo(algorithm,start,goal,grid_numerical):
    # grid, start_goals = read_grid_from_file(mapFile)
    # grid_numerical = [[1 if cell == 'X' else 0 for cell in row] for row in grid]
    # grid_numerical = np.flipud(grid_numerical)
    # for start,goal in start_goals:
    #     #start_flip = (len(grid_numerical)-1-start[0],start[1])
    #     #goal_flip = (len(grid_numerical)-1-goal[0],goal[1])
    # start_flip = start
    # goal_flip = goal
    path,counter = algorithm(start,goal,grid_numerical)
    #plot_grid(grid_numerical,path,start,goal)
    return path,counter


def plot_metrics(algos: dict,metric: str,xLabel,yLabel,title):
    barWidth = 0.25
    # only first map
    goalsInFirstMap = len(algos[list(algos.keys())[0]]["stats"])
    bar_x=[]
    br1 = np.arange(goalsInFirstMap) 
    bar_x.append(br1)
    br2 = [x + barWidth for x in br1] 
    bar_x.append(br2)
    br3 = [x + barWidth for x in br2] 
    bar_x.append(br3)
    br4 = [x + barWidth for x in br3] 
    bar_x.append(br4)

    for rangeIndex in range(0,goalsInFirstMap):
        fig, ax = plt.subplots()
        barIndex = 0
        for algo in algos:
            stat = algos[algo]["stats"][rangeIndex]
            #plt.plot(stat[metric], label=f"{algo} on {stat['map']}")
            # plt.bar(np.arange(len(stat[metric])),stat[metric],label=f"{algo} on {stat['map']}",width=barWidth)
            plt.bar(bar_x[barIndex],stat[metric],label=f"{algo} on {stat['map']}",width=barWidth)
            barIndex+=1
        # ax.set_xticks(np.arange(-1.0, goalsInFirstMap, 1))
        plt.xticks([r + barWidth for r in range(len(stat[metric]))],['1', '2', '3'])
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.legend()
        plt.show()

def run_single_algo(map,algo):
    grid, start_goals = read_grid_from_file(map)
    grid_numerical = [[1 if cell == 'X' else 0 for cell in row] for row in grid]
    grid_numerical = np.flipud(grid_numerical)
    for start,goal in start_goals:
        run_algo(algo,start,goal,grid_numerical)

def pathPlanningAnalysis():
    algos = setup()
    # Uncomment to run only one algorithm
    # algos = {"BFS":algos.get("BFS")}
    import time
    for algo in algos:
        mapCount = len(algos[algo]["stats"])
        
        grid, start_goals = read_grid_from_file(algos[list(algos.keys())[0]]["stats"][0]["map"])
        goals = len(start_goals)
        fig, axs = plt.subplots(mapCount, goals)
        mapIndex = 0
        
        for stat in algos[algo]["stats"]:
            print(f"Running {algo} on {stat['map']}")
            grid, start_goals = read_grid_from_file(stat['map'])
            grid_numerical = [[1 if cell == 'X' else 0 for cell in row] for row in grid]
            grid_numerical = np.flipud(grid_numerical)
            goalIndex = 0
            for start,goal in start_goals:
                start_time = time.time()
                path, visited_states_count = run_algo(algos[algo]["algorithm"],start,goal,grid_numerical)
                end_time = time.time()
                execution_time = end_time - start_time
                
                if stat.get("path_length") is None:
                    stat["path_length"] = []
                stat["path_length"].append(len(path))
                plot_grid(axs[mapIndex,goalIndex],grid_numerical,path,start,goal)    
                
                if stat.get("execution_times") is None:
                    stat["execution_times"] = []
                stat["execution_times"].append(execution_time)
                
                if stat.get("visited_states_count") is None:
                    stat["visited_states_count"] = []
                stat["visited_states_count"].append(visited_states_count)
                
                print(f"Execution Time: {execution_time} seconds")
                print(f"Visited states count: {visited_states_count}")
                goalIndex+=1
            mapIndex+=1
        fig.suptitle(f"{algo}")
        plt.show()
                #return algos
    




    return algos
    file_path = './src/maps/map1.txt'
    #run_algo(file_path,breath_first_search)
    #run_algo(file_path,depth_first_search)
    
    
    file_path = './src/maps/map2.txt'
    run_algo(file_path,breath_first_search)
    #run_algo(file_path,depth_first_search)

    file_path = './src/maps/map3.txt'
    run_algo(file_path,breath_first_search)
    #run_algo(file_path,depth_first_search)
    
    plt.show()


def setup():
    algorithms = {
        "BFS": {
            "algorithm": breath_first_search,
            "stats": 
                [
                    {
                    "map": "./src/maps/map1.txt",
                    "execution_times": [],
                    "visited_states_count": [],
                    "memory_usage": []
                    },
                    {
                    "map": "./src/maps/map2.txt",
                    },
                    {
                    "map": "./src/maps/map3.txt",
                    }

                ]
         
        },
        "DFS": {
            "algorithm": depth_first_search,
            "stats": 
                [
                    {
                    "map": "./src/maps/map1.txt",
                    },
                    {
                    "map": "./src/maps/map2.txt",
                    },
                    {
                    "map": "./src/maps/map3.txt",
                    }

                ]
            },
        "Dijkstras": {
            "algorithm": dijkstra,
            "stats": 
                [
                    {
                    "map": "./src/maps/map1.txt",
                    },
                    {
                    "map": "./src/maps/map2.txt",
                    },
                    {
                    "map": "./src/maps/map3.txt",
                    }

                ]
            },
        "Uniform Cost Search": {
            "algorithm": uniform_cost_search,
            "stats": 
                [
                    {
                    "map": "./src/maps/map1.txt",
                    },
                    {
                    "map": "./src/maps/map2.txt",
                    },
                    {
                    "map": "./src/maps/map3.txt",
                    }

                ]
            }
        }
    
    
    return algorithms

if __name__ == "__main__":
    # readGrid()
    # plotTest()
    #run_single_algo('./src/maps/map3.txt',dijkstra)
    #run_single_algo('./src/maps/map3.txt',uniform_cost_search)
    
    algos = pathPlanningAnalysis()
    
    plot_metrics(algos,"execution_times","Goal # (0-indexed)","Time (s)","Execution Time")
    plot_metrics(algos,"visited_states_count","Goal # (0-indexed)","Visited States Count","# of States Visited")
    plot_metrics(algos,"path_length","Goal # (0-indexed)","Path Length","Length of Path")
    for key in algos:
        algos[key]["algorithm"] = algos[key]["algorithm"].__name__
    import json
    with open("results.txt", "w") as f:
        f.write(json.dumps(algos, indent=4))    
    print("Press any key to exit")
    plt.ioff()
    plt.show()
    import sys
    # sys.stdin.read(1)
    