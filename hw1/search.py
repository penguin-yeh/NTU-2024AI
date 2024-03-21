"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

Follow the project description for details.

Good luck and happy searching!
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    print("Solution:", [s, s, w, s, w, w, s, w])
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    from util import Stack
    from game import Directions
    frontier = Stack()
    frontier.push(problem.getStartState())
    explored_set = []
    goal_path = []
    parents = {}
    actions = {}
    test = True
    
    while not frontier.isEmpty():  
        
        cur_node = frontier.pop()
        if problem.isGoalState(cur_node):
            print("finished!!")
            break # finish, find goal_node(which is cur_node)
        explored_set.append(cur_node)
        successor = problem.getSuccessors(cur_node)
        successor_len = len(successor)
        # print("current node: ", cur_node)
        
        for i in range(successor_len):
            #save parent-child relations
            # next = successor[successor_len - 1 - i]
            next = successor[i]
            # if next[0] == (1, 1):
            #     print("---", next)
            
            if len(next[0]) == 3:
                for i in range(len(next[0][1])):
                    if next[0][0] == next[0][1][i]:
                        new_tuple = ()
                        for j in range(4):
                            if j == i:
                                new_tuple += (1,)
                            else:
                                new_tuple += (cur_node[2][j],)
                        # print(new_tuple)
                        new_tuple_first = (next[0][0], next[0][1], new_tuple)
                        new_tuple_2 = (new_tuple_first, next[1], next[2])
                        # new_tuple_2 += (new_tuple,)
                        next = new_tuple_2
                        break
                    
            # if parents.get(next[0]) is None:
            #     parents[next[0]] = cur_node # only coordinates
            #     actions[next[0]] = next[1] # only save actions
            # print(problem.isGoalState(next))
            
            # print(next)
            next_tmp = next[0] # only coordinates left
            
            if parents.get(next_tmp) is None:
                parents[next_tmp] = cur_node # only coordinates
                # if len(next) == 5:
                #     # print(next_tmp)
                #     actions[next_tmp] = next[3] # only save actions
                #     # print(actions[next[0]])
                # else:
                actions[next_tmp] = next[1]
            # if len(next) == 5:
            #     next = (next[0], next[1], next[2]) # only coordinates left
            #     # print(actions[next[0]])
            # else:
            next = next[0]
            
            frontier_flag = False
            explored_flag = False
            
            # wouldn't work, since Python do deep 
            # tmp = frontier 
            tmp = [] # just for make a copy of frontier in list format
            
            while not frontier.isEmpty():
                tmp_ = frontier.pop()
                tmp.append(tmp_)
                # print(next)
            
            #recover frontier
            for j in range(len(tmp)):
                frontier.push(tmp[len(tmp) - 1 - j])
                # print(tmp[len(tmp) - 1 - i])
                        
            for item in tmp:
                if next == item:
                    frontier_flag = True
                    break
            for item in explored_set:
                if next == item:
                    explored_flag = True
                    break
            # print(frontier_flag)
            # print(explored_flag)
            if not frontier_flag and not explored_flag:
                frontier.push(next)
            # return
    # return 
        # print(problem.isGoalState((1,1)))
        # return
    # print('goal node:' ,  cur_node)
    # print(problem.isGoalState(cur_node))
    
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    path = 0
    # print(problem.getStartState())
    while cur_node != problem.getStartState():
        explored_flag = False
        if problem.isGoalState(cur_node):
            explored_flag = True
        else:
            for item in explored_set:
                if cur_node == item:
                    explored_flag = True
                    break
        if explored_flag is True:
            # print("currrr", cur_node)
            # print(actions[cur_node])
            # if actions[cur_node] == 'East':
            #     goal_path.insert(0, Directions.EAST)
            # elif actions[cur_node] == 'West':
            #     goal_path.insert(0, Directions.WEST)
            # if actions[cur_node] == 'South':
            #     goal_path.insert(0, Directions.SOUTH)
            # if actions[cur_node] == 'North':
            #     goal_path.insert(0, Directions.NORTH)
            goal_path.insert(0, actions[cur_node])
            cur_node = parents[cur_node]
            path += 1
        # print("cur_node:", cur_node)
        # cur_node = parents[cur_node]
        # print("cur_node:", cur_node)
        # cur_node = parents[cur_node]
        # print("cur_node:", cur_node)
        # return
    # print("my path len:", path)
    return goal_path
            
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    from game import Directions
    from eightpuzzle import EightPuzzleState
    frontier = Queue()
    frontier.push(problem.getStartState())
    explored_set = []
    goal_path = []
    parents = {}
    actions = {}
    test = True
    
    # if len(problem.getStartState()) == 3:
    #     corners = problem.getStartState()[1]
    #     record = problem.getStartState()[2]
    #     print("corners:", corners)
    #     print("record:", record)
    
    #Start:(34, 16)
    #successsor:[((34, 15), 'South', 1), ((33, 16), 'West', 1)]
    
    #Start:[(4, 5), ((1, 1), (1, 6), (6, 1), (6, 6)), [0, 0, 0, 0]]
    #successsor:[((4, 6), ((1, 1), (1, 6), (6, 1), (6, 6)), [0, 0, 0, 0], 'North', 1), ((5, 5), ((1, 1), (1, 6), (6, 1), (6, 6)), [0, 0, 0, 0], 'East', 1), ((3, 5), ((1, 1), (1, 6), (6, 1), (6, 6)), [0, 0, 0, 0], 'West', 1)]
    
    # print("start:", problem.getStartState())
    # print("len of start:", len(problem.getStartState()))
    # print(problem.getSuccessors(problem.getStartState()))
    # print(problem.isGoalState(problem.getStartState()))
    # print(problem.getSuccessors(problem.getStartState()))
    
    while not frontier.isEmpty():  
        cur_node = frontier.pop()
        if problem.isGoalState(cur_node):
            print("finished!!")
            break # finish, find goal_node(which is cur_node)
        explored_set.append(cur_node)
        successor = problem.getSuccessors(cur_node)
        successor_len = len(successor)
        # print("current node: ", cur_node)
        
        for i in range(successor_len):
            #save parent-child relations
            # next = successor[successor_len - 1 - i]
            next = successor[i]
            # print(next)
            # print(len(next))
            # if next[0] == (1, 1):
            #     print("---", next)
            # print(next)
            # print(len(next[0]))
            # print(type(next[0]))
            if len(next[0]) == 3:
                for i in range(len(next[0][1])):
                    if next[0][0] == next[0][1][i]:
                        new_tuple = ()
                        for j in range(4):
                            if j == i:
                                new_tuple += (1,)
                            else:
                                new_tuple += (cur_node[2][j],)
                        # print(new_tuple)
                        new_tuple_first = (next[0][0], next[0][1], new_tuple)
                        new_tuple_2 = (new_tuple_first, next[1], next[2])
                        # new_tuple_2 += (new_tuple,)
                        next = new_tuple_2
                        break
            
            # if len(next) == 5:
            #     next_tmp = (next[0], next[1], next[2]) # only coordinates left
            #     # print(actions[next[0]])
            # else:
            next_tmp = next[0]
            
            # print(len(next))
                                            
            if parents.get(next_tmp) is None:
                parents[next_tmp] = cur_node # only coordinates
                # if len(next) == 5:
                #     # print(next_tmp)
                #     print("yessss")
                #     actions[next_tmp] = next[3] # only save actions
                #     # print(actions[next[0]])
                # else:
                actions[next_tmp] = next[1]
            # print(problem.isGoalState(next))
            
            # if len(next) == 5:
            #     next = (next[0], next[1], next[2]) # only coordinates left
            #     # print(actions[next[0]])
            # else:
            next = next[0]
            


            
            frontier_flag = False
            explored_flag = False
            
            # wouldn't work, since Python do deep 
            # tmp = frontier 
            tmp = [] # just for make a copy of frontier in list format
            
            while not frontier.isEmpty():
                tmp_ = frontier.pop()
                tmp.append(tmp_)
                # print(next)
            
            #recover frontier
            for j in range(len(tmp)):
                frontier.push(tmp[j])
                # print(tmp[len(tmp) - 1 - i])
                        
            for item in tmp:
                if next == item:
                    frontier_flag = True
                    break
            for item in explored_set:
                if next == item:
                    explored_flag = True
                    break
            # print(frontier_flag)
            # print(explored_flag)
            if not frontier_flag and not explored_flag:
                frontier.push(next)
            # return
    # return 
        # print(problem.isGoalState((1,1)))
        # return
    # print('goal node:' ,  cur_node)
    # print(problem.isGoalState(cur_node))
    
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    path = 0
    # print(problem.getStartState())
    while cur_node != problem.getStartState():
        explored_flag = False
        if problem.isGoalState(cur_node):
            explored_flag = True
        else:
            for item in explored_set:
                if cur_node == item:
                    explored_flag = True
                    break
        if explored_flag is True:
            # print("currrr", cur_node)
            # print(actions[cur_node])
            # if actions[cur_node] == 'East':
            #     goal_path.insert(0, Directions.EAST)
            # elif actions[cur_node] == 'West':
            #     goal_path.insert(0, Directions.WEST)
            # if actions[cur_node] == 'South':
            #     goal_path.insert(0, Directions.SOUTH)
            # if actions[cur_node] == 'North':
            #     goal_path.insert(0, Directions.NORTH)
            goal_path.insert(0, actions[cur_node])
            cur_node = parents[cur_node]
            path += 1
        # print("cur_node:", cur_node)
        # cur_node = parents[cur_node]
        # print("cur_node:", cur_node)
        # cur_node = parents[cur_node]
        # print("cur_node:", cur_node)
        # return
    # print("my path len:", path)
    return goal_path
    
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # seems like to use PriorityQueue() instead of Stack/Queue to get lowest-priority item
    
    from util import PriorityQueue
    from game import Directions
    frontier = PriorityQueue()
    # start node
    frontier.push(problem.getStartState(), 99999)
    explored_set = []
    goal_path = []
    parents = {}
    prioritys = {}
    actions = {}
    # print(problem.getStartState())
    # print(problem.getStartState()[0])
    # print(problem.isGoalState(problem.getStartState()))
    
    while not frontier.isEmpty():
        
        # only pop lowest-priority coordinates
        cur_node = frontier.pop()
        if problem.isGoalState(cur_node):
            print("finished!!")
            break # finish, find goal_node(which is cur_node)
        explored_set.append(cur_node)
        successor = problem.getSuccessors(cur_node)
        successor_len = len(successor)
        
        for i in range(successor_len):
            # print("len", successor_len)
            next = successor[i]
            # print(next[1]) #actions
            
            if prioritys.get(next[0]) is not None:
                # print("Yes", prioritys[next[0]])
                prev_priority = prioritys[next[0]]
                new_priority = next[2]
                if prev_priority > new_priority:
                    prioritys[next[0]] = next[2]
                    parents[next[0]] = cur_node
                    actions[next[0]] = next[1]
            else:
                prioritys[next[0]] = next[2]
                parents[next[0]] = cur_node
                actions[next[0]] = next[1]
            
                
            next = next[0] # child, only coordinates left
            
            frontier_flag = False
            explored_flag = False
            
            tmp = [] # just for make a copy of frontier in list format
            
            while not frontier.isEmpty():
                tmp_ = frontier.pop()
                tmp.append(tmp_)
                # print(next)
            
            #recover frontier
            for j in range(len(tmp)):
                if tmp[j] == next:
                    frontier.push(tmp[j], prev_priority)
                else:
                    frontier.push(tmp[j], prioritys[tmp[j]])
                # print(tmp[len(tmp) - 1 - i])
            for item in tmp:
                if next == item:
                    frontier_flag = True
                    break
            for item in explored_set:
                if next == item:
                    explored_flag = True
                    break
            if not frontier_flag and not explored_flag:
                frontier.push(next, prioritys[next])
            elif frontier_flag == True:
                if prev_priority > new_priority:
                    frontier.update(next, new_priority)
    
    path = 0
    # print(problem.getStartState())
    first_flag = True
    while cur_node != problem.getStartState():
        if first_flag:
            explored_flag = True
            first_flag = False
        else:
            for item in explored_set:
                if cur_node == item:
                    explored_flag = True
                    break
        if explored_flag is True:
            # print("currrr", cur_node)
            # print(actions[cur_node])
            # if actions[cur_node] == 'East':
            #     goal_path.insert(0, Directions.EAST)
            # elif actions[cur_node] == 'West':
            #     goal_path.insert(0, Directions.WEST)
            # if actions[cur_node] == 'South':
            #     goal_path.insert(0, Directions.SOUTH)
            # if actions[cur_node] == 'North':
            #     goal_path.insert(0, Directions.NORTH)
            goal_path.insert(0, actions[cur_node])
            cur_node = parents[cur_node]
        path += 1
    # print("my path len:", path)
    
    # print(problem.getSuccessors(problem.getStartState()))    
    # # below is actions
    # print(problem.getSuccessors(problem.getStartState())[1][0])
     
    # print(problem.getSuccessors(problem.getSuccessors(problem.getStartState())[1][0]))
    # # below is priority
    # print(problem.getSuccessors(problem.getStartState())[1][2])
    # # below is next cooridvates/node(A or B ...)
    # print(problem.getSuccessors(problem.getStartState())[1][0][0])
    
    # print(problem.getCostOfActions(problem.getStartState()[1]))
    # print(problem.getSuccessors(problem.getStartState()))
    # frontier.push(problem.getStartState()[0], problem.getStartState()[1])
    # print(problem.getStartState()[1])

    return goal_path
    
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # print(heuristic(problem.getStartState(), problem))
    
    from util import PriorityQueue
    from game import Directions
    frontier = PriorityQueue()
    # start node
    frontier.push(problem.getStartState(), 0 + heuristic(problem.getStartState(), problem))
    explored_set = []
    goal_path = []
    parents = {}
    prioritys = {}
    actions = {}
    # print("start:", problem.getStartState())
    prioritys[problem.getStartState()] = 0
    # print(problem.getStartState())
    # print(problem.getStartState()[0])
    # print(problem.isGoalState(problem.getStartState()))
    
    while not frontier.isEmpty():
        
        # only pop lowest-priority coordinates
        cur_node = frontier.pop()
        # print("cur:", cur_node)
        # print(len(cur_node))

        if problem.isGoalState(cur_node):
            # print("goal h():", heuristic(cur_node, problem))
            print("finished!!")
            break # finish, find goal_node(which is cur_node)
        explored_set.append(cur_node)
        
        successor = problem.getSuccessors(cur_node)
        successor_len = len(successor)
        
        # print("current :", heuristic(cur_node, problem))
        for i in range(successor_len):
            # print("len", successor_len)
            next = successor[i]
            # print("successors :", heuristic(next[0], problem))
            # print("delta:", heuristic(cur_node, problem) - heuristic(next[0], problem))
            # if heuristic(cur_node, problem) - heuristic(next[0], problem) > 1:
            #     print("errorrrrrrr")
            # if cur_node[0] == (1, 12):
                # print(cur_node)
                # print(next, "->h():", heuristic(next[0], problem), "->g()", prioritys[cur_node])
            # if (heuristic(cur_node, problem) - heuristic(next[0], problem)) > 1:
            #     continue
            # print(next)
            # print(next[0])
            # print(len(next))
            
            # print("successors:", next)
            # print(next[1]) #actions
            # if len(next[0]) == 3:
            #     for i in range(len(next[0][1])):
            #         if next[0][0] == next[0][1][i]:
            #             new_tuple = ()
            #             for j in range(4):
            #                 if j == i:
            #                     new_tuple += (1,)
            #                 else:
            #                     new_tuple += (cur_node[2][j],)
            #             # print("newww:", new_tuple)
            #             # cur_node = cur_node[:-1]
            #             new_tuple_first = (next[0][0], next[0][1], new_tuple)
            #             new_tuple_2 = (new_tuple_first, next[1], next[2])
            #             # new_tuple_2 += (new_tuple,)
            #             # new_tuple_2 += (next[1], next[2],)
            #             # print("new:", new_tuple_2, "new")
            #             next = new_tuple_2
            #             break
            # if len(next[0]) == 3:
            #     next_tmp = (next[0]) # only coordinates left
            #     score = next[2]
            #     action_tmp = next[1]
            #     # print(actions[next[0]])
            # else:
            next_tmp = next[0]
            score = next[2]
            action_tmp = next[1]
            
            # print("suc h:", heuristic(next_tmp, problem))
            
            
            # print(next)
            if prioritys.get(next_tmp) is not None:
                # pass
                # print("Yes", prioritys[next[0]])
                prev_priority = prioritys[next_tmp]
                new_priority = prioritys[cur_node] + score 
                if prev_priority > new_priority:
                    pass
                    prioritys[next_tmp] = prioritys[cur_node] + score 
                    parents[next_tmp] = cur_node
                    actions[next_tmp] = action_tmp
            else:
                prioritys[next_tmp] = prioritys[cur_node] + score 
                parents[next_tmp] = cur_node
                actions[next_tmp] = action_tmp
            
            # if len(next) == 5:
            #     next = (next[0], next[1], next[2]) # only coordinates left
            # # print(actions[next[0]])
            # else:
            next = next[0]
            # next = next[0] # child, only coordinates left
            
            frontier_flag = False
            explored_flag = False
            
            tmp = [] # just for make a copy of frontier in list format
            
            while not frontier.isEmpty():
                tmp_ = frontier.pop()
                tmp.append(tmp_)
                # print(next)
            
            #recover frontier
            for j in range(len(tmp)):
                # if tmp[j] == next:
                #     frontier.push(tmp[j], prev_priority)
                # else:
                frontier.push(tmp[j], prioritys[tmp[j]] + heuristic(tmp[j], problem))
                # print(tmp[len(tmp) - 1 - i])
            for item in tmp:
                if next == item:
                    frontier_flag = True
                    break
            for item in explored_set:
                if next == item:
                    explored_flag = True
                    break
                
            if not frontier_flag and not explored_flag:
                frontier.push(next, prioritys[next] + heuristic(next, problem))
            elif frontier_flag == True:
                if prev_priority > new_priority:
                    frontier.update(next, new_priority + heuristic(next, problem))
                    # prioritys[next_tmp] = prioritys[cur_node] + score 
                    # parents[next_tmp] = cur_node
                    # actions[next_tmp] = action_tmp
            elif explored_flag == True:
                if prev_priority > new_priority:
                    explored_set.remove(next)
                    frontier.push(next, new_priority + heuristic(next, problem))
                    # prioritys[next_tmp] = prioritys[cur_node] + score 
                    # parents[next_tmp] = cur_node
                    # actions[next_tmp] = action_tmp
    
    path = 0
    # print(problem.getStartState())
    first_flag = True
    while cur_node != problem.getStartState():
        # print(cur_node)
        if first_flag:
            explored_flag = True
            first_flag = False
        else:
            for item in explored_set:
                if cur_node == item:
                    explored_flag = True
                    break
        if explored_flag is True:
            # print("currrr", cur_node)
            # print(actions[cur_node])
            # if actions[cur_node] == 'East':
            #     goal_path.insert(0, Directions.EAST)
            # elif actions[cur_node] == 'West':
            #     goal_path.insert(0, Directions.WEST)
            # if actions[cur_node] == 'South':
            #     goal_path.insert(0, Directions.SOUTH)
            # if actions[cur_node] == 'North':
            #     goal_path.insert(0, Directions.NORTH)
            goal_path.insert(0, actions[cur_node])
            cur_node = parents[cur_node]
        path += 1
        
        # print(cur_node)
        # cur_node = parents[cur_node]
        # print(cur_node)
        # cur_node = parents[cur_node]
        # print(cur_node)
        # cur_node = parents[cur_node]
        # return 
    return goal_path
    
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
