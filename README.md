# Pacman Search Project & Pacman AI
### Purpose of Project
탐색 알고리즘(BFS, DFS, Greedy)과 MDP and Value iteration, Q-learning을 이용하여 Pacman AI 구현을 완성하는 Mid & Final Term project(2022-1 CSE4007). 
- lecture note from pf: https://sites.google.com/hanyang.ac.kr/2021ai
- Pacman and autograding is technically based on http://ai.berkeley.edu.
​<p align="center">  
<img src="https://user-images.githubusercontent.com/76833405/216820615-29f412c9-755f-4a31-94c3-80aa2bf39901.gif">*Pacman search in mediumScaryMaze*
</p>


# Project Task description

### Pacman Search Project
1. Task1, Finding a Fixed Food Dot using Depth First Search
2. Task2, Breadth First Search
3. Task3, Varying the Cost Function
4. Task4, Greedy Algorithm

### Pacman AI
1. Task1, MDP and Value Iteration
2. Task2, Q-Learning
3. Task3, Q-Learning for Pacman

# How to start
### Pacman Search Project
1. Task1, DFS
Implement the depth-first search (DFS) algorithm in the depthFirstSearch function in search.py. Code should quickly find a solution for below:

```console
python2.7 pacman.py -l tinyMaze -p SearchAgent
python2.7 pacman.py -l mediumMaze -p SearchAgent
python2.7 pacman.py -l bigMaze -z .5 -p SearchAgent
```

2. Task2, BFS
To write a graph search algorithm that avoids expanding any already visited states.

```console
python2.7 pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python2.7 pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
```

3. Task3, Varying the Cost Function
By changing the cost function, we can encourage Pacman to find different paths. Implement the uniform-cost graph search algorithm in the uniformCostSearch function in search.py.

```console
python2.7 pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python2.7 pacman.py -l mediumDottedMaze -p StayEastSearchAgent
python2.7 pacman.py -l mediumScaryMaze -p StayWestSearchAgent
```

4. Greedy Algorithm
Write an agent that always greedily eats the closest dot. Implement the function findPathToClosestDot in searchAgents.py. Our agent solves this maze (suboptimally!) in under a second with a path cost of 350:

```console
python2.7 pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5
```

### Pacman AI
1. Task1, MDP and Value Iteration
Write a value iteration agent in ValueIterationAgent, which has been partially specified for you in valueIterationAgents.py.

2. Task2, Q-Learning
You will now write a Q-learning agent, which does very little on construction, but instead learns by trial and error from interactions with the environment through its update(state, action, nextState, reward) method.

3. Task3, Q-Learning for Pacman
If you want to watch 10 training games to see what's going on, use the command:

```console
python2.7 pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10
```

### Update
2022.06 Project 완료
2023.02 Project Decription update