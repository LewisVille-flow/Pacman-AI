#-*- coding: utf-8 -*-

# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        # 인자로 넘겨지는 것: mdp, opts.discount = 0.9, opts.iters = 100

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # mdp problem -> need to construct and get all values ahead(off-line)
        # value iteration eq:
        # V(k+1) = max(a) (sum w.r.t s' Transition *(reward + r * V(k))  # is in lecture pdf 11 50/52


        sPrime = self.mdp.getStates()

        for i in range(self.iterations):
            tempValues = util.Counter()

            for sP in sPrime:
                max_sum_s = -999999
                a = mdp.getPossibleActions(sP)

                for action in a:
                    s = self.mdp.getTransitionStatesAndProbs(sP, action)
                    # s = (nextState, prob)

                    sum_s = 0.0

                    for nextState, prob in s:
                        sum_s += prob * (self.mdp.getReward(sP, action, nextState) + self.discount * self.values[nextState])    
                    
                    max_sum_s = max(max_sum_s, sum_s)

                if max_sum_s != -999999:
                    tempValues[sP] = max_sum_s
                    
            for sP in sPrime:
                self.values[sP] = tempValues[sP]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        "*** YOUR CODE HERE ***"
        # Q value = sum w.r.t s' of (Transition *(reward + r*Value))
        # this eq. is in lecture pdf 11, 28/52

        # Transition = Returns list of (nextState, prob) pairs 
        #               representing the states reachable
        #               from 'state' by taking 'action' along
        #               with their transition probabilities


        sPrime = self.mdp.getTransitionStatesAndProbs(state, action)
        # sPrime = (nextState, prob)

        qValue = 0.0

        for nextState, prob in sPrime:
            # Q value eqaution is again,
            # Q value = sum w.r.t s' of (Transition *(reward + r*Value))
            # T = prob, reward = self.mdp.getReward, r = discount, value = values(nextstate)

            qValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState]) 
        
        return qValue

    def computeActionFromValues(self, state):
        # task: need to computes the best action according to the value function 
        # given by self.values.
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # policy iteration 단계에서 policy extraction을 진행: one step look ahead.

        possibleActions = self.mdp.getPossibleActions(state) # Returns list of possible actions from 'state'.

        outPolicy = None
        maxqvalue = -999999

        if self.mdp.isTerminal(state):                      # terminal case
            return None

        # outpolicy = argmax{sum w.r.t s' of (reward + r*Value), which is the q-value}
        # this eq. is in lecture pdf12, 37/47
        for actions in possibleActions:
           qValue = self.computeQValueFromValues(state, actions)

           if qValue > maxqvalue:
                maxqvalue = qValue
                outPolicy = actions 

        return outPolicy

        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
