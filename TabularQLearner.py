import random
import numpy as np




class TabularQLearner:

    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9, epsilon = 0.98, epsilon_decay = 0.999, dyna=0):
        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.prev_state = 0
        self.prev_action = 0
        self.Q_Table = np.random.rand(states, actions)
        self.experience = []


    def best_action(self, s):
        return np.argmax(self.Q_Table[s])

    def train(self, s, r):
        # Receive new state s and new reward r. Update Q-table and return selected action.
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        # How will you know the previous state and action?
        new_action = self.best_action(s)
        if random.uniform(0, 1) < self.epsilon:
            new_action = random.randint(0, self.actions - 1)
            self.epsilon = self.epsilon_decay * self.epsilon



        self.Q_Table[self.prev_state, self.prev_action] = (1 - self.alpha) * self.Q_Table[self.prev_state,
        self.prev_action] + self.alpha * (r + self.gamma * self.Q_Table[s, new_action])

        self.experience.append((self.prev_state, self.prev_action, s, r))
        self.prev_state = s
        self.prev_action = new_action


        for i in range(self.dyna):
            sample = random.randint(0, len(self.experience) - 1)
            ps = self.experience[sample][0]
            pa = self.experience[sample][1]
            s = self.experience[sample][2]
            r = self.experience[sample][3]
            na = self.best_action(s)
            self.Q_Table[ps, pa] = (1 - self.alpha) * self.Q_Table[ps, pa] + self.alpha * (
                    r + self.gamma * self.Q_Table[s, na])

        return new_action

    def test (self, s):
        # Receive new state s.  Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # When testing, you probably do not want to take random actions... (What good would it do?)

        self.prev_state = s
        self.prev_action = self.best_action(s)

        return self.prev_action
