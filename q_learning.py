import numpy as np
import gym
import matplotlib.pyplot as plt

MAX_EPISODES = 1000


class QLearningSolver:
    """Class containing the Q-learning algorithm that might be used for different discrete environments."""

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        learning_rate: float = 0.1,
        discount: float = 0.9,
        epsilon: float = 0.1,
    ):
        # zaimplementowac tablice Q
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon

        self.q_table = np.random.uniform(
            low=-2, high=0, size=(observation_space, action_space)
        )

        self.episodes_cumulative_reward = []
        self.steps_per_episode = 0
        self.avg_rewards = []
        self.avg_reward_per_episode = 0
        self.avg_penalties_per_episode = 0

    def __call__(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return Q-value of given state and action."""
        return self.q_table[state, action]

    def update(self, state: np.ndarray, action: np.ndarray, new_q: float) -> None:
        """Update Q-value of given state and action."""
        self.q_table[state, action] = new_q

    def get_best_action(self, state: np.ndarray) -> np.ndarray:
        """Return action that maximizes Q-value for a given state."""
        return np.argmax(self.q_table[state, :])

    def get_max_q(self, state: np.ndarray) -> np.ndarray:
        """Return maximum Q-value for a given state."""
        return np.max(self.q_table[state, :])

    def train(self, env, train_episodes):
        penalties = 0
        train_steps = 0
        for episode in range(train_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            cumulative_reward = 0

            while not done and not truncated:
                if np.random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.get_best_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                max_future_q = self.get_max_q(next_state)
                current_q = self(state, action)
                new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                    reward + self.discount * max_future_q
                )
                self.update(state, action, new_q)
                state = next_state

                if reward == -10:
                    penalties += 1
                cumulative_reward += reward
                train_steps += 1

            self.avg_reward_per_episode += cumulative_reward
            self.avg_rewards.append(cumulative_reward / train_steps)
            self.episodes_cumulative_reward.append(cumulative_reward)
        self.steps_per_episode = train_steps / train_episodes
        self.avg_reward_per_episode = self.avg_reward_per_episode / train_episodes
        self.avg_penalties_per_episode = penalties / train_episodes

    def __repr__(self):
        """Elegant representation of Q-learning solver."""
        return f"""
        Q-learning solver
        
        observation_space: {self.observation_space},
        action_space: {self.action_space},
        learning_rate: {self.learning_rate},
        discount: {self.discount},
        epsilon: {self.epsilon}"""

    def __str__(self):
        # do wizualizacji
        return self.__repr__()


episodes = 1000
env = gym.make("Taxi-v3")
q_solver = QLearningSolver(
    env.observation_space.n, env.action_space.n, discount=0.9, learning_rate=0.9
)
q_solver.train(
    env,
    episodes,
)
env.close()

env = gym.make("Taxi-v3", render_mode="human")

state, _ = env.reset()
print("taxi")
done = False
truncated = False
while not done and not truncated:
    if np.random.random() > q_solver.epsilon:
        action = q_solver.get_best_action(state)
    else:
        action = np.random.randint(0, env.action_space.n)
    next_state, reward, done, truncated, info = env.step(action)
    env.render()
    state = next_state
    if reward == 20:
        print("reached the goal!")
        break
env.close()
