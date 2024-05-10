import numpy as np
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import pyspiel


class BotAgent(rl_agent.AbstractAgent):
    """Agent class that wraps a bot.

    Note, the environment must include the OpenSpiel state in its observations,
    which means it must have been created with use_full_state=True.

    This is a simple wrapper that lets the RPS bots be interpreted as agents under
    the RL API.
    """

    def __init__(self, num_actions, bot, name="bot_agent"):
        assert num_actions > 0
        self._bot = bot
        self._num_actions = num_actions

    def restart(self):
        self._bot.restart()

    def step(self, time_step, is_evaluation=False):
        # If it is the end of the episode, don't select an action.
        if time_step.last():
            return
        _, state = pyspiel.deserialize_game_and_state(
            time_step.observations["serialized_state"])
        action = self._bot.step(state)
        probs = np.zeros(self._num_actions)
        probs[action] = 1.0
        return rl_agent.StepOutput(action=action, probs=probs)


#  This function is to evaluate the agents. Do not change.

def eval_agents(env, agents, num_players, num_episodes, verbose=False):
    """Evaluate the agent.

    Runs a number of episodes and returns the average returns for each agent as
    a numpy array.

    Arguments:
      env: the RL environment,
      agents: a list of agents (size 2),
      num_players: number of players in the game (for RRPS, this is 2),
      num_episodes: number of evaluation episodes to run.
      verbose: whether to print updates after each episode.
    """
    sum_episode_rewards = np.zeros(num_players)
    for ep in range(num_episodes):
        for agent in agents:
            # Bots need to be restarted at the start of the episode.
            if hasattr(agent, "restart"):
                agent.restart()
        time_step = env.reset()
        episode_rewards = np.zeros(num_players)
        while not time_step.last():
            agents_output = [
                agent.step(time_step, is_evaluation=True) for agent in agents
            ]
            action_list = [
                agent_output.action for agent_output in agents_output]
            time_step = env.step(action_list)
            episode_rewards += time_step.rewards
        sum_episode_rewards += episode_rewards
        if verbose:
            print(f"Finished episode {ep}, "
                  + f"avg returns: {sum_episode_rewards / (ep+1)}")

    return sum_episode_rewards / num_episodes


def print_roshambo_bot_names_and_ids(roshambo_bot_names):
    print("Roshambo bot population:")
    for i in range(len(roshambo_bot_names)):
        print(f"{i}: {roshambo_bot_names[i]}")


def create_roshambo_bot_agent(player_id, num_actions, bot_names, pop_id):
    name = bot_names[pop_id]
    # Creates an OpenSpiel bot with the default number of throws
    # (pyspiel.ROSHAMBO_NUM_THROWS). To create one for a different number of
    # throws per episode, add the number as the third argument here.
    bot = pyspiel.make_roshambo_bot(player_id, name)
    return BotAgent(num_actions, bot, name=name)


"""#The following functions are used to load the bots from the original RRPS competition."""

# Some basic info and initialize the population

# print(pyspiel.ROSHAMBO_NUM_BOTS)    # 43 bots
# print(pyspiel.ROSHAMBO_NUM_THROWS)  # 1000 steps per episode

# The recall is how many of the most recent actions are presented to the RL
# agents as part of their observations. Note: this is just for the RL agents
# like DQN etc... every bot has access to the full history.
RECALL = 20

# The population of 43 bots. See the RRPS paper for high-level descriptions of
# what each bot does.

print("Loading bot population...")
pop_size = pyspiel.ROSHAMBO_NUM_BOTS
print(f"Population size: {pop_size}")
roshambo_bot_names = pyspiel.roshambo_bot_names()
roshambo_bot_names.sort()
# print_roshambo_bot_names_and_ids(roshambo_bot_names)

bot_id = 0
roshambo_bot_ids = {}
for name in roshambo_bot_names:
    roshambo_bot_ids[name] = bot_id
    bot_id += 1

"""#Example showing how to load to agents from the RRPS bot population and evalute them against each other."""

# Example: create an RL environment, and two agents from the bot population and
# evaluate these two agents head-to-head.

# Note that the include_full_state variable has to be enabled because the
# BotAgent needs access to the full state.
env = rl_environment.Environment(
    "repeated_game(stage_game=matrix_rps(),num_repetitions=" +
    f"{pyspiel.ROSHAMBO_NUM_THROWS}," +
    f"recall={RECALL})",
    include_full_state=True)
num_players = 2
num_actions = env.action_spec()["num_actions"]
# Learning agents might need this:
# info_state_size = env.observation_spec()["info_state"][0]

# Create two bot agents
p0_pop_id = 0   # actr_lag2_decay
p1_pop_id = 1   # adddriftbot2
agents = [
    create_roshambo_bot_agent(0, num_actions, roshambo_bot_names, p0_pop_id),
    create_roshambo_bot_agent(1, num_actions, roshambo_bot_names, p1_pop_id)
]

# print("Starting eval run.")
# avg_eval_returns = eval_agents(env, agents, num_players, 10, verbose=True)

# print("Avg return ", avg_eval_returns)


class RLAgent(rl_agent.AbstractAgent):

    def __init__(self, num_actions, name="bot_agent"):
        assert num_actions > 0
        self._num_actions = num_actions  # 3
        self._opponent_history = []
        self._play_order = {}

    def _predict_next_move(self):
        # take the last ten actions from the opponent's histoty
        recent_history = "".join(map(str, self._opponent_history[-10:]))

        # update the count for this sequence in the play order dictionary
        if recent_history in self._play_order:
            self._play_order[recent_history] += 1
        else:
            self._play_order[recent_history] = 1

        # store potential next moves of agent
        potential_moves = []

        # iterate all possible actions to find potential moves
        for next_move in range(self._num_actions):
            # ignore the most previous action
            # take last nine and the new potential move
            potential_move = recent_history[1:] + str(next_move)
            potential_moves.append(potential_move)

        # store observed sequences and their counts
        observed_sequences = {}

        for move in potential_moves:
            # check if the potential move has been observed before
            if move in self._play_order:
                # add to observed_sequences
                observed_sequences[move] = self._play_order[move]

        # check if any sequences have been observed
        if observed_sequences:
            most_common_sequence = max(
                observed_sequences, key=observed_sequences.get)
            # the prediction is the last action in the most common sequence
            prediction = int(most_common_sequence[-1])
        else:
            # if there are no observed sequences, play randomly
            prediction = np.random.randint(self._num_actions)

        return prediction

    def _counter_action(self, action):
        # based on the rules of Rock-Paper-Scissors
        if action == 0:
            return 1
        elif action == 1:
            return 2
        elif action == 2:
            return 0

    def step(self, time_step, is_evaluation=False):
        # If it is the end of the episode, don't select an action.
        if time_step.last():
            return
        # Note: If the environment was created with include_full_state=True, then
        # game and state can be obtained as follows:

        game, state = pyspiel.deserialize_game_and_state(
            time_step.observations["serialized_state"])

        if len(state.history()) > 0:
            self._opponent_history.append(state.history()[-1])

        # if not enough data collected, play randomly
        if len(self._opponent_history) < 10:
            action = np.random.randint(self._num_actions)
        else:
            prediction = self._predict_next_move()
            action = self._counter_action(prediction)

        probs = np.ones(self._num_actions) / self._num_actions
        return rl_agent.StepOutput(action=action, probs=probs)


my_agent = RLAgent(3, name="yingjia_agent")
print(my_agent._num_actions)


def run_evaluations(num_evals):
    for p1_pop_id in range(43):
        agents = [
            my_agent,
            create_roshambo_bot_agent(
                1, num_actions, roshambo_bot_names, p1_pop_id)
        ]

        avg_eval_returns = eval_agents(
            env, agents, num_players, num_evals, verbose=False)
        print(f"Avg return for p1_pop_id {p1_pop_id}: ", avg_eval_returns)


run_evaluations(7)
