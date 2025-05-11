import argparse
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##############################
# MINIMAX WITH ALPHA-BETA PRUNING
##############################

minimax_cache = {}

def get_valid_moves(board):
    """Return a list of column indices where a move can be made (i.e. top cell is empty)."""
    return [c for c in range(board.shape[1]) if board[0, c] == 0]

def drop_piece(board, col, piece):
    """Return a new board with piece dropped in the given column.
       Assumes the move is valid."""
    new_board = board.copy()
    rows = board.shape[0]
    for r in range(rows - 1, -1, -1):
        if new_board[r, col] == 0:
            new_board[r, col] = piece
            break
    return new_board

def winning_move(board, piece):
    """Check whether the given piece has a winning connect-4 in the board."""
    rows, cols = board.shape
    # Check horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if np.all(board[r, c:c + 4] == piece):
                return True
    # Check vertical
    for c in range(cols):
        for r in range(rows - 3):
            if np.all(board[r:r + 4, c] == piece):
                return True
    # Check positively sloped diagonals
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r + i, c + i] == piece for i in range(4)):
                return True
    # Check negatively sloped diagonals
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(board[r - i, c + i] == piece for i in range(4)):
                return True
    return False

def terminal_node(board):
    """Return True if board is terminal: win for any player or full board."""
    return winning_move(board, 1) or winning_move(board, -1) or len(get_valid_moves(board)) == 0

def evaluate_board(board, player):
    """
    Simple evaluation function.
    If the given player wins, return a high positive score;
    if the opponent wins, return a high negative score.
    Otherwise, use a simple heuristic (center control).
    """
    opp = -player
    if winning_move(board, player):
        return 1000
    elif winning_move(board, opp):
        return -1000
    else:
        # Extra weight to center column pieces.
        center = board[:, board.shape[1] // 2]
        center_count = np.count_nonzero(center == player)
        return center_count * 3

def minimax(board, depth, alpha, beta, maximizingPlayer, player):
    """
    Minimax search with alpha-beta pruning and caching.
    - board: current board state (numpy array)
    - depth: search depth
    - alpha, beta: pruning values
    - maximizingPlayer: True if the current layer is maximizing
    - player: the RL agent’s designated player (e.g. 1). The minimax opponent will play as -player.
    Returns (score, best_move). If no moves available, best_move is None.
    """
    key = (board.tobytes(), depth, maximizingPlayer, player)
    if key in minimax_cache:
        return minimax_cache[key]

    valid_moves = sorted(get_valid_moves(board), key=lambda c: abs(c - board.shape[1] // 2))
    is_terminal = terminal_node(board)
    if depth == 0 or is_terminal:
        score = evaluate_board(board, player)
        return score, None

    if maximizingPlayer:
        value = -np.inf
        best_move = valid_moves[0]
        for col in valid_moves:
            new_board = drop_piece(board, col, player)
            score, _ = minimax(new_board, depth - 1, alpha, beta, False, player)
            if score > value:
                value = score
                best_move = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        minimax_cache[key] = (value, best_move)
        return value, best_move
    else:
        value = np.inf
        best_move = valid_moves[0]
        opp = -player
        for col in valid_moves:
            new_board = drop_piece(board, col, opp)
            score, _ = minimax(new_board, depth - 1, alpha, beta, True, player)
            if score < value:
                value = score
                best_move = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        minimax_cache[key] = (value, best_move)
        return value, best_move

def minimax_decision(board, depth, player):
    """Return the best move for the given player using minimax search. Clears cache before search."""
    global minimax_cache
    minimax_cache.clear()
    _, best_move = minimax(board, depth, -np.inf, np.inf, True, player)
    return best_move

##############################
# ONE-HOT ENCODING HELPER
##############################
def one_hot_state(state):
    """
    Convert the board state (shape: (6,7) with values in {-1,0,1})
    to a one-hot encoded tensor of shape (3, 6, 7):
      - Channel 0: cells occupied by -1
      - Channel 1: empty cells (0)
      - Channel 2: cells occupied by +1
    """
    board = state.copy()
    one_hot = np.zeros((3, board.shape[0], board.shape[1]), dtype=np.float32)
    one_hot[0] = (board == -1).astype(np.float32)
    one_hot[1] = (board == 0).astype(np.float32)
    one_hot[2] = (board == 1).astype(np.float32)
    return one_hot

##############################
# CONNECT 4 ENVIRONMENT
##############################
class Connect4Env:
    def __init__(self, rows=6, cols=7, win_length=4):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.board = None
        self.current_player = 1  # 1 or -1
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self._get_state()

    def _get_state(self):
        # Return the board from the perspective of the current player.
        return self.board * self.current_player

    def valid_moves(self):
        """Return a list of valid column indices."""
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def step(self, action):
        """
        Drop a piece in the given column.
        Returns: next_state, reward, done, info.
        """
        if action not in self.valid_moves():
            return self._get_state(), -10, True, {"illegal_move": True}

        # Drop the piece (find the lowest available row).
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        if self.check_win(row, action):
            return self._get_state(), 1, True, {"win": True}
        if len(self.valid_moves()) == 0:
            return self._get_state(), 0, True, {"draw": True}

        # Switch player (flip perspective).
        self.current_player *= -1
        return self._get_state(), 0.01, False, {}

    def check_win(self, row, col):
        """Check if the move at (row, col) resulted in a win for the current player."""
        def count_direction(delta_row, delta_col):
            r, c = row, col
            count = 0
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == self.current_player:
                count += 1
                r += delta_row
                c += delta_col
            return count

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            if count_direction(dr, dc) + count_direction(-dr, -dc) - 1 >= self.win_length:
                return True
        return False

    def render(self):
        """Print the board with symbols."""
        symbol_map = {1: "X", -1: "O", 0: "."}
        for row in self.board:
            print(" ".join(symbol_map[val] for val in row))
        print("0 1 2 3 4 5 6")
        print()

##############################
# DQN MODEL WITH CONVOLUTIONAL LAYERS
##############################
class DQN(nn.Module):
    def __init__(self, input_channels=3, conv_channels=[32, 64, 128], fc_dim=512, output_dim=7):
        super(DQN, self).__init__()
        layers = []
        in_channels = input_channels
        for out_channels in conv_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        conv_out_size = conv_channels[-1] * 6 * 7
        self.fc1 = nn.Linear(conv_out_size, fc_dim)
        self.fc2 = nn.Linear(fc_dim, output_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # x: (batch, 3, 6, 7)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        return self.fc2(x)

##############################
# DQN AGENT
##############################
class DQNAgent:
    def __init__(self, lr=0.0005, gamma=0.99, batch_size=64, memory_size=10000, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_net = DQN().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state, valid_actions, use_epsilon=True):
        """
        Given a raw board state (shape (6,7)), first one-hot encode it then
        use the network to compute Q-values. After masking invalid moves,
        apply softmax and sample an action.
        """
        if use_epsilon and np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        oh_state = one_hot_state(state)
        state_tensor = torch.FloatTensor(oh_state).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state_tensor).cpu().data.numpy()[0]
        mask = np.ones(7, dtype=bool)
        mask[valid_actions] = False
        q_values[mask] = -np.inf
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        action = np.random.choice(np.arange(7), p=probs)
        return int(action)

    def remember(self, state, action, reward, next_state, done, valid_actions_next):
        self.memory.append((state, action, reward, next_state, done, valid_actions_next))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = torch.FloatTensor([one_hot_state(s) for (s, a, r, s_next, d, va) in minibatch]).to(self.device)
        action_batch = torch.LongTensor([a for (s, a, r, s_next, d, va) in minibatch]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor([r for (s, a, r, s_next, d, va) in minibatch]).to(self.device)
        next_state_batch = torch.FloatTensor([one_hot_state(s_next) for (s, a, r, s_next, d, va) in minibatch]).to(self.device)
        done_batch = torch.FloatTensor([float(d) for (s, a, r, s_next, d, va) in minibatch]).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        next_q_all = self.policy_net(next_state_batch).detach()
        next_q_values = []
        for i, (_, _, _, _, done_flag, valid_actions_next) in enumerate(minibatch):
            if done_flag:
                next_q_values.append(0.0)
            else:
                q_vals = next_q_all[i].cpu().data.numpy()
                mask = np.ones(7, dtype=bool)
                mask[valid_actions_next] = False
                q_vals[mask] = -np.inf
                next_q_values.append(np.max(q_vals))
        next_q_values = torch.FloatTensor(next_q_values).to(self.device)
        target_q = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

##############################
# PPO MODEL WITH CONVOLUTIONAL LAYERS
##############################
class PPONet(nn.Module):
    def __init__(self, input_channels=3, conv_channels=[32, 64, 128], fc_dim=512, action_dim=7):
        super(PPONet, self).__init__()
        layers = []
        in_channels = input_channels
        for out_channels in conv_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        conv_out_size = conv_channels[-1] * 6 * 7

        # Five linear layers with LeakyReLU activations
        self.fc1 = nn.Linear(conv_out_size, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, fc_dim)
        self.fc4 = nn.Linear(fc_dim, fc_dim)
        self.fc5 = nn.Linear(fc_dim, fc_dim)

        self.policy_logits = nn.Linear(fc_dim, action_dim)
        self.value = nn.Linear(fc_dim, 1)

    def forward(self, x):
        # x: (batch, 3, 6, 7)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value

##############################
# PPO AGENT
##############################
class PPOAgent:
    def __init__(self, lr=0.0003, gamma=0.99, clip_epsilon=0.2, epochs=4, batch_size=64, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy = PPONet().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.Transition = namedtuple('Transition', ['state', 'action', 'logprob', 'reward', 'done', 'value'])
        self.memory = []

    def select_action(self, state, valid_actions):
        oh_state = one_hot_state(state)
        state_tensor = torch.FloatTensor(oh_state).unsqueeze(0).to(self.device)
        logits, value = self.policy(state_tensor)
        logits = logits.cpu().data.numpy()[0]
        mask = np.ones(7, dtype=bool)
        mask[valid_actions] = False
        logits[mask] = -1e9
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        action = np.random.choice(np.arange(7), p=probs)
        logprob = np.log(probs[action] + 1e-8)
        return int(action), logprob, value.item()

    def store(self, state, action, logprob, reward, done, value):
        self.memory.append(self.Transition(state, action, logprob, reward, done, value))

    def clear_memory(self):
        self.memory = []

    def compute_returns_and_advantages(self, last_value):
        transitions = self.memory
        returns = []
        R = last_value
        for trans in reversed(transitions):
            R = trans.reward + self.gamma * R * (1 - trans.done)
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        values = torch.tensor([t.value for t in transitions], dtype=torch.float32).to(self.device)
        advantages = returns - values
        return returns, advantages

    def update(self):
        if len(self.memory) == 0:
            return
        transitions = self.memory
        states = torch.FloatTensor([one_hot_state(t.state) for t in transitions]).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        old_logprobs = torch.FloatTensor([t.logprob for t in transitions]).to(self.device)
        last_value = 0
        returns, advantages = self.compute_returns_and_advantages(last_value)
        dataset = torch.utils.data.TensorDataset(states, actions, old_logprobs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            for batch_states, batch_actions, batch_old_logprobs, batch_returns, batch_advantages in loader:
                logits, values = self.policy(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                entropy_loss = -dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.clear_memory()

##############################
# TRAINING FUNCTIONS (PLAYING AGAINST MINIMAX)
##############################
def train_dqn(episodes=1000, print_every=50, model_path="connect4_dqn_minimax.pth", minimax_depth=3, device='cpu'):
    env = Connect4Env()
    agent = DQNAgent(device=device)
    win_history = []
    for e in range(1, episodes + 1):
        state = env.reset()
        done = False
        moves = 0
        while not done:
            # RL Agent's turn
            valid_moves = env.valid_moves()
            action = agent.select_action(state, valid_moves)
            next_state, reward, done, info = env.step(action)
            # If RL agent wins or makes an illegal move, store and break.
            if done:
                agent.remember(state, action, reward, next_state, done, env.valid_moves())
                break
            # Now let the minimax opponent move.
            # The minimax opponent plays as the current player.
            opp_move = minimax_decision(env.board.copy(), minimax_depth, env.current_player)
            next_state2, reward_opp, done, info = env.step(opp_move)
            # Combine the rewards from the agent’s move and the opponent’s move.
            combined_reward = reward #+ reward_opp
            agent.remember(state, action, combined_reward, next_state2, done, env.valid_moves())
            state = next_state2
            moves += 1
            if info.get("illegal_move", False):
                break
            # Perform replay step after each full cycle.
            agent.replay()
        win_history.append(1 if reward == 1 else 0)
        if e % print_every == 0:
            win_rate = np.mean(win_history[-print_every:]) * 100
            print(f"DQN Episode {e}/{episodes} - Moves: {moves}, Final Reward: {reward:.2f}, Win rate (last {print_every}): {win_rate:.1f}%, Epsilon: {agent.epsilon:.3f}")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"DQN training complete. Model saved to {model_path}.")

def train_ppo(episodes=1000, print_every=50, model_path="connect4_ppo_minimax.pth", minimax_depth=3, device='cpu'):
    env = Connect4Env()
    agent = PPOAgent(device=device)
    win_history = []
    moves_history = []
    for e in range(1, episodes + 1):
        state = env.reset()
        done = False
        moves = 0
        while not done:
            # RL Agent's turn
            valid_moves = env.valid_moves()
            action, logprob, value = agent.select_action(state, valid_moves)
            next_state, reward, done, info = env.step(action)
            if done:
                agent.store(state, action, logprob, reward, done, value)
                break
            # Minimax opponent's move.
            opp_move = minimax_decision(env.board.copy(), minimax_depth, env.current_player)
            next_state2, reward_opp, done, info = env.step(opp_move)
            combined_reward = reward # + reward_opp
            agent.store(state, action, logprob, combined_reward, done, value)
            state = next_state2
            moves += 1
            if info.get("illegal_move", False):
                break
        agent.update()
        win_history.append(1 if reward == 1 else 0)
        moves_history.append(moves)
        if e % print_every == 0:
            win_rate = np.mean(win_history[-print_every:]) * 100
            moves_avg = np.mean(moves_history[-print_every:])
            print(f"PPO Episode {e}/{episodes} - Final Reward: {reward:.2f}, Avg Moves (last {print_every}): {moves_avg:.1f} Win rate (last {print_every}): {win_rate:.1f}%, minimax depth: {minimax_depth}")
            torch.save(agent.policy.state_dict(), model_path)
            if win_rate > 75:
                minimax_depth = min(5, minimax_depth + 1)
    torch.save(agent.policy.state_dict(), model_path)
    print(f"PPO training complete. Model saved to {model_path}.")

##############################
# INTERACTIVE PLAY FUNCTION
##############################
def play_against_agent(model_path, algo="dqn", device='cpu'):
    env = Connect4Env()
    if algo == "dqn":
        agent = DQNAgent(device=device)
        try:
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
            agent.policy_net.eval()
        except Exception as e:
            print(f"Error loading DQN model: {e}")
            return
    elif algo == "ppo":
        agent = PPOAgent(device=device)
        try:
            agent.policy.load_state_dict(torch.load(model_path, map_location=device))
            agent.policy.eval()
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            return
    else:
        print("Invalid algorithm specified for play mode.")
        return

    first = input("Do you want to go first? (y/n): ").lower().strip()
    human_first = (first == "y")
    state = env.reset()
    if not human_first:
        valid_moves = env.valid_moves()
        if algo == "dqn":
            action = agent.select_action(state, valid_moves, use_epsilon=False)
        else:
            action, _, _ = agent.select_action(state, valid_moves)
        state, reward, done, info = env.step(action)
        print("Agent made its move:")
        env.render()

    done = False
    while not done:
        env.render()
        valid = env.valid_moves()
        move = None
        while True:
            try:
                move = int(input(f"Your move (choose column from {valid}): "))
            except ValueError:
                print("Please enter an integer.")
                continue
            if move not in valid:
                print("Invalid move. Try again.")
            else:
                break
        state, reward, done, info = env.step(move)
        env.render()
        if done:
            if info.get("win", False):
                print("You win!")
            elif info.get("draw", False):
                print("It's a draw!")
            elif info.get("illegal_move", False):
                print("Illegal move made.")
            break
        valid = env.valid_moves()
        if algo == "dqn":
            action = agent.select_action(state, valid, use_epsilon=False)
        else:
            action, _, _ = agent.select_action(state, valid)
        print(f"Agent chooses column {action}.")
        state, reward, done, info = env.step(action)
        if done:
            env.render()
            if info.get("win", False):
                print("Agent wins!")
            elif info.get("draw", False):
                print("It's a draw!")
            elif info.get("illegal_move", False):
                print("Agent made an illegal move.")
            break

##############################
# MAIN
##############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect4 RL Trainer with Minimax Opponent (Alpha-Beta Pruning)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "play"],
                        help="Mode: 'train' to train the agent, 'play' to play against it.")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"],
                        help="Algorithm: choose between 'dqn' and 'ppo'.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save/load the model.")
    parser.add_argument("--minimax_depth", type=int, default=3, help="Depth for the minimax opponent search.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_path is None:
        if args.algo == "dqn":
            args.model_path = "connect4_dqn_minimax.pth"
        else:
            args.model_path = "connect4_ppo_minimax.pth"

    if args.mode == "train":
        if args.algo == "dqn":
            train_dqn(episodes=args.episodes, model_path=args.model_path, minimax_depth=args.minimax_depth, device=device)
        elif args.algo == "ppo":
            train_ppo(episodes=args.episodes, model_path=args.model_path, minimax_depth=args.minimax_depth, device=device)
    elif args.mode == "play":
        play_against_agent(model_path=args.model_path, algo=args.algo, device=device)
