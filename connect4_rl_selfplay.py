import argparse
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##############################
# Helper: One-Hot Encoding
##############################
def one_hot_state(state):
    """
    Convert the board state (shape: (6,7) with values in {-1, 0, 1})
    to a one-hot encoded tensor of shape (3, 6, 7):
      - Channel 0: cells occupied by -1 (black)
      - Channel 1: empty cells (0)
      - Channel 2: cells occupied by +1 (white)
    """
    board = state.copy()
    one_hot = np.zeros((3, board.shape[0], board.shape[1]), dtype=np.float32)
    one_hot[0] = (board == -1).astype(np.float32)
    one_hot[1] = (board == 0).astype(np.float32)
    one_hot[2] = (board == 1).astype(np.float32)
    return one_hot

##############################
# Connect 4 Environment
##############################
class Connect4Env:
    def __init__(self, rows=6, cols=7, win_length=4):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.board = None
        self.current_player = 1  # +1 for white (who starts), -1 for black
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self._get_state()

    def _get_state(self):
        # Returns the board from the perspective of the current player.
        # (The board is multiplied by current_player.)
        return self.board * self.current_player

    def valid_moves(self):
        """Return a list of valid column indices where a move can be made."""
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def step(self, action):
        """
        Drop a piece in the given column.
        Returns: next_state, reward, done, info.
        Reward is computed from the perspective of the current player.
        """
        if action not in self.valid_moves():
            return self._get_state(), -10, True, {"illegal_move": True}

        # Drop the piece into the lowest available row in column.
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        if self.check_win(row, action):
            return self._get_state(), 1, True, {"win": True}

        if len(self.valid_moves()) == 0:
            return self._get_state(), 0, True, {"draw": True}

        # Switch player (flip perspective)
        self.current_player *= -1
        return self._get_state(), 0, False, {}

    def check_win(self, row, col):
        """Check if the last move at (row, col) wins the game."""
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
        """Print the board using symbols."""
        symbol_map = {1: "X", -1: "O", 0: "."}
        for row in self.board:
            print(" ".join(symbol_map[val] for val in row))
        print("0 1 2 3 4 5 6")
        print()

##############################
# PPO Network with Conv2d Layers (Larger Model, LeakyReLU)
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
# PPO Agent for Self-Play
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
# Self-Play Training Loop using PPO
##############################
def train_selfplay(episodes=1000, print_every=50, model_path="connect4_ppo_selfplay.pth", device='cpu'):
    env = Connect4Env()
    agent = PPOAgent(device=device)
    
    # Statistics
    white_wins = 0
    black_wins = 0
    draws = 0
    moves_history = []
    
    for e in range(1, episodes + 1):
        state = env.reset()
        episode_moves = 0
        done = False
        # Temporary list to store the episode's transitions.
        episode_memory_white = []
        episode_memory_black = []
        
        while not done:
            # White's turn
            valid_moves = env.valid_moves()
            action, logprob, value = agent.select_action(state, valid_moves)
            # action = random.choice(valid_moves)
            next_state, reward, done, info = env.step(action)
            next_valid_moves = env.valid_moves() if not done else []
            episode_memory_white.append((state, action, logprob, reward, done, value))
            if done:
                if info.get("win", False):
                    white_wins += 1
                elif info.get("draw", False):
                    draws += 1
                break
            
            # Black's turn
            state = next_state
            valid_moves = env.valid_moves()
            action, logprob, value = agent.select_action(state, valid_moves)
            next_state, reward, done, info = env.step(action)
            episode_memory_black.append((state, action, logprob, reward, done, value))
            state = next_state

            if done:
                if info.get("win", False):
                    black_wins += 1
                elif info.get("draw", False):
                    draws += 1
                break

            episode_moves += 1
            if info.get("illegal_move", False):
                done = True
                break

        for mem in episode_memory_white:
            agent.store(*mem)
        agent.update()
        for mem in episode_memory_black:
            agent.store(*mem)
        agent.update()
    
        moves_history.append(episode_moves)
        
        # Every print_every episodes, print stats and save model.
        if e % print_every == 0:
            avg_moves = np.mean(moves_history[-print_every:])
            total = print_every
            white_rate = (white_wins / total) * 100
            black_rate = (black_wins / total) * 100
            draw_rate = (draws / total) * 100
            print(f"Episode {e}/{episodes}: Avg Moves: {avg_moves:.1f} | White Wins: {white_wins} ({white_rate:.1f}%) | Black Wins: {black_wins} ({black_rate:.1f}%) | Draws: {draws} ({draw_rate:.1f}%)")
            white_wins = black_wins = draws = 0
            torch.save(agent.policy.state_dict(), model_path)
    
    # Save one final time.
    torch.save(agent.policy.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}.")

##############################
# Main
##############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect4 Self-Play PPO Trainer")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--print_every", type=int, default=50, help="Interval (in episodes) for printing stats and saving the model.")
    parser.add_argument("--model_path", type=str, default="connect4_ppo_selfplay.pth", help="Path to save the model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_selfplay(episodes=args.episodes, print_every=args.print_every, model_path=args.model_path, device=device)
