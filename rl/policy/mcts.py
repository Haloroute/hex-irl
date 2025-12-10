import math, torch

import numpy as np

from collections import deque
from random import choice
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torch import Tensor
from tqdm.auto import tqdm

from rl.environment import HexEnv


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search (MCTS) tree.
    """
    def __init__(
        self,
        env: HexEnv,
        observation: Tensor,
        action: Tensor = None,
        done: bool = False,
        parent: 'MCTSNode' = None
    ):
        """
        Create a new MCTS node.

        :param env: the Hex environment
        :type env: HexEnv
        :param observation: the current observation tensor, representing the game state
        :type observation: Tensor
        :param action: the action taken to reach this node from its parent
        :type action: Tensor
        :param done: whether this node represents a terminal state
        :type done: bool
        :param parent: the parent MCTSNode, or None if this is the root
        :type parent: 'MCTSNode'
        """
        self.env: HexEnv = env
        self.observation: Tensor = observation
        self.action: Tensor = action
        self.done: bool = done
        self.parent: MCTSNode = parent
        self.children: list[MCTSNode] = []
        self.n_wins: int = 0
        self.n_visits: int = 0
        self.untried_moves: list[int] = self._get_untried_moves()

    def add_child(self, child_node: 'MCTSNode') -> None:
        child_node.parent = self
        self.children.append(child_node)

    def _get_untried_moves(self) -> list[int]:
        # Get the list of untried moves from the current observation
        possible_moves: Tensor = self.env.get_action_mask(self.observation, flatten=True) # (max_board_size ** 2,)

        # Convert to list of indices
        untried_moves = [i for i in range(possible_moves.shape[0]) if possible_moves[i].item()]
        return untried_moves

    def uct(self) -> float:
        """Upper Confidence bound applied to Trees (UCT) score."""
        n_wins = self.n_wins
        n_parent_visits = self.parent.n_visits if self.parent else 0
        n_visits = self.n_visits + 1e-8  # Avoid division by zero
        if n_parent_visits and n_visits:
            return n_wins / n_visits + math.sqrt(2 * math.log(n_parent_visits) / n_visits)
        else:
            return math.inf


class MCTSPolicy(TensorDictModuleBase):
    """
    TensorDict-compatible MCTSPolicy policy that reads input state/action metadata from a tensordict
    and writes the chosen action back into the same structure.
    """
    def __init__(
        self, 
        env: HexEnv,
        observation_key: NestedKey = "observation",
        action_mask_key: NestedKey = "action_mask",
        action_key: NestedKey = "action",
        itermax: int = 1000,
        show_predictions: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the MCTSPolicy.

        :param env: Hex environment (must be single-instance).
        :type env: HexEnv
        :param observation_key: Key in the tensordict where the observation is stored.
        :type observation_key: NestedKey
        :param action_mask_key: Key in the tensordict where the action mask is stored.
        :type action_mask_key: NestedKey
        :param action_key: Key in the tensordict where the selected action will be stored.
        :type action_key: NestedKey
        :param itermax: Number of MCTS iterations to perform per action selection.
        :type itermax: int
        :param show_predictions: Whether to visualize MCTS predictions using the environment's UI.
        :type show_predictions: bool
        :param verbose: Whether to print MCTS statistics to the console.
        :type verbose: bool
        """
        super().__init__()
        self.in_keys, self.out_keys = [observation_key, action_mask_key], [action_key]
        self.env = env
        self.observation_key = observation_key
        self.action_mask_key = action_mask_key
        self.action_key = action_key
        self.itermax = itermax
        self.show_predictions = show_predictions
        self.verbose = verbose

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Entry point invoked by TorchRL; unwraps tensors, runs MCTSPolicy, then stores the selected action.
        """
        observation: Tensor = tensordict.get(self.observation_key) # (max_board_size, max_board_size, channels)
        current_player: int = int(observation[0, 0, 2].item())
        action: Tensor = self._run_mcts(observation, current_player) # (1,), already flattened
        tensordict.set(self.action_key, action)
        return tensordict

    def _run_mcts(
        self,
        board_state: Tensor,
        current_player: int
    ) -> Tensor:
        """
        Run the MCTS algorithm to select the best action.

        :param board_state: The current board state tensor
        :type board_state: Tensor
        :param current_player: The index of the current player (0 or 1)
        :type current_player: int
        :swap_available: Whether the swap move is available
        :type swap_available: bool
        """
        root_node = MCTSNode(self.env, board_state) # Root node
        stack = deque() # Stack contains moves ordered from root to current node

        # MCTS main loop
        for _ in range(self.itermax):
            node = root_node

            # Selection
            # Traverse the tree to find a node to expand
            while len(node.untried_moves) == 0 and len(node.children) != 0 and not node.done:
                uct_values = [child.uct() for child in node.children]
                if all(value == math.inf for value in uct_values):
                    node = choice(node.children)
                else:
                    node = node.children[np.argmax(uct_values)]
                action: Tensor = node.action
                stack.append(action)

            # Expansion
            # Expand a child node if there are untried moves
            if len(node.untried_moves) != 0 and not node.done:
                action_index = choice(node.untried_moves)
                action_tensor = torch.tensor([action_index], dtype=torch.long)
                next_state: TensorDict = self.env.step(
                    TensorDict({
                        self.observation_key: node.observation,
                        self.action_key: action_tensor
                    })
                ).get("next")
                node.untried_moves.remove(action_index)
                child_node = MCTSNode(
                    env=node.env,
                    observation=next_state.get(self.observation_key),
                    action=action_tensor,
                    done=next_state.get("done").item(),
                    parent=node
                )
                node.add_child(child_node)
                stack.append(action_index)
                node = child_node

            # Simulation
            # Play out a random game from the current node
            if not node.done:
                current_observation = node.observation
                while True:  # do-while loop
                    current_tensordict = self.env.rand_step(TensorDict({
                        self.observation_key: current_observation
                    })).get("next")
                    current_observation = current_tensordict.get(self.observation_key) # (max_board_size, max_board_size, channels)
                    if current_tensordict.get("done").item():
                        win_player: int = current_observation[0, 0, 2].item()
                        break
            else:
                win_player: int = node.observation[0, 0, 2].item()

            # Backpropagation
            # Update win/visit counts along the path
            while node is not None:
                node.n_visits += 1
                if win_player == current_player:
                    node.n_wins += 1
                node = node.parent

            # Update the root node's stats as well
            root_node.n_visits += 1
            if win_player == current_player:
                root_node.n_wins += 1

        # Select the best move from the root node
        ucts = [child.uct() for child in root_node.children]
        best_child: MCTSNode = root_node.children[np.argmax(ucts)]
        best_move: Tensor = best_child.action # (1,)

        return best_move


    # def _print_output(self, output: Iterable[Tuple[int, int, Tuple[int, int]]], result: Tuple[int, int]) -> None:
    #     """Render a Rich table summarizing search statistics."""
    #     sorted_output = sorted(output, key=lambda k: (k[2][0], k[2][1]))
    #     console = Console()
    #     table = Table(show_header=True, header_style="bold red")
    #     table.add_column("Wins", justify="center")
    #     table.add_column("Visits", justify="center")
    #     table.add_column("Move", justify="center")
    #     for wins, visits, move in sorted_output:
    #         if move == result:
    #             w = f"[cyan]{wins}[/cyan]"
    #             v = f"[cyan]{visits}[/cyan]"
    #             m = f"[cyan]{str(tuple(map(int, move)))}[/cyan]"
    #         else:
    #             w, v, m = str(wins), str(visits), str(tuple(map(int, move)))
    #         table.add_row(w, v, m)
    #     console.print(table)