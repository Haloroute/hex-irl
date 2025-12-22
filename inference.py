"""
Hex Game Interactive Inference - Human vs AI

This script loads a trained SAC model and allows human players to play against the AI
using a Pygame-based graphical interface.
"""

import os, pygame, torch

import numpy as np

from pathlib import Path
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor
from torchrl.envs import TransformedEnv, ActionMask
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.modules import ProbabilisticActor, MaskedCategorical

# Import custom modules
from rl.environment import HexEnv
from rl.model.network import HexModel
from rl.policy.wrapper import ModelWrapper
from rl.ui import UI
from rl.config import (
    DEVICE, STORAGE_DEVICE, BOARD_SIZE,
    MODEL_PARAMS, CHECKPOINT_DIR
)


class HexGamePlayer:
    """Interactive Hex game with human vs AI."""
    def __init__(
            self, 
            checkpoint_path: str, 
            board_size: int = BOARD_SIZE,
            human_first: bool = True
        ):
        self.board_size = board_size
        
        # Initialize Pygame
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
        pygame.init()
        pygame.display.set_caption("Hex - Human vs AI")
        
        # Initialize UI
        self.ui = UI(board_size=board_size)
        
        # Initialize environment
        self.env = TransformedEnv(
            HexEnv(
                board_size=board_size,
                max_board_size=board_size,
                device=STORAGE_DEVICE
            ),
            ActionMask()
        )
        
        # Load AI model
        self.actor = self._load_model(checkpoint_path)
        
        # Game state
        self.current_tensordict: TensorDict = None
        self.human_first: bool = human_first
        self.current_player: int = 0 if human_first else 1  # 0 = Red (Human), 1 = Blue (AI)
        self.game_over: bool = False
        
    def _load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint."""
        print("=" * 60)
        print("LOADING MODEL")
        print("=" * 60)
        
        # Create actor model
        model = HexModel(**MODEL_PARAMS).train().to(DEVICE)
        model_wrapper = ModelWrapper(model, temperature=0)
        network = TensorDictModule(
            model_wrapper,
            in_keys=["observation", "action_mask"],
            out_keys=["logits", "mask"]
        )
        actor = ProbabilisticActor(
            network,
            in_keys=["logits", "mask"],
            spec=self.env.action_spec,
            distribution_class=MaskedCategorical
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        network.load_state_dict(checkpoint['state_dict'])
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"  Win Rate: {checkpoint['win_rate']:.1%}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print("=" * 60)
        
        return actor
    
    # def _node_to_action(self, node: int) -> int:
    #     """Convert UI node index to action index."""
    #     return node
    
    # def _action_to_node(self, action: int) -> int:
    #     """Convert action index to UI node index."""
    #     return action
    
    def _update_ui_from_observation(self):
        """Update UI colors based on current observation."""
        current_observation: np.ndarray = self.current_tensordict['observation'].squeeze().numpy(force=True)
        # current_observation: (board_size, board_size, n_channels)
        # Channel 0: Player 0 (Red) positions
        # Channel 1: Player 1 (Blue) positions
        # Channel 2: Current player indicator (0 for Red, 1 for Blue)
        # Channel 3: Playable areas (1 for empty, 0 for occupied)
        # Channel 4: Is swapable (1 or 0)

        for row in range(self.board_size):
            for col in range(self.board_size):
                node = row * self.board_size + col
                if current_observation[row, col, 0] == 1:  # Red player
                    self.ui.color[node] = self.ui.red
                elif current_observation[row, col, 1] == 1:  # Blue player
                    self.ui.color[node] = self.ui.blue
                else:
                    self.ui.color[node] = self.ui.white
    
    def _get_available_moves(self) -> list:
        """Get list of available moves (empty cells)."""
        action_mask = self.current_tensordict['action_mask'].squeeze().numpy(force=True)
        return [i for i, available in enumerate(action_mask) if available]
    
    def _human_turn(self):
        """Handle human player's turn."""
        available_moves = self._get_available_moves()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_node = self.ui.get_node_click()

                if clicked_node is not None and clicked_node in available_moves:
                    # Valid move
                    self._execute_action(clicked_node)
                    return "move_made"

        return "waiting"

    def _ai_turn(self):
        """Handle AI player's turn."""
        # Get AI action
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            action_tensordict = self.actor(self.current_tensordict.to(DEVICE))
            action = action_tensordict['action'].item()

        # Execute action
        self._execute_action(action)

        # Small delay for visibility
        pygame.time.wait(300)

    def _execute_action(self, action: int):
        """Execute an action and update game state."""
        action_tensordict = self.current_tensordict.set("action", torch.tensor([action]))
        self.current_tensordict: TensorDict = self.env.step(action_tensordict).get("next") # Get next tensordict

        # Check if game is over
        if self.current_tensordict['done'].item():
            self.game_over = True
            self._show_game_result(self.current_player)

        # Update current player
        self.current_player = 1 - self.current_player

    def _show_game_result(self, winner: int):
        """Display game result."""
        print("\n" + "=" * 60)
        if self.human_first and winner == 1 or not self.human_first and winner == 0:
            print(f"🎉 HUMAN WINS! ({'Red' if self.human_first else 'Blue'})")
        elif self.human_first and winner == 0 or not self.human_first and winner == 1:
            print(f"🤖 AI WINS! ({'Blue' if self.human_first else 'Red'})")
        else:
            print("DRAW!")
        print("=" * 60)
    
    def reset_game(self):
        """Reset game to initial state."""
        self.current_tensordict: TensorDict = self.env.reset()
        self.current_player = 0 if self.human_first else 1  # Red (Human) starts
        self.game_over = False

        # Reset UI colors
        for i in range(self.board_size ** 2):
            self.ui.color[i] = self.ui.white
        
        print("\n" + "=" * 60)
        print("NEW GAME STARTED")
        if self.human_first:
            print("You are Red (Player 0) - Connect left to right")
            print("AI is Blue (Player 1) - Connect top to bottom")
        else:
            print("You are Blue (Player 1) - Connect top to bottom")
            print("AI is Red (Player 0) - Connect left to right")
        print("=" * 60)

    def play(self):
        """Main game loop."""
        self.reset_game()
        running = True
        while running:
            # Update UI from observation
            self._update_ui_from_observation()

            # Draw board
            self.ui.draw_board()

            if not self.game_over:
                # Show whose turn it is
                turn_text = "Your Turn (Red)" if self.current_player == 0 else "AI Thinking... (Blue)"
                text_surface = self.ui.fonts.render(turn_text, True, self.ui.white)
                self.ui.screen.blit(text_surface, (10, 10))

                # Handle turns
                if self.current_player == 0:  # Human turn
                    result = self._human_turn()
                    if result == "quit":
                        running = False
                else:  # AI turn
                    self._ai_turn()
            else:
                # Game over - show restart prompt
                restart_text = self.ui.fonts.render("Press R to restart or Q to quit", True, self.ui.green)
                self.ui.screen.blit(restart_text, (10, 10))

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            self.reset_game()
                        elif event.key == pygame.K_q:
                            running = False

            pygame.display.update()
            self.ui.clock.tick(30)

        pygame.quit()


def main():
    """Main entry point."""
    print("=" * 60)
    print("HEX GAME - HUMAN VS AI")
    print("=" * 60)
    
    # Find best checkpoint
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_path = checkpoint_dir / f"hex_{BOARD_SIZE}x{BOARD_SIZE}_best.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train a model first using train.py")
        return
    
    # Create game player
    game = HexGamePlayer(
        checkpoint_path=str(checkpoint_path),
        board_size=BOARD_SIZE,
        human_first=True
    )
    
    # Start playing
    game.play()


if __name__ == "__main__":
    main()