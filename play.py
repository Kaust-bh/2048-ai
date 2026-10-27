"""Watch the trained DQN play in Pygame. SPACE pause, +/- speed, R restart, ESC quit."""
import argparse
import pygame
from game import Game2048
from agent import DQNAgent

BG = (250, 248, 239)
BOARD_BG = (187, 173, 160)
EMPTY = (205, 193, 180)
TEXT = (70, 60, 50)
TILE_COLORS = {
    2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
    16: (245, 149, 99), 32: (246, 124, 95), 64: (246, 94, 59),
    128: (237, 207, 114), 256: (237, 204, 97), 512: (237, 200, 80),
    1024: (237, 197, 63), 2048: (237, 194, 46),
}


def main(checkpoint):
    pygame.init()
    width, height = 520, 610
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("2048 DQN AI")
    font = pygame.font.SysFont("Arial", 28, bold=True)
    small = pygame.font.SysFont("Arial", 17)
    clock = pygame.time.Clock()

    env = Game2048()
    agent = DQNAgent()
    agent.load(checkpoint)
    agent.epsilon = 0.0
    paused, delay = False, 180
    last_action = "Starting"
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    env.reset()
                    last_action = "Restarted"
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    delay = max(20, delay - 30)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    delay = min(1000, delay + 30)

        if not paused and not env.done:
            action = agent.act(env.state_vector(), explore=False, valid_actions=env.valid_actions())
            _, _, _, info = env.step(action)
            last_action = ("up", "right", "down", "left")[action]

        screen.fill(BG)
        title = font.render("2048 DQN AI", True, TEXT)
        screen.blit(title, (20, 15))
        score_text = small.render(f"Score: {env.score}     Best tile: {int(env.board.max())}", True, TEXT)
        screen.blit(score_text, (22, 55))
        status = "GAME OVER" if env.done else ("PAUSED" if paused else f"Move: {last_action}")
        screen.blit(small.render(status, True, TEXT), (22, 80))

        board_x, board_y, gap, cell = 20, 115, 10, 110
        pygame.draw.rect(screen, BOARD_BG, (board_x, board_y, 480, 480), border_radius=8)
        for r in range(4):
            for c in range(4):
                value = int(env.board[r, c])
                x = board_x + gap + c * (cell + 5)
                y = board_y + gap + r * (cell + 5)
                color = TILE_COLORS.get(value, (60, 58, 50)) if value else EMPTY
                pygame.draw.rect(screen, color, (x, y, cell, cell), border_radius=5)
                if value:
                    label = font.render(str(value), True, (255, 255, 255) if value >= 8 else TEXT)
                    rect = label.get_rect(center=(x + cell // 2, y + cell // 2))
                    screen.blit(label, rect)

        controls = small.render("SPACE pause | +/- speed | R restart | ESC quit", True, TEXT)
        screen.blit(controls, (20, 600 - 18))
        pygame.display.flip()
        clock.tick(max(1, 1000 // delay))

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dqn_2048.pt")
    args = parser.parse_args()
    main(args.checkpoint)
