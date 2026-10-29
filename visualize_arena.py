"""Interactive Pygame arena for the project's five 2048 agents."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import time

import pygame

from arena_agents import ArenaState, Decision, ExpectimaxAdapter, build_adapters

# Palette: warm paper, ink, and coral accents.
BG = (20, 25, 30)
PANEL = (29, 36, 43)
PANEL_ALT = (37, 46, 54)
INK = (238, 241, 238)
MUTED = (155, 169, 172)
ACCENT = (239, 125, 92)
GREEN = (111, 201, 145)
BOARD = (72, 81, 87)
TILES = {
    0: (54, 63, 69), 2: (238, 228, 218), 4: (237, 224, 200),
    8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
    64: (233, 85, 65), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46),
}
ACTIONS = ("up", "right", "down", "left")


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


class ArenaApp:
    def __init__(self, seed=None, depth=2):
        pygame.init()
        self.screen = pygame.display.set_mode((1180, 760), pygame.RESIZABLE)
        pygame.display.set_caption("2048 AI Arena")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Avenir Next", 18)
        self.small = pygame.font.SysFont("Avenir Next", 14)
        self.title_font = pygame.font.SysFont("Avenir Next", 32, bold=True)
        self.tile_font = pygame.font.SysFont("Avenir Next", 38, bold=True)
        self.state = ArenaState(seed=seed)
        self.depth = depth
        self.adapters = build_adapters(depth)
        self.names = list(self.adapters)
        self.selected = self.names[0] if self.names else "No models available"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.pending_epoch = None
        self.running = True
        self.playing = False
        self.step_pending = False
        self.speed = 6.0
        self.diagnostics_visible = True
        self.error = ""
        self.last_tick = time.perf_counter()

    def close(self):
        self.running = False
        self.executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()

    def reset(self):
        self.state.reset()
        self.playing = False
        self.step_pending = False
        self.pending = None
        self.pending_epoch = None
        self.error = ""

    def select(self, name):
        if name in self.adapters:
            self.selected = name
            self.reset()

    def request_decision(self):
        if self.pending is not None or self.state.env.done or self.state.paused:
            return
        adapter = self.adapters.get(self.selected)
        if adapter is None:
            return
        board = self.state.env.board.copy()
        legal = self.state.env.valid_actions()
        epoch = self.state.epoch
        self.pending_epoch = epoch
        self.pending = self.executor.submit(adapter.decide, board, legal)

    def poll_decision(self):
        if self.pending is None or not self.pending.done():
            return
        future = self.pending
        self.pending = None
        try:
            decision = future.result()
            self.state.apply(self.pending_epoch, decision)
            if self.step_pending:
                self.state.pause()
                self.step_pending = False
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state.pause()
        self.pending_epoch = None

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.playing = not self.playing
                self.state.resume() if self.playing else self.state.pause()
            elif event.key == pygame.K_s:
                if self.state.paused:
                    self.state.resume()
                    self.step_pending = True
                    self.request_decision()
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_d:
                self.diagnostics_visible = not self.diagnostics_visible
            elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                if self.selected == "Expectimax":
                    delta = -1 if event.key == pygame.K_LEFTBRACKET else 1
                    self.depth = max(1, min(3, self.depth + delta))
                    self.adapters["Expectimax"] = ExpectimaxAdapter(self.depth)
                    self.reset()
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                self.speed = min(20.0, self.speed + 1)
            elif event.key == pygame.K_MINUS:
                self.speed = max(1.0, self.speed - 1)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            if x < 280:
                for index, name in enumerate(self.names):
                    if 132 + index * 36 <= y < 162 + index * 36:
                        self.select(name)

    def draw_text(self, text, x, y, color=INK, font=None):
        self.screen.blit((font or self.font).render(str(text), True, color), (x, y))

    def draw(self):
        width, height = self.screen.get_size()
        self.screen.fill(BG)
        left, right = 280, 330
        self.screen.fill(PANEL, (0, 0, left, height))
        self.screen.fill(PANEL, (width - right, 0, right, height))
        self.draw_text("2048 AI ARENA", 24, 24, ACCENT, self.title_font)
        self.draw_text("SELECT MODEL", 24, 92, MUTED, self.small)
        for index, name in enumerate(self.names):
            y = 124 + index * 38
            active = name == self.selected
            pygame.draw.rect(self.screen, ACCENT if active else PANEL_ALT, (18, y, 244, 30), border_radius=5)
            self.draw_text(name, 28, y + 6, BG if active else INK, self.small)
        self.draw_text("SPACE  play / pause", 24, height - 108, MUTED, self.small)
        self.draw_text("S  step    R  reset    D  diagnostics", 24, height - 84, MUTED, self.small)
        self.draw_text(f"Speed {self.speed:.0f} moves/min", 24, height - 58, GREEN, self.small)
        self.draw_text("DEPTH  [ / ]", 24, height - 158, MUTED, self.small)
        self.draw_text(str(self.depth) if self.selected == "Expectimax" else "N/A", 24, height - 136, INK)

        board_size = min(height - 100, width - left - right - 70)
        board_x = left + (width - left - right - board_size) // 2
        board_y = (height - board_size) // 2 + 20
        gap = max(7, board_size // 60)
        cell = (board_size - gap * 5) // 4
        pygame.draw.rect(self.screen, BOARD, (board_x, board_y, board_size, board_size), border_radius=12)
        for row in range(4):
            for col in range(4):
                value = int(self.state.env.board[row, col])
                x = board_x + gap + col * (cell + gap)
                y = board_y + gap + row * (cell + gap)
                pygame.draw.rect(self.screen, TILES.get(value, (70, 60, 50)), (x, y, cell, cell), border_radius=7)
                if value:
                    label = self.tile_font.render(str(value), True, BG if value < 8 else INK)
                    self.screen.blit(label, label.get_rect(center=(x + cell // 2, y + cell // 2)))
        self.draw_text(f"Score  {self.state.env.score}", board_x, board_y - 42, INK)
        self.draw_text(f"Best  {int(self.state.env.board.max())}   Moves  {self.state.move_count}", board_x + 210, board_y - 42, MUTED)
        if self.state.env.done:
            self.draw_text("GAME OVER  press R", board_x + 25, board_y + board_size // 2, ACCENT, self.title_font)
        elif self.pending:
            self.draw_text("Thinking...", board_x + 25, board_y + board_size + 18, GREEN)

        self.draw_text("LIVE DIAGNOSTICS", width - right + 24, 28, ACCENT, self.title_font)
        self.draw_text(self.selected, width - right + 24, 74, INK)
        decision = self.state.last_decision
        if decision:
            self.draw_text(f"Chosen: {decision.action_name.upper()}", width - right + 24, 112, GREEN)
            self.draw_text(f"Latency: {decision.latency_seconds * 1000:.2f} ms", width - right + 24, 140, MUTED, self.small)
            y = 184
            self.draw_text("ACTION VALUES", width - right + 24, y, MUTED, self.small)
            for action in range(4):
                value = decision.values.get(action, "N/A")
                legal = "legal" if action in decision.legal_actions else "masked"
                self.draw_text(f"{ACTIONS[action]:>5}  {format_value(value):>10}  {legal}", width - right + 24, y + 24 + action * 23, INK if legal == "legal" else MUTED, self.small)
            if self.diagnostics_visible:
                y += 132
                self.draw_text("DETAILS", width - right + 24, y, MUTED, self.small)
                for key, value in decision.diagnostics.items():
                    if key == "raw_q_values" or key == "masked_q_values":
                        continue
                    self.draw_text(f"{key}: {format_value(value)}", width - right + 24, y + 22, INK, self.small)
                    y += 21
        else:
            self.draw_text("No decision yet", width - right + 24, 112, MUTED)
        if self.error:
            self.draw_text(self.error[:42], 24, height - 25, ACCENT, self.small)
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.poll_decision()
            if self.playing and not self.state.paused and self.pending is None:
                self.request_decision()
            self.draw()
            self.clock.tick(30)
        self.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    ArenaApp(seed=args.seed, depth=args.depth).run()


if __name__ == "__main__":
    main()
