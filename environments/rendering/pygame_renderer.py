"""
environments/rendering/pygame_renderer.py
==========================================
Factory floor 2D visualisation using Pygame.
Used during development to visually verify environment logic.
NOT used during training (headless).

Shows:
    - Machine status (colour coded: green=OP, yellow=PM, orange=CM, red=FAIL)
    - Health bars per machine
    - Job queue depth
    - Resource inventory bars
    - Current timestep and episode stats
    - Gantt-style job progress

Usage:
    python scripts/render_episode.py --config configs/phase1.yaml
"""

from typing import List, Optional
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from environments.transitions.degradation import MachineState, MachineStatus
from environments.transitions.job_dynamics import Job, OpStatus
from environments.transitions.resource_dynamics import ResourceState


# Colour palette
COLOURS = {
    "bg":          (20,  20,  35),
    "panel":       (30,  30,  50),
    "text":        (220, 220, 220),
    "text_dim":    (140, 140, 160),
    "OP":          (60,  180, 75),    # green
    "PM":          (255, 215, 0),     # yellow
    "CM":          (255, 140, 0),     # orange
    "FAIL":        (220, 50,  50),    # red
    "health_high": (60,  180, 75),
    "health_med":  (255, 165, 0),
    "health_low":  (220, 50,  50),
    "job_done":    (80,  120, 80),
    "job_active":  (70,  130, 180),
    "job_late":    (180, 60,  60),
    "border":      (70,  70,  100),
}

STATUS_NAMES = {
    MachineStatus.OP:   "OP",
    MachineStatus.PM:   "PM",
    MachineStatus.CM:   "CM",
    MachineStatus.FAIL: "FAIL",
}


class PygameRenderer:
    """
    Renders manufacturing environment state to a pygame window.
    Call render() each step to update the display.
    """

    def __init__(self, config: dict):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for rendering: pip install pygame")

        self.config     = config
        self.n_machines = len(config.get("machines", []))
        self.width      = 1200
        self.height     = 700
        self.fps        = 10

        pygame.init()
        pygame.display.set_caption("Manufacturing MARL Environment")
        self.screen  = pygame.display.set_mode((self.width, self.height))
        self.clock   = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_md = pygame.font.SysFont("monospace", 14)
        self.font_sm = pygame.font.SysFont("monospace", 11)


    def render(
        self,
        machine_states: List[MachineState],
        jobs:           List[Job],
        resource_state: ResourceState,
        current_step:   int,
        valid_pairs:    list,
    ) -> Optional[np.ndarray]:
        """
        Renders one frame of the environment.

        Args:
            machine_states: Current machine states
            jobs:           Current job list
            resource_state: Current resource state
            current_step:   Current timestep
            valid_pairs:    Valid (j,k,m) pairs for Agent 2

        Returns:
            RGB array if needed, else None
        """
        # Handle pygame events (quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        # Clear screen
        self.screen.fill(COLOURS["bg"])

        # Draw sections
        self._draw_header(current_step, len(jobs), len(valid_pairs))
        self._draw_machines(machine_states)
        self._draw_jobs(jobs, current_step)
        self._draw_resources(resource_state)

        pygame.display.flip()
        self.clock.tick(self.fps)
        return None


    def _draw_header(self, step: int, n_jobs: int, n_valid: int) -> None:
        t = self.font_lg.render(
            f"Manufacturing MARL  |  Step: {step:>4}  |  "
            f"Active Jobs: {n_jobs:>3}  |  Valid Actions: {n_valid:>3}",
            True, COLOURS["text"]
        )
        self.screen.blit(t, (20, 15))
        pygame.draw.line(self.screen, COLOURS["border"],
                         (0, 45), (self.width, 45), 1)


    def _draw_machines(self, machine_states: List[MachineState]) -> None:
        """Draw machine status panels with health bars."""
        title = self.font_md.render("MACHINES", True, COLOURS["text_dim"])
        self.screen.blit(title, (20, 55))

        panel_w = min(210, (self.width - 40) // max(self.n_machines, 1))
        panel_h = 160

        for i, s in enumerate(machine_states):
            x = 20 + i * (panel_w + 8)
            y = 75

            # Panel background
            colour = COLOURS.get(STATUS_NAMES.get(s.status, "OP"), COLOURS["OP"])
            pygame.draw.rect(self.screen, COLOURS["panel"],
                             (x, y, panel_w, panel_h), border_radius=6)
            pygame.draw.rect(self.screen, colour,
                             (x, y, panel_w, 6), border_radius=3)

            # Machine name and status
            cfg_name = self.config["machines"][i].get("name", f"M{i}")
            name_t  = self.font_md.render(cfg_name[:12], True, COLOURS["text"])
            status_t = self.font_sm.render(
                STATUS_NAMES.get(s.status, "?"), True, colour
            )
            self.screen.blit(name_t,   (x + 6, y + 12))
            self.screen.blit(status_t, (x + 6, y + 30))

            # Health bar
            bar_x, bar_y = x + 6, y + 50
            bar_w = panel_w - 12
            bar_h = 14
            pygame.draw.rect(self.screen, COLOURS["border"],
                             (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            fill_w  = int(bar_w * s.health / 100.0)
            h_colour = (COLOURS["health_high"] if s.health > 60
                        else COLOURS["health_med"] if s.health > 30
                        else COLOURS["health_low"])
            if fill_w > 0:
                pygame.draw.rect(self.screen, h_colour,
                                 (bar_x, bar_y, fill_w, bar_h), border_radius=3)
            h_t = self.font_sm.render(f"Health: {s.health:.0f}%", True, COLOURS["text"])
            self.screen.blit(h_t, (bar_x, bar_y + bar_h + 3))

            # Hazard rate
            hz_t = self.font_sm.render(
                f"λ: {s.hazard_rate:.2e}", True, COLOURS["text_dim"]
            )
            self.screen.blit(hz_t, (x + 6, y + 90))

            # Virtual age
            va_t = self.font_sm.render(
                f"V.Age: {s.virtual_age:.0f}h", True, COLOURS["text_dim"]
            )
            self.screen.blit(va_t, (x + 6, y + 108))

            # Maintenance countdown
            if s.maint_steps_remaining > 0:
                mc_t = self.font_sm.render(
                    f"Maint: {s.maint_steps_remaining} steps", True, colour
                )
                self.screen.blit(mc_t, (x + 6, y + 126))


    def _draw_jobs(self, jobs: List[Job], current_step: int) -> None:
        """Draw job progress bars (simplified Gantt)."""
        title = self.font_md.render("JOBS (first 12)", True, COLOURS["text_dim"])
        self.screen.blit(title, (20, 250))

        bar_h   = 20
        bar_gap = 4
        max_y   = 260

        for i, job in enumerate(jobs[:12]):
            y = max_y + 20 + i * (bar_h + bar_gap)
            x = 20

            # Due date bar
            bar_w = 500
            pygame.draw.rect(self.screen, COLOURS["border"],
                             (x, y, bar_w, bar_h), border_radius=2)

            # Completion progress
            fill_w = int(bar_w * job.completion_ratio)
            j_colour = (COLOURS["job_done"] if job.is_complete
                        else COLOURS["job_late"] if current_step > job.due_date
                        else COLOURS["job_active"])
            if fill_w > 0:
                pygame.draw.rect(self.screen, j_colour,
                                 (x, y, fill_w, bar_h), border_radius=2)

            label = self.font_sm.render(
                f"J{job.job_id:>2} w={job.weight:.0f} "
                f"due={job.due_date:.0f} "
                f"{job.completion_ratio*100:.0f}%",
                True, COLOURS["text"]
            )
            self.screen.blit(label, (x + bar_w + 8, y + 3))


    def _draw_resources(self, resource_state: ResourceState) -> None:
        """Draw resource inventory and pipeline."""
        title = self.font_md.render("RESOURCES", True, COLOURS["text_dim"])
        self.screen.blit(title, (700, 55))

        ren_names = [r["name"] for r in self.config["resources"]["renewable"]]
        con_names = [r["name"] for r in self.config["resources"]["consumable"]]

        y = 75
        for i, (name, avail, cap) in enumerate(zip(
            ren_names,
            resource_state.renewable_available,
            resource_state.renewable_capacity,
        )):
            t = self.font_sm.render(
                f"{name[:12]:12s}: {avail}/{cap}", True, COLOURS["text"]
            )
            self.screen.blit(t, (700, y + i * 18))

        y += len(ren_names) * 18 + 10
        for i, (name, inv) in enumerate(zip(
            con_names, resource_state.consumable_inventory
        )):
            pipeline_str = " ".join(
                f"{int(v)}" for v in resource_state.pending_orders[i]
                if v > 0
            ) or "—"
            t = self.font_sm.render(
                f"{name[:12]:12s}: {inv:.0f} | pipe:[{pipeline_str}]",
                True, COLOURS["text"]
            )
            self.screen.blit(t, (700, y + i * 18))


    def close(self) -> None:
        pygame.quit()
