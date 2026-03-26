"""
viz_factory.py  —  Manufacturing MARL: Factory Floor Visualiser
================================================================
Atari-meets-factory-HUD aesthetic.
Run:  python viz_factory.py

Controls:
  SPACE       pause / resume
  → / ←       step forward / back (paused)
  ↑ / ↓       speed up / slow down
  R           reset episode
  1           toggle Agent 1 brain panel
  2           toggle Agent 2 brain panel
  J           toggle Jobs kanban / Gantt
  T           toggle resource overlay
  Q / ESC     quit
"""
from __future__ import annotations
import sys, os, math, random, time, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml, pygame, pygame.gfxdraw
with open("configs/base.yaml") as f:
    CONFIG = yaml.safe_load(f)
CONFIG["episode"]["t_max_train"] = 200
CONFIG["jobs"]["n_jobs_train"]   = 10
CONFIG["stochasticity_level"]    = 1

from environments.mfg_env        import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
from environments.transitions.degradation import MachineStatus

# ══════════════════════════════════════════════════════════════════
# PALETTE
# ══════════════════════════════════════════════════════════════════
P = {
    "bg":       (6,   8,  16),
    "floor":    (10,  14, 24),
    "grid":     (18,  24, 40),
    "panel":    (13,  17, 30),
    "panel2":   (20,  26, 44),
    "border":   (30,  42, 72),
    "border_hi":(55,  80,140),
    # status
    "OP":    (0,  220, 110),
    "PM":    (255,190,   0),
    "CM":    (255,100,  10),
    "FAIL":  (255, 30,  55),
    # agents
    "a1":    (255,160,   0),   # amber  — PDM Agent
    "a2":    ( 0, 200, 255),   # cyan   — Job Shop Agent
    # jobs
    "j_wait":  ( 50, 65,110),
    "j_run":   (  0,160,255),
    "j_done":  (  0,200, 80),
    "j_late":  (255, 50, 70),
    # resources
    "r_ren":   ( 80,180,255),
    "r_con":   (200,100,255),
    # text
    "txt":     (210,220,255),
    "dim":     ( 70, 90,140),
    "hi":      (255,255,255),
    "grn":     (  0,220,110),
    "red":     (255, 45, 70),
    "amb":     (255,185,  0),
    "cyan":    (  0,200,255),
    # machine parts
    "steel":   ( 60, 75,105),
    "steel2":  ( 40, 52, 78),
    "steel3":  ( 80,100,140),
    "spark":   (255,240,  0),
    "heat":    (255, 80,  0),
}

W, H = 1600, 900
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Manufacturing MARL — Factory Floor")
clock  = pygame.time.Clock()

def font(sz, bold=False):
    for f in ["Courier New","Consolas","Lucida Console"]:
        try: return pygame.font.SysFont(f, sz, bold=bold)
        except: pass
    return pygame.font.Font(None, sz)

Fb = font(22, True);  Fm = font(14, True);  Fs = font(11);  Ft = font(9)
Fbig = font(28, True)

# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════
def rect(surf, col, x, y, w, h, r=0, alpha=255):
    if alpha < 255:
        s = pygame.Surface((w,h), pygame.SRCALPHA)
        s.fill((*col, alpha))
        surf.blit(s,(x,y))
    else:
        pygame.draw.rect(surf, col, (x,y,w,h), border_radius=r)

def border(surf, col, x, y, w, h, t=1, r=0):
    pygame.draw.rect(surf, col, (x,y,w,h), t, border_radius=r)

def txt(surf, text, x, y, col, f=Fs, anchor="tl"):
    s = f.render(str(text), True, col)
    if anchor == "c":  x -= s.get_width()//2
    elif anchor == "tr": x -= s.get_width()
    surf.blit(s,(x,y))
    return s.get_width()

def lerp(a,b,t): return a + (b-a)*t
def lerpC(c1,c2,t): return tuple(int(lerp(c1[i],c2[i],t)) for i in range(3))
def healthC(h):
    if h>60: return lerpC(P["PM"],P["OP"],(h-60)/40)
    elif h>30: return lerpC(P["FAIL"],P["PM"],(h-30)/30)
    return P["FAIL"]

def hbar(surf, x,y,w,h, frac, hi, lo=None, bg=None):
    rect(surf, bg or P["panel2"], x,y,w,h, 2)
    fw = max(int(w*frac),0)
    if fw:
        col = lerpC(lo,hi,frac) if lo else hi
        rect(surf, col, x,y,fw,h, 2)

def glow_circle(surf, col, cx, cy, r, alpha=80):
    s = pygame.Surface((r*2,r*2), pygame.SRCALPHA)
    pygame.draw.circle(s, (*col, alpha), (r,r), r)
    surf.blit(s, (cx-r, cy-r))

def pulse(t, speed=2, lo=0.4, hi=1.0):
    return lo + (hi-lo)*(0.5+0.5*math.sin(t*speed))

STATUS_COL  = {0:P["OP"],1:P["PM"],2:P["CM"],3:P["FAIL"]}
STATUS_NAME = {0:"OP",1:"PM",2:"CM",3:"FAIL"}

# ══════════════════════════════════════════════════════════════════
# MACHINE DRAWING — factory floor top-down icons
# ══════════════════════════════════════════════════════════════════
MACHINE_ICONS = ["CNC Mill","Lathe","Press","Grinder","Drill"]

def draw_machine_icon(surf, kind, x, y, w, h, status, health, busy, t_anim, maint_left=0):
    sc  = STATUS_COL[status]
    hc  = healthC(health)
    p   = pulse(t_anim)

    # Shadow
    rect(surf, (0,0,0), x+4,y+4,w,h, 8, alpha=60)

    # Body
    body_col = lerpC(P["steel2"],P["steel"], health/100)
    rect(surf, body_col, x,y,w,h, 8)
    border(surf, sc, x,y,w,h, 2, 8)

    cx, cy = x+w//2, y+h//2

    if kind == "CNC Mill":
        # Bed
        rect(surf, P["steel3"], x+6,cy-6,w-12,h//3+6, 4)
        # Spindle
        pygame.draw.circle(surf, P["steel2"], (cx,cy-8), 12)
        pygame.draw.circle(surf, P["steel3"], (cx,cy-8), 8)
        if busy:
            pygame.draw.circle(surf, P["spark"], (cx,cy-8), 4)
            glow_circle(surf, P["spark"], cx, cy-8, 10, int(100*p))
        # Tool
        pygame.draw.line(surf, P["steel3"], (cx,cy-8),(cx,cy+8), 3)

    elif kind == "Lathe":
        # Chuck
        pygame.draw.circle(surf, P["steel3"], (x+18,cy), 14)
        pygame.draw.circle(surf, P["steel2"], (x+18,cy), 8)
        for ang in [0,60,120,180,240,300]:
            ax = x+18+int(11*math.cos(math.radians(ang + t_anim*40 if busy else ang)))
            ay = cy+int(11*math.sin(math.radians(ang + t_anim*40 if busy else ang)))
            pygame.draw.circle(surf, P["steel"], (ax,ay), 3)
        # Bed
        rect(surf, P["steel3"], x+32,cy-5,w-46,10,3)
        # Tailstock
        pygame.draw.circle(surf, P["steel2"], (x+w-14,cy), 10)
        if busy:
            glow_circle(surf, P["cyan"], x+18,cy, 14, int(60*p))

    elif kind == "Press":
        # Frame
        rect(surf, P["steel3"], x+6,y+4, w-12,8, 2)
        rect(surf, P["steel3"], x+6,y+h-12,w-12,8, 2)
        rect(surf, P["steel3"], x+6,y+4,8,h-8, 2)
        rect(surf, P["steel3"], x+w-14,y+4,8,h-8, 2)
        # Ram
        ram_y = cy-10 if not busy else cy-10+int(12*abs(math.sin(t_anim*3)))
        rect(surf, P["steel2"], cx-10,ram_y,20,20,3)
        if busy:
            glow_circle(surf, P["heat"], cx,ram_y+20, 8, int(120*p))

    elif kind == "Grinder":
        # Wheel
        wheel_angle = t_anim*60 if busy else 0
        pygame.draw.circle(surf, P["steel3"], (cx,cy), 18)
        pygame.draw.circle(surf, P["steel2"], (cx,cy), 12)
        for i in range(8):
            ang = math.radians(wheel_angle + i*45)
            wx  = cx+int(14*math.cos(ang))
            wy  = cy+int(14*math.sin(ang))
            pygame.draw.circle(surf, P["steel"], (wx,wy), 3)
        if busy:
            for _ in range(3):
                sx = cx+random.randint(-14,14)
                sy = cy+random.randint(-14,14)
                pygame.draw.circle(surf, P["spark"], (sx,sy), 1)
            glow_circle(surf, P["spark"], cx,cy, 20, int(80*p))

    elif kind == "Drill":
        # Column
        rect(surf, P["steel3"], cx-5,y+4,10,h-8,3)
        # Head
        rect(surf, P["steel2"], cx-14,y+6,28,20,4)
        # Bit
        drill_y = cy-4 if not busy else cy-4+int(8*abs(math.sin(t_anim*4)))
        pygame.draw.polygon(surf, P["steel3"], [
            (cx-4,drill_y),(cx+4,drill_y),(cx,drill_y+18)
        ])
        if busy:
            glow_circle(surf, P["cyan"], cx,drill_y+18, 6, int(100*p))

    # STATUS LED (top-right corner)
    led_col = lerpC(sc, P["hi"], 0.4*p) if status != MachineStatus.OP else sc
    pygame.draw.circle(surf, led_col, (x+w-10,y+10), 6)
    if status in (MachineStatus.FAIL, MachineStatus.PM, MachineStatus.CM):
        glow_circle(surf, led_col, x+w-10,y+10, 10, int(120*p))

    # Health bar at bottom
    hbar(surf, x+4,y+h-10,w-8,6, health/100, hc, P["FAIL"])

    # Maintenance countdown arc
    if maint_left > 0:
        max_m = 8
        ang   = int(360*(maint_left/max_m))
        try:
            pygame.draw.arc(surf, sc, (x+4,y+4,w-8,h-8),
                            math.radians(90), math.radians(90+ang), 3)
        except: pass

    # Machine name
    kind_short = MACHINE_ICONS.index(kind) if kind in MACHINE_ICONS else 0
    txt(surf, f"M{kind_short}", cx,y+8, P["dim"], Ft, "c")

# ══════════════════════════════════════════════════════════════════
# AGENT BRAIN PANELS
# ══════════════════════════════════════════════════════════════════

class AgentPanel:
    """Collapsible side panel showing agent observations, last action, masks."""
    def __init__(self, agent_id, side, col):
        self.id    = agent_id   # 1 or 2
        self.side  = side       # 'left' or 'right'
        self.col   = col
        self.open  = False
        self.anim  = 0.0        # 0=closed 1=open
        self.W_open  = 380
        self.W_close = 42

    @property
    def width(self): return int(lerp(self.W_close, self.W_open, self.anim))

    def toggle(self): self.open = not self.open

    def update(self, dt):
        target = 1.0 if self.open else 0.0
        self.anim += (target - self.anim) * min(dt*8, 1)

    def draw(self, surf, runner, t_anim):
        w   = self.width
        x   = 0 if self.side == "left" else W - w
        y   = 40
        h   = H - 40

        # Panel bg
        rect(surf, P["panel"], x,y,w,h)
        border(surf, self.col, x,y,w,h, 1)

        # Tab (always visible)
        tab_x = x if self.side == "right" else x+w-32
        rect(surf, self.col, tab_x,y+H//2-60,32,120, 6)
        label = f"A{self.id}"
        txt(surf, label, tab_x+16, y+H//2-8, P["bg"], Fm, "c")
        arrow = "◀" if (self.side=="right") == self.open else "▶"
        if self.side == "left": arrow = "▶" if not self.open else "◀"
        txt(surf, arrow, tab_x+16, y+H//2+14, P["bg"], Ft, "c")

        if self.anim < 0.05: return

        alpha_scale = min(self.anim * 3, 1.0)
        px = x+8 if self.side=="left" else x+8

        # Header
        aname = "PDM AGENT (θ₁)" if self.id==1 else "JOB SHOP AGENT (θ₂)"
        txt(surf, aname, px+5, y+10, self.col, Fm)
        pygame.draw.line(surf, self.col, (px,y+28),(px+w-20,y+28),1)

        if self.id == 1:
            self._draw_agent1(surf, runner, px, y+35, w-20, t_anim)
        else:
            self._draw_agent2(surf, runner, px, y+35, w-20, t_anim)

    def _draw_agent1(self, surf, runner, px, py, pw, t):
        env  = runner.env
        states = env.machine_states

        txt(surf, "OBSERVATION SPACE", px, py, P["dim"], Ft)
        py += 14

        txt(surf, "Machine Health States:", px, py, P["amb"], Ft); py+=11
        for s in states:
            name = CONFIG["machines"][s.machine_id]["name"].split()[0][:6]
            hc   = healthC(s.health)
            txt(surf, f"  {name:6s}", px, py, P["txt"], Ft)
            hbar(surf, px+52,py+1,pw-60,8, s.health/100, hc, P["FAIL"])
            txt(surf, f"{s.health:.0f}%", px+pw-24,py, hc, Ft)
            py += 12

        py += 4
        txt(surf, "Resource Pipeline:", px, py, P["amb"], Ft); py+=11
        rs = env.resource_state
        for i,r in enumerate(CONFIG["resources"]["consumable"]):
            inv  = rs.consumable_inventory[i]
            pipe = rs.pending_orders[i].sum()
            txt(surf, f"  {r['name'][:8]:8s} inv:{inv:.0f} pipe:{pipe:.0f}",
                px, py, P["txt"], Ft); py+=11

        py += 8
        txt(surf, "LAST ACTION", px, py, P["dim"], Ft); py+=14
        txt(surf, "Maintenance decisions:", px, py, P["amb"], Ft); py+=11

        for i, s in enumerate(states):
            name = CONFIG["machines"][i]["name"].split()[0][:6]
            act  = runner.last_a1_maint[i] if runner.last_a1_maint else 0
            mask = runner.last_mask1[i]    if runner.last_mask1 else [True,True,True]

            # Action taken
            act_str = ["NONE","PM ✓","CM ✓"][act]
            act_col = [P["dim"],P["PM"],P["CM"]][act]
            txt(surf, f"  {name:6s}", px,py, P["txt"], Ft)
            txt(surf, act_str, px+54,py, act_col, Ft)

            # Masked actions shown as red X
            mx = px+100
            for ai, (aname2, mc) in enumerate([("∅",P["dim"]),("PM",P["PM"]),("CM",P["CM"])]):
                allowed = mask[ai] if ai < len(mask) else True
                bc = mc if allowed else (60,20,20)
                tc = mc if allowed else P["FAIL"]
                rect(surf, bc, mx,py,18,10, 2, alpha=80 if allowed else 120)
                txt(surf, "✗" if not allowed else aname2, mx+2,py, tc if allowed else P["red"], Ft)
                if not allowed:
                    pygame.draw.line(surf, P["red"],(mx,py),(mx+18,py+10),1)
                mx += 22
            py += 12

        py += 6
        txt(surf, "WHY MASKED:", px, py, P["dim"], Ft); py+=11
        for reason in runner.mask_reasons[-4:]:
            txt(surf, f"  • {reason[:38]}", px, py, P["red"], Ft); py+=10

        py += 8
        txt(surf, "REWARD DECOMPOSITION", px, py, P["dim"], Ft); py+=14
        items = [
            ("Availability",  runner.r_avail,  P["grn"]),
            ("Maint cost",   -runner.r_maint,  P["CM"]),
            ("Fail penalty",  runner.r_fail1,   P["red"]),
            ("r1 TOTAL",      runner.ep_r1/max(runner.step_n,1), P["amb"]),
        ]
        for label, val, col in items:
            txt(surf, f"  {label:14s}", px, py, P["dim"], Ft)
            txt(surf, f"{val:>+7.2f}", px+pw-50, py, col, Ft)
            py += 11

    def _draw_agent2(self, surf, runner, px, py, pw, t):
        env = runner.env

        txt(surf, "TRIPARTITE GRAPH STATE", px, py, P["dim"], Ft); py+=14

        # Mini TGIN graph visualisation (representational)
        gx, gy, gw, gh = px, py, pw, 100
        rect(surf, P["panel2"], gx,gy,gw,gh, 4)
        border(surf, P["a2"], gx,gy,gw,gh,1,4)

        # Three layers: Ops | Machines | Jobs
        jobs_active = [j for j in env.jobs if not j.is_complete]
        ops_pending = [(j.job_id,k,op) for j in jobs_active
                       for k,op in enumerate(j.operations) if op.status in (0,1,2)]

        n_ops  = min(len(ops_pending),8)
        n_mach = len(env.machine_states)
        n_jobs = min(len(jobs_active),6)

        layers = [
            ("OPS",  n_ops,  gx+gw*0.15, P["j_run"]),
            ("MACH", n_mach, gx+gw*0.50, P["a1"]),
            ("JOBS", n_jobs, gx+gw*0.85, P["a2"]),
        ]
        node_pos = {}
        for lname, count, lx, lc in layers:
            if count == 0: continue
            spacing = min(gh//(count+1), 16)
            for i in range(count):
                ny = gy+gh//(count+1)*(i+1)
                nx = int(lx)
                glow_circle(surf, lc, nx,ny, 5, 60)
                pygame.draw.circle(surf, lc, (nx,ny), 5)
                node_pos[(lname,i)] = (nx,ny)
            txt(surf, lname, int(lx),gy+2, lc, Ft, "c")

        # Draw edges (random representational)
        random.seed(42)
        for i in range(min(n_ops,8)):
            for j in range(min(n_mach,5)):
                if random.random() < 0.4:
                    p1 = node_pos.get(("OPS",i))
                    p2 = node_pos.get(("MACH",j))
                    if p1 and p2:
                        pygame.draw.line(surf, (*P["a2"],40), p1,p2,1)
        for i in range(min(n_mach,5)):
            for j in range(min(n_jobs,6)):
                if random.random() < 0.5:
                    p1 = node_pos.get(("MACH",i))
                    p2 = node_pos.get(("JOBS",j))
                    if p1 and p2:
                        pygame.draw.line(surf, (*P["a1"],40), p1,p2,1)

        txt(surf, "←msg pass→", gx+gw//2,gy+gh-10, P["dim"],Ft,"c")
        py += gh + 6

        txt(surf, "LAST ACTION", px, py, P["dim"], Ft); py+=14
        if runner.last_a2:
            j,k,m = runner.last_a2
            mname = CONFIG["machines"][m]["name"] if m<5 else "?"
            txt(surf, f"  Assigned: Job {j} Op {k}", px, py, P["a2"], Ft); py+=11
            txt(surf, f"  Machine:  {mname}",        px, py, P["a2"], Ft); py+=11
            txt(surf, f"  Reason:   health={env.machine_states[m].health:.0f}%"
                      f" λ={env.machine_states[m].hazard_rate:.2e}",
                px, py, P["dim"], Ft); py+=11
        else:
            txt(surf, "  WAIT (no valid pairs)", px, py, P["dim"], Ft); py+=11

        py += 4
        txt(surf, f"Valid (op,mach) pairs: {len(env._valid_pairs)}", px, py, P["dim"], Ft); py+=11
        txt(surf, "Masked: unavail/maint/busy mach", px, py, P["red"], Ft); py+=14

        txt(surf, "REWARD DECOMPOSITION", px, py, P["dim"], Ft); py+=14
        items = [
            ("Tardiness",  runner.r_tard,   P["red"]),
            ("Completion", runner.r_comp,   P["grn"]),
            ("Hlth bonus", runner.r_health, P["a2"]),
            ("Fail share", runner.r_fail2,  P["red"]),
            ("r2 TOTAL",   runner.ep_r2/max(runner.step_n,1), P["a2"]),
        ]
        for label, val, col in items:
            txt(surf, f"  {label:14s}", px, py, P["dim"], Ft)
            txt(surf, f"{val:>+7.2f}", px+pw-50, py, col, Ft)
            py += 11


# ══════════════════════════════════════════════════════════════════
# JOB KANBAN PANEL
# ══════════════════════════════════════════════════════════════════

class JobsPanel:
    def __init__(self):
        self.open = False
        self.anim = 0.0
        self.view = "kanban"   # 'kanban' or 'gantt'

    def toggle(self): self.open = not self.open
    def toggle_view(self): self.view = "gantt" if self.view=="kanban" else "kanban"

    def update(self, dt):
        target = 1.0 if self.open else 0.0
        self.anim += (target-self.anim)*min(dt*8,1)

    def draw(self, surf, runner, a1_panel, a2_panel, t_anim):
        # Base height when closed (tab only)
        tab_h  = 28
        panel_h= int(lerp(tab_h, 240, self.anim))
        lx     = a1_panel.width
        rx     = W - a2_panel.width
        pw     = rx - lx
        py     = H - panel_h

        rect(surf, P["panel"], lx, py, pw, panel_h)
        border(surf, P["a2"], lx, py, pw, panel_h, 1)

        # Tab row
        txt(surf, "  JOBS & OPERATIONS", lx+10, py+8, P["a2"], Fm)
        vbtn = "[GANTT]" if self.view=="kanban" else "[KANBAN]"
        txt(surf, vbtn, lx+pw-90, py+8, P["dim"], Fs)
        txt(surf, "J: toggle", lx+pw-200, py+8, P["dim"], Ft)

        if self.anim < 0.05: return

        jobs = runner.env.jobs
        t    = runner.env.current_step
        T    = CONFIG["episode"]["t_max_train"]

        if self.view == "kanban":
            self._draw_kanban(surf, jobs, t, lx, py+28, pw, panel_h-28)
        else:
            self._draw_gantt(surf, jobs, t, T, lx, py+28, pw, panel_h-28)

    def _draw_kanban(self, surf, jobs, t, x, y, w, h):
        cols = ["PENDING","IN PROGRESS","DONE","LATE"]
        cw   = w//4
        col_colors = [P["j_wait"],P["j_run"],P["j_done"],P["j_late"]]

        pending = [j for j in jobs if not j.is_complete and
                   all(op.status==0 for op in j.operations)]
        running = [j for j in jobs if not j.is_complete and
                   any(op.status==2 for op in j.operations)]
        done    = [j for j in jobs if j.is_complete and j.tardiness==0]
        late    = [j for j in jobs if j.is_complete and j.tardiness>0] + \
                  [j for j in jobs if not j.is_complete and t>j.due_date]

        groups = [pending, running, done, late]

        for ci, (cname, cjobs, cc) in enumerate(zip(cols,groups,col_colors)):
            cx = x + ci*cw
            rect(surf, P["panel2"], cx+2,y,cw-4,h,4)
            border(surf, cc, cx+2,y,cw-4,h,1,4)
            txt(surf, f"{cname} ({len(cjobs)})", cx+cw//2,y+4, cc, Ft,"c")

            for ji, job in enumerate(cjobs[:5]):
                jy = y+20+ji*34
                jx = cx+6
                jw = cw-12
                urgency = max(0, min(1, (job.due_date-t)/max(job.due_date,1)))
                jc  = lerpC(P["j_late"],cc, urgency) if cname!="DONE" else cc
                rect(surf, (*jc,40), jx,jy,jw,30, 4, alpha=60)
                border(surf, jc, jx,jy,jw,30,1,4)

                txt(surf, f"J{job.job_id} w={job.weight:.0f}", jx+4,jy+3, P["txt"],Ft)
                txt(surf, f"due:{job.due_date:.0f}", jx+4,jy+14, P["dim"],Ft)

                # Op dots
                for k, op in enumerate(job.operations[:6]):
                    dot_col = [P["dim"],P["j_wait"],P["j_run"],P["j_done"]][op.status]
                    dx = jx+jw-8-k*10
                    pygame.draw.circle(surf, dot_col, (dx,jy+8), 4)

                # Progress bar
                hbar(surf, jx+2,jy+26,jw-4,3, job.completion_ratio, jc, P["j_wait"])

            if len(cjobs)>5:
                txt(surf, f"+{len(cjobs)-5} more", cx+cw//2, y+20+5*34+2, P["dim"],Ft,"c")

    def _draw_gantt(self, surf, jobs, t, T, x, y, w, h):
        label_w = 60
        bar_x   = x+label_w+4
        bar_w   = w-label_w-14
        row_h   = min(26, (h-16)//max(len(jobs),1))

        # Time axis
        for pct in [0.25,0.5,0.75,1.0]:
            tx = bar_x+int(bar_w*pct)
            pygame.draw.line(surf, P["border"], (tx,y),(tx,y+h-4),1)
            txt(surf, f"t={int(T*pct)}", tx,y, P["dim"],Ft,"c")

        # Current time
        ctx = bar_x+int(bar_w*min(t/T,1.0))
        pygame.draw.line(surf, P["a2"], (ctx,y),(ctx,y+h-4),2)

        for ri, job in enumerate(jobs[:8]):
            ry  = y+16+ri*row_h
            late = not job.is_complete and t>job.due_date
            jc   = P["j_late"] if late else (P["j_done"] if job.is_complete else P["j_run"])

            txt(surf, f"J{job.job_id}", x+4,ry+row_h//2-5, jc,Ft)

            # Background
            rect(surf, P["panel2"], bar_x,ry+2,bar_w,row_h-4,2)

            # Fill
            fw = int(bar_w*job.completion_ratio)
            if fw: rect(surf, (*jc,120), bar_x,ry+2,fw,row_h-4,2,alpha=120)

            # Due date line
            ddx = bar_x+int(bar_w*min(job.due_date/T,1.0))
            pygame.draw.line(surf,(255,50,70),(ddx,ry),(ddx,ry+row_h),1)

            # Op segments
            n = max(job.n_ops,1)
            for k,op in enumerate(job.operations):
                ox = bar_x+int(bar_w*k/n)
                ow = max(int(bar_w/n)-1,3)
                oc = [P["j_wait"],P["panel"],P["a2"],P["j_done"]][op.status]
                if op.status in (1,2,3):
                    rect(surf, (*oc,180), ox,ry+4,ow,row_h-8,2,alpha=180)

            if job.tardiness>0:
                txt(surf, f"+{job.tardiness:.0f}!", bar_x+bar_w+2,ry+2, P["red"],Ft)


# ══════════════════════════════════════════════════════════════════
# RESOURCE OVERLAY
# ══════════════════════════════════════════════════════════════════

class ResourceOverlay:
    def __init__(self):
        self.open = False
        self.anim = 0.0

    def toggle(self): self.open = not self.open
    def update(self, dt):
        target = 1.0 if self.open else 0.0
        self.anim += (target-self.anim)*min(dt*8,1)

    def draw(self, surf, runner, lx, rx, t_anim):
        if self.anim < 0.02: return
        pw  = rx-lx
        oh  = int(lerp(0, 200, self.anim))
        ox, oy = lx, 40

        rect(surf, P["panel"], ox,oy,pw,oh, alpha=220)
        border(surf, P["r_ren"], ox,oy,pw,oh, 1)
        txt(surf, "RESOURCES — T: close", ox+pw//2,oy+4, P["r_ren"],Fm,"c")

        if self.anim < 0.2: return
        rs   = runner.env.resource_state
        ren  = CONFIG["resources"]["renewable"]
        con  = CONFIG["resources"]["consumable"]
        all_r = [(r,True) for r in ren] + [(r,False) for r in con]
        cw   = pw // len(all_r)

        for i, (r, renewable) in enumerate(all_r):
            rx2  = ox+i*cw+4
            col  = P["r_ren"] if renewable else P["r_con"]
            name = r["name"][:10]
            txt(surf, name, rx2+cw//2, oy+24, col, Ft,"c")

            if renewable:
                avl = int(rs.renewable_available[i])
                cap = int(rs.renewable_capacity[i])
                frac = avl/max(cap,1)
                hbar(surf, rx2,oy+38,cw-8,80, frac, col, P["panel2"])
                txt(surf, f"{avl}/{cap}", rx2+cw//2,oy+82, col,Fs,"c")
                txt(surf, "RENEW", rx2+cw//2,oy+96, P["dim"],Ft,"c")
            else:
                ci   = i - len(ren)
                inv  = rs.consumable_inventory[ci]
                cap2 = r["initial_inventory"]*1.5
                frac = min(inv/max(cap2,1),1.0)
                hbar(surf, rx2,oy+38,cw-8,40, frac, col, P["j_late"])
                txt(surf, f"{inv:.0f}", rx2+cw//2,oy+52, col,Fs,"c")
                # Pipeline
                txt(surf, "PIPELINE:", rx2,oy+88, P["dim"],Ft)
                lead = r["lead_time_shifts"]
                sw   = max((cw-8)//max(lead,1),4)
                for lag in range(min(lead,6)):
                    qty = rs.pending_orders[ci,lag] if lag<rs.pending_orders.shape[1] else 0
                    pc  = col if qty>0 else P["panel2"]
                    rect(surf, pc, rx2+lag*sw,oy+98,sw-1,12,2)
                    if qty>0: txt(surf,f"{qty:.0f}",rx2+lag*sw+1,oy+99,P["hi"],Ft)
                txt(surf,"→ARRIVE",rx2,oy+114,P["dim"],Ft)


# ══════════════════════════════════════════════════════════════════
# ENVIRONMENT RUNNER
# ══════════════════════════════════════════════════════════════════

class EnvRunner:
    def __init__(self):
        self.env = ManufacturingEnv(CONFIG)
        self.log = collections.deque(maxlen=80)
        self.reset_all()

    def reset_all(self):
        obs_dict, _ = self.env.reset(seed=random.randint(0,9999))
        self.done = False
        self.step_n = 0
        self.ep_n   = 0
        self.ep_r1  = 0.0
        self.ep_r2  = 0.0
        self.total_failures = 0
        self.total_pm = 0
        self.total_cm = 0
        self.last_a1_maint  = [0]*5
        self.last_a2        = None
        self.last_mask1     = [[True,False,False]]*5
        self.mask_reasons   = []
        self.history_health = {m:collections.deque([100.0]*80,maxlen=80) for m in range(5)}
        # reward decomp accumulators
        self.r_avail=0.; self.r_maint=0.; self.r_fail1=0.
        self.r_tard=0.;  self.r_comp=0.;  self.r_health=0.; self.r_fail2=0.
        self.log.append(("System initialised", P["grn"]))
        self.log.append(("Rule-based: PM<45hp | CM on FAIL", P["dim"]))

    def _rule_action1(self):
        maint = []
        reasons = []
        masks  = []
        for s in self.env.machine_states:
            i    = s.machine_id
            name = CONFIG["machines"][i]["name"].split()[0]
            # Build mask for display
            busy = self.env.machine_busy[i]
            mask = [
                True,                                   # NONE always ok
                s.status==MachineStatus.OP and not busy, # PM
                s.status==MachineStatus.FAIL,            # CM
            ]
            masks.append(mask)

            if s.status == MachineStatus.FAIL:
                maint.append(2); self.total_cm+=1
                self.log.append((f"[A1] CM initiated: {name}", P["CM"]))
            elif s.status==MachineStatus.OP and not busy and s.health<45:
                maint.append(1); self.total_pm+=1
                self.log.append((f"[A1] PM initiated: {name} hp={s.health:.0f}", P["PM"]))
            else:
                maint.append(0)
                if busy:     reasons.append(f"{name}: PM masked — machine busy")
                if s.status!=MachineStatus.OP and s.status!=MachineStatus.FAIL:
                    reasons.append(f"{name}: already in maintenance")

        self.last_a1_maint = maint
        self.last_mask1    = masks
        self.mask_reasons  = reasons[-4:]
        # Fake reward decomp
        self.r_avail = sum(1 for s in self.env.machine_states
                           if s.status==MachineStatus.OP)/5 * 2.0
        self.r_maint = maint.count(1)*1.0 + maint.count(2)*3.0
        return {
            "maintenance": np.array(maint,dtype=int),
            "reorder":     np.zeros(3,dtype=float),
        }

    def step(self):
        if self.done:
            self.ep_n+=1; self.ep_r1=0.; self.ep_r2=0.
            self.log.append(("━━━ NEW EPISODE ━━━", P["amb"]))
            self.env.reset(seed=random.randint(0,9999)); self.done=False; return

        action1 = self._rule_action1()
        self.env._step_agent1(action1)

        pairs = self.env._valid_pairs
        if pairs:
            action2_idx = 0
            j,k,m = pairs[0]
            self.last_a2 = (j,k,m)
            mname = CONFIG["machines"][m]["name"].split()[0]
            self.log.append((f"[A2] J{j} Op{k} → {mname}", P["a2"]))
            # Fake reward decomp
            self.r_comp   = 0.0
            self.r_health = self.env.machine_states[m].health/100*0.5
            self.r_tard   = -sum(j2.tardiness for j2 in self.env.jobs)/200
        else:
            action2_idx   = 0
            self.last_a2  = None
            self.r_comp   = 0.; self.r_health=0.; self.r_tard=0.

        self.env._step_agent2(action2_idx)
        self.env._resolve_physics()
        self.env._compute_rewards()

        r1 = self.env.rewards[AGENT_PDM]
        r2 = self.env.rewards[AGENT_JOBSHOP]
        self.ep_r1 += r1; self.ep_r2 += r2

        self.r_fail1 = -len(self.env._newly_failed)*30*0.3
        self.r_fail2 = self.r_fail1

        for m in self.env._newly_failed:
            name = CONFIG["machines"][m]["name"].split()[0]
            self.log.append((f"[FAIL] {name} FAILED! CM required", P["FAIL"]))
            self.total_failures+=1

        for jid in self.env._completed_job_ids:
            self.log.append((f"[DONE] Job {jid} completed ✓", P["grn"]))
            self.r_comp += 3.0

        for s in self.env.machine_states:
            self.history_health[s.machine_id].append(s.health)

        self.step_n+=1
        self.done = self.env.terminations[AGENT_PDM] or self.env.truncations[AGENT_PDM]


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════

def draw_header(surf, runner, paused, speed, t_anim):
    rect(surf, P["panel2"], 0,0,W,38)
    pygame.draw.line(surf, P["a2"], (0,38),(W,38),1)

    # Scrolling title
    title = "  MANUFACTURING MARL — THREE-TIER OPTIMIZATION FRAMEWORK  "
    tw    = Fb.render(title, True, P["cyan"]).get_width()
    offset= int(t_anim*30) % (tw+W)
    for ox in [-tw, 0, tw]:
        ts = Fb.render(title, True, P["cyan"])
        surf.blit(ts, (W//2 - tw//2 + ox - offset, 6))

    # Status
    sc  = P["amb"] if paused else P["grn"]
    sl  = "⏸ PAUSED" if paused else f"▶ {speed}x"
    txt(surf, sl, W-90,10, sc, Fm)

    # Controls hint
    txt(surf, "SPACE·pause  →·step  ↑↓·speed  1·Agent1  2·Agent2  J·jobs  T·resources  R·reset  Q·quit",
        10,26, P["dim"], Ft)

    # Episode info
    e  = runner.ep_n
    s  = runner.step_n
    t  = runner.env.current_step
    T  = CONFIG["episode"]["t_max_train"]
    txt(surf, f"EP:{e}  STEP:{s}  t:{t}/{T}", W//2,8, P["txt"],Fm,"c")


# ══════════════════════════════════════════════════════════════════
# FACTORY FLOOR
# ══════════════════════════════════════════════════════════════════

def draw_floor(surf, runner, lx, rx, top_y, bot_y, t_anim):
    fw = rx-lx
    fh = bot_y-top_y

    # Floor
    rect(surf, P["floor"], lx,top_y,fw,fh)

    # Grid
    gs = 40
    for gx in range(lx, rx, gs):
        pygame.draw.line(surf, P["grid"],(gx,top_y),(gx,bot_y),1)
    for gy in range(top_y, bot_y, gs):
        pygame.draw.line(surf, P["grid"],(lx,gy),(rx,gy),1)

    border(surf, P["border"], lx,top_y,fw,fh)

    # Machine positions (2 rows)
    n     = len(runner.env.machine_states)
    mw,mh = 140,160
    gap   = (fw - n*mw)//(n+1)
    row_y = top_y + (fh-mh)//2

    states = runner.env.machine_states
    busy   = runner.env.machine_busy

    for i,s in enumerate(states):
        mx = lx + gap + i*(mw+gap)
        my = row_y
        kind = CONFIG["machines"][i]["name"]

        draw_machine_icon(surf, kind, mx,my,mw,mh,
                          s.status, s.health, busy[i], t_anim,
                          s.maint_steps_remaining)

        # Animated beam from Agent 1 if it acted on this machine
        if runner.last_a1_maint and runner.last_a1_maint[i] in (1,2):
            ac = P["PM"] if runner.last_a1_maint[i]==1 else P["CM"]
            beam_y = my - 30
            alpha  = int(100*pulse(t_anim,3))
            glow_circle(surf, ac, mx+mw//2, my,20,alpha)
            pygame.draw.line(surf, ac,(mx+mw//2,beam_y),(mx+mw//2,my),2)
            alabel = "PM" if runner.last_a1_maint[i]==1 else "CM"
            txt(surf, f"[A1] {alabel}", mx+mw//2,beam_y-14, ac,Ft,"c")

        # Animated beam from Agent 2 if scheduling this machine
        if runner.last_a2 and runner.last_a2[2]==i:
            beam_y = my+mh+20
            glow_circle(surf, P["a2"], mx+mw//2,my+mh,16,int(80*pulse(t_anim,4)))
            pygame.draw.line(surf, P["a2"],(mx+mw//2,my+mh),(mx+mw//2,beam_y),2)
            j,k,_ = runner.last_a2
            txt(surf, f"[A2] J{j}·Op{k}", mx+mw//2,beam_y+4, P["a2"],Ft,"c")

        # Machine label below
        sc   = STATUS_COL[s.status]
        name = CONFIG["machines"][i]["name"]
        txt(surf, name, mx+mw//2, my+mh+30, P["txt"],Ft,"c")
        txt(surf, STATUS_NAME[s.status], mx+mw//2,my+mh+42, sc,Ft,"c")
        txt(surf, f"hp:{s.health:.0f}%", mx+mw//2,my+mh+54, healthC(s.health),Ft,"c")

        # Health sparkline below machine
        hist = list(runner.history_health[i])[-60:]
        if len(hist)>1:
            pts = []
            sw,sh = mw,18
            sy = my+mh+68
            for ji,hv in enumerate(hist):
                px2 = mx + int(ji/len(hist)*sw)
                py2 = sy+sh - int(hv/100*sh)
                pts.append((px2,py2))
            if len(pts)>1:
                pygame.draw.lines(surf, healthC(s.health), False, pts, 1)
            pygame.draw.rect(surf,P["border"],(mx,sy,mw,sh),1)

    # Connection lines between machines (conveyor feel)
    for i in range(n-1):
        x1 = lx+gap+(i+1)*(mw+gap)-gap//2
        x2 = lx+gap+(i+1)*(mw+gap)
        y2 = row_y + mh//2
        pygame.draw.line(surf, P["border"],(x1,y2),(x2,y2),2)
        # Animated packet on conveyor if job running
        if busy[i] or busy[i+1]:
            prog = (t_anim*0.5) % 1.0
            px3  = int(x1+(x2-x1)*prog)
            pygame.draw.circle(surf, P["a2"],(px3,y2),4)


# ══════════════════════════════════════════════════════════════════
# EVENT LOG
# ══════════════════════════════════════════════════════════════════

def draw_log(surf, runner, x, y, w, h):
    rect(surf, P["panel"], x,y,w,h,4)
    border(surf, P["border"], x,y,w,h,1,4)
    txt(surf, "EVENT LOG", x+8,y+6, P["dim"],Ft)
    pygame.draw.line(surf, P["border"],(x,y+18),(x+w,y+18),1)

    entries = list(runner.log)
    max_rows = (h-22)//13
    for i, (msg, col) in enumerate(entries[-max_rows:]):
        alpha = 80 + int(175*(i/max(max_rows-1,1)))
        c     = tuple(min(255,int(col[j]*alpha/255)) for j in range(3))
        txt(surf, msg[:w//7], x+6, y+20+i*13, c, Ft)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    runner   = EnvRunner()
    a1_panel = AgentPanel(1, "left",  P["a1"])
    a2_panel = AgentPanel(2, "right", P["a2"])
    jobs_pan = JobsPanel()
    res_ov   = ResourceOverlay()

    paused    = False
    speeds    = [0.5,1,2,5,10,20]
    spd_idx   = 2
    frame_acc = 0.0
    t_anim    = 0.0

    runner.log.append(("Press 1/2 to open Agent panels", P["dim"]))
    runner.log.append(("Press J for jobs, T for resources", P["dim"]))

    while True:
        dt = clock.tick(60)/1000.0
        t_anim += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE): pygame.quit(); sys.exit()
                if k == pygame.K_SPACE:  paused = not paused
                if k == pygame.K_RIGHT and paused: runner.step()
                if k == pygame.K_UP:    spd_idx = min(spd_idx+1,len(speeds)-1)
                if k == pygame.K_DOWN:  spd_idx = max(spd_idx-1,0)
                if k == pygame.K_r:     runner.reset_all()
                if k == pygame.K_1:     a1_panel.toggle()
                if k == pygame.K_2:     a2_panel.toggle()
                if k == pygame.K_j:
                    if jobs_pan.open and jobs_pan.anim>0.9:
                        jobs_pan.toggle_view()
                    else:
                        jobs_pan.toggle()
                if k == pygame.K_t:     res_ov.toggle()

        if not paused:
            frame_acc += dt
            interval   = 1.0/speeds[spd_idx]
            while frame_acc >= interval:
                runner.step(); frame_acc -= interval

        a1_panel.update(dt); a2_panel.update(dt)
        jobs_pan.update(dt); res_ov.update(dt)

        screen.fill(P["bg"])

        lx = a1_panel.width
        rx = W - a2_panel.width
        jh = int(lerp(28,240,jobs_pan.anim))
        top_y = 40
        bot_y = H - jh

        draw_header(screen, runner, paused, speeds[spd_idx], t_anim)
        draw_floor(screen, runner, lx, rx, top_y, bot_y-4, t_anim)

        # Log panel (top right gap, only if a2 not fully open)
        if a2_panel.anim < 0.5:
            log_w = max(W-rx-8,50) if rx<W-50 else 0
        # Draw log inside a2 panel gap
        log_x = rx+2
        log_w = W-rx-2
        if log_w > 60:
            draw_log(screen, runner, log_x,top_y, log_w, bot_y-top_y-4)

        a1_panel.draw(screen, runner, t_anim)
        a2_panel.draw(screen, runner, t_anim)
        jobs_pan.draw(screen, runner, a1_panel, a2_panel, t_anim)
        res_ov.draw(screen, runner, lx, rx, t_anim)

        # Scanlines
        sl_surf = pygame.Surface((W,H), pygame.SRCALPHA)
        for y in range(0,H,3): pygame.draw.line(sl_surf,(0,0,0,18),(0,y),(W,y))
        screen.blit(sl_surf,(0,0))

        pygame.display.flip()

if __name__=="__main__":
    main()
