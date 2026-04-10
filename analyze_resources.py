"""
analyze_resources.py
=====================
Runs one episode and tracks ALL resource state at every timestep.
Produces sawtooth inventory plots, order event markers, and resource summary.

Usage:
    python analyze_resources.py                          # default seed
    python analyze_resources.py --seed 42 --policy smart
    python analyze_resources.py --checkpoint outputs/checkpoints/latest.pt

Output:
    outputs/resource_analysis/
    ├── consumable_inventory.png       # Sawtooth plot: 3 consumables over 150 steps
    ├── renewable_availability.png     # Step plot: 3 renewables over 150 steps
    ├── order_events.png               # When orders placed + when they arrive
    ├── resource_summary.csv           # Per-step raw data
    └── analysis_report.txt            # Text summary
"""
import argparse, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environments.mfg_env import ManufacturingEnv, AGENT_PDM
from environments.transitions.degradation import MachineStatus
from environments.transitions.job_dynamics import OpStatus


def run_episode_with_tracking(config, seed=42, policy="smart"):
    env = ManufacturingEnv(config)
    env.reset(seed=seed)

    steps = []
    for step in range(env.t_max):
        # ── Record state BEFORE actions ──────────────────────
        cons_inv = env.resource_state.consumable_inventory.copy()
        ren_avail = env.resource_state.renewable_available.copy()
        ren_cap = env.resource_state.renewable_capacity.copy()
        pipeline = env.resource_state.pending_orders.copy() if hasattr(env.resource_state, 'pending_orders') else np.zeros(3)
        
        healths = [s.health for s in env.machine_states]
        statuses = [s.status for s in env.machine_states]
        busy = env.machine_busy[:]
        n_valid = len(env._valid_pairs)

        # ── Choose actions ───────────────────────────────────
        if policy == "smart":
            maint = np.array([1 if env.machine_states[i].status == MachineStatus.OP
                               and not env.machine_busy[i]
                               and env.machine_states[i].health < 75
                               else 0 for i in range(5)], dtype=int)
            inv = env.resource_state.consumable_inventory
            reorder = np.array([8.0 if inv[i] < 10 else 0.0 for i in range(len(inv))])
        elif policy == "reactive":
            maint = np.zeros(5, dtype=int)
            reorder = np.array([5.0 if env.resource_state.consumable_inventory[i] < 8 else 0.0
                                for i in range(len(env.resource_state.consumable_inventory))])
        else:  # random
            rng = np.random.default_rng(seed + step)
            maint = (rng.random(5) < 0.1).astype(int)
            reorder = rng.choice([0, 3, 8], size=len(env.resource_state.consumable_inventory)).astype(float)

        env._step_agent1({"maintenance": maint, "reorder": reorder})
        
        # Record what was actually ordered and PM'd
        pm_applied = env._pm_applied_this_step[:]
        order_placed = reorder.copy()

        if env._valid_pairs:
            env._step_agent2(0)
        else:
            env._step_agent2(None)
        
        env._resolve_physics()
        env._compute_rewards()

        # ── Record state AFTER actions ───────────────────────
        cons_inv_after = env.resource_state.consumable_inventory.copy()
        ren_avail_after = env.resource_state.renewable_available.copy()

        steps.append({
            "step": step,
            "cons_0": cons_inv[0], "cons_1": cons_inv[1], "cons_2": cons_inv[2],
            "cons_0_after": cons_inv_after[0], "cons_1_after": cons_inv_after[1], "cons_2_after": cons_inv_after[2],
            "ren_0": ren_avail[0], "ren_1": ren_avail[1], "ren_2": ren_avail[2],
            "ren_0_after": ren_avail_after[0], "ren_1_after": ren_avail_after[1], "ren_2_after": ren_avail_after[2],
            "ren_cap_0": ren_cap[0], "ren_cap_1": ren_cap[1], "ren_cap_2": ren_cap[2],
            "order_0": order_placed[0], "order_1": order_placed[1], "order_2": order_placed[2],
            "pm_0": pm_applied[0], "pm_1": pm_applied[1], "pm_2": pm_applied[2],
            "pm_3": pm_applied[3], "pm_4": pm_applied[4],
            "n_pm_this_step": sum(pm_applied),
            "n_cm_this_step": env._auto_cm_count,
            "n_valid_pairs": n_valid,
            "h_0": healths[0], "h_1": healths[1], "h_2": healths[2], "h_3": healths[3], "h_4": healths[4],
            "status_str": "".join(["O", "P", "C", "F"][s] for s in statuses),
            "busy_str": "".join("B" if b else "." for b in busy),
            "failures": env._episode_failures,
            "completions": env._episode_completions,
            "r1": env.rewards[AGENT_PDM],
            "r2": env.rewards["jobshop_agent"],
            "ordering_cost": env._last_ordering_cost,
        })

    return steps, env


def plot_consumables(steps, outdir, config):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    con_names = [c["name"] for c in config["resources"]["consumable"]]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    for i, (ax, name, color) in enumerate(zip(axes, con_names, colors)):
        t = [s["step"] for s in steps]
        inv_before = [s[f"cons_{i}"] for s in steps]
        inv_after = [s[f"cons_{i}_after"] for s in steps]
        orders = [s[f"order_{i}"] for s in steps]
        
        ax.plot(t, inv_after, color=color, linewidth=1.5, label=f"{name} inventory")
        ax.fill_between(t, 0, inv_after, alpha=0.15, color=color)
        
        # Mark order events
        order_steps = [s["step"] for s in steps if s[f"order_{i}"] > 0]
        order_vals = [s[f"cons_{i}_after"] for s in steps if s[f"order_{i}"] > 0]
        order_qtys = [s[f"order_{i}"] for s in steps if s[f"order_{i}"] > 0]
        if order_steps:
            ax.scatter(order_steps, order_vals, color="gold", s=40, zorder=5, 
                      marker="^", label=f"Order placed ({len(order_steps)} times)")
        
        # Mark PM consumption events
        pm_steps = [s["step"] for s in steps if s["n_pm_this_step"] > 0]
        pm_vals = [s[f"cons_{i}_after"] for s in steps if s["n_pm_this_step"] > 0]
        if pm_steps:
            ax.scatter(pm_steps, pm_vals, color="blue", s=30, zorder=5,
                      marker="v", label=f"PM consumed ({len(pm_steps)} events)")
        
        # Mark CM consumption events
        cm_steps = [s["step"] for s in steps if s["n_cm_this_step"] > 0]
        cm_vals = [s[f"cons_{i}_after"] for s in steps if s["n_cm_this_step"] > 0]
        if cm_steps:
            ax.scatter(cm_steps, cm_vals, color="red", s=30, zorder=5,
                      marker="x", label=f"CM consumed ({len(cm_steps)} events)")
        
        # Reorder point line
        rop = config["resources"]["consumable"][i].get("reorder_point", 8)
        ax.axhline(y=rop, color="orange", linestyle="--", alpha=0.5, label=f"ROP={rop}")
        
        ax.set_ylabel(f"{name}\n(units)", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    
    axes[-1].set_xlabel("Step (shift)", fontsize=12)
    axes[0].set_title("Consumable Inventory Over Episode (Sawtooth Pattern)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "consumable_inventory.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: consumable_inventory.png")


def plot_renewables(steps, outdir, config):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    ren_names = [r["name"] for r in config["resources"]["renewable"]]
    colors = ["#9b59b6", "#f39c12", "#1abc9c"]
    
    for i, (ax, name, color) in enumerate(zip(axes, ren_names, colors)):
        t = [s["step"] for s in steps]
        avail = [s[f"ren_{i}_after"] for s in steps]
        cap = [s[f"ren_cap_{i}"] for s in steps]
        
        ax.step(t, avail, where="post", color=color, linewidth=2, label=f"{name} available")
        ax.step(t, cap, where="post", color=color, linewidth=1, linestyle="--", alpha=0.4, label=f"Capacity")
        ax.fill_between(t, 0, avail, step="post", alpha=0.15, color=color)
        
        ax.set_ylabel(f"{name}\n(units)", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0, top=max(cap) + 1)
    
    axes[-1].set_xlabel("Step (shift)", fontsize=12)
    axes[0].set_title("Renewable Resource Availability Over Episode", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "renewable_availability.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: renewable_availability.png")


def plot_orders(steps, outdir, config):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    con_names = [c["name"] for c in config["resources"]["consumable"]]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    # Top: order quantities per step
    ax = axes[0]
    for i, (name, color) in enumerate(zip(con_names, colors)):
        t = [s["step"] for s in steps]
        qty = [s[f"order_{i}"] for s in steps]
        ax.bar([x + i*0.25 for x in t], qty, width=0.25, color=color, alpha=0.7, label=name)
    ax.set_ylabel("Order qty", fontsize=11)
    ax.set_title("Order Events Over Episode", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    # Bottom: cumulative ordering cost
    ax2 = axes[1]
    cum_cost = np.cumsum([s["ordering_cost"] for s in steps])
    ax2.plot(t, cum_cost, color="black", linewidth=2)
    ax2.fill_between(t, 0, cum_cost, alpha=0.1, color="black")
    ax2.set_ylabel("Cumulative\nordering cost", fontsize=11)
    ax2.set_xlabel("Step (shift)", fontsize=12)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "order_events.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: order_events.png")


def plot_combined_timeline(steps, outdir):
    fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=True)
    t = [s["step"] for s in steps]
    
    # 1. Machine health
    for m in range(5):
        axes[0].plot(t, [s[f"h_{m}"] for s in steps], linewidth=1, label=f"M{m}")
    axes[0].axhline(y=75, color="red", linestyle="--", alpha=0.5, label="PM gate")
    axes[0].set_ylabel("Health %"); axes[0].legend(fontsize=8, ncol=6); axes[0].grid(alpha=0.3)
    axes[0].set_title("Episode Timeline: Health → Resources → Rewards", fontsize=14, fontweight="bold")
    
    # 2. Machine status
    status_map = {"O": 0, "P": 1, "C": 2, "F": 3}
    for m in range(5):
        vals = [status_map[s["status_str"][m]] for s in steps]
        axes[1].step(t, [v + m*0.15 for v in vals], where="post", linewidth=2, label=f"M{m}")
    axes[1].set_ylabel("Status\n(0=OP,1=PM,2=CM,3=F)"); axes[1].legend(fontsize=8, ncol=5); axes[1].grid(alpha=0.3)
    
    # 3. Consumable inventory (all 3)
    for i, color in enumerate(["#2ecc71", "#3498db", "#e74c3c"]):
        axes[2].plot(t, [s[f"cons_{i}_after"] for s in steps], color=color, linewidth=1.5, label=f"Con{i}")
    axes[2].set_ylabel("Consumable\ninventory"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    
    # 4. Renewable availability
    for i, color in enumerate(["#9b59b6", "#f39c12", "#1abc9c"]):
        axes[3].step(t, [s[f"ren_{i}_after"] for s in steps], where="post", color=color, linewidth=2, label=f"Ren{i}")
    axes[3].set_ylabel("Renewable\navailable"); axes[3].legend(fontsize=9); axes[3].grid(alpha=0.3)
    
    # 5. Per-step rewards
    axes[4].plot(t, [s["r1"] for s in steps], color="blue", linewidth=1, alpha=0.7, label="r1")
    axes[4].plot(t, [s["r2"] for s in steps], color="green", linewidth=1, alpha=0.7, label="r2")
    axes[4].set_ylabel("Per-step\nreward"); axes[4].legend(fontsize=9); axes[4].grid(alpha=0.3)
    axes[4].set_xlabel("Step (shift)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "combined_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: combined_timeline.png")


def write_report(steps, env, outdir, config):
    con_names = [c["name"] for c in config["resources"]["consumable"]]
    ren_names = [r["name"] for r in config["resources"]["renewable"]]
    
    lines = []
    lines.append("RESOURCE ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Episode: {env.t_max} steps, seed used")
    lines.append(f"Failures: {env._episode_failures}, PM: {env._episode_pm}, CM: {env._episode_cm}")
    lines.append(f"Jobs completed: {env._episode_completions}")
    lines.append("")
    
    # Consumable summary
    lines.append("CONSUMABLE RESOURCES:")
    for i, name in enumerate(con_names):
        inv_series = [s[f"cons_{i}_after"] for s in steps]
        orders = [s[f"order_{i}"] for s in steps]
        n_orders = sum(1 for o in orders if o > 0)
        total_ordered = sum(orders)
        lines.append(f"  {name}:")
        lines.append(f"    Initial: {steps[0][f'cons_{i}']:.0f}")
        lines.append(f"    Final:   {inv_series[-1]:.0f}")
        lines.append(f"    Min:     {min(inv_series):.0f} (step {np.argmin(inv_series)})")
        lines.append(f"    Max:     {max(inv_series):.0f}")
        lines.append(f"    Mean:    {np.mean(inv_series):.1f}")
        lines.append(f"    Orders:  {n_orders} times, total {total_ordered:.0f} units")
        lines.append(f"    Stockout steps: {sum(1 for v in inv_series if v < 1)}")
    
    lines.append("")
    lines.append("RENEWABLE RESOURCES:")
    for i, name in enumerate(ren_names):
        avail_series = [s[f"ren_{i}_after"] for s in steps]
        cap = steps[0][f"ren_cap_{i}"]
        lines.append(f"  {name} (capacity={cap:.0f}):")
        lines.append(f"    Mean available: {np.mean(avail_series):.1f}")
        lines.append(f"    Min available:  {min(avail_series):.0f} (step {np.argmin(avail_series)})")
        lines.append(f"    Steps at 0:     {sum(1 for v in avail_series if v < 1)}")
        lines.append(f"    Utilization:    {1 - np.mean(avail_series)/cap:.1%}")
    
    lines.append("")
    lines.append("ORDER COST:")
    lines.append(f"  Total ordering cost: {sum(s['ordering_cost'] for s in steps):.0f}")
    lines.append(f"  Total holding cost:  {sum(0.005 * sum(s[f'cons_{i}_after'] for i in range(3)) for s in steps):.1f}")
    
    report = "\n".join(lines)
    with open(os.path.join(outdir, "analysis_report.txt"), "w") as f:
        f.write(report)
    print(f"  Saved: analysis_report.txt")
    print()
    print(report)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--policy", default="smart", choices=["smart", "reactive", "random"])
    pa.add_argument("--outdir", default="outputs/resource_analysis")
    pa.add_argument("--config", default="configs/base.yaml")
    args = pa.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["stochasticity_level"] = 1

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Running episode (seed={args.seed}, policy={args.policy})...")
    steps, env = run_episode_with_tracking(config, seed=args.seed, policy=args.policy)

    # Save raw CSV
    csv_path = os.path.join(args.outdir, "resource_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=steps[0].keys())
        w.writeheader()
        w.writerows(steps)
    print(f"  Saved: resource_summary.csv")

    print("Generating plots...")
    plot_consumables(steps, args.outdir, config)
    plot_renewables(steps, args.outdir, config)
    plot_orders(steps, args.outdir, config)
    plot_combined_timeline(steps, args.outdir)
    write_report(steps, env, args.outdir, config)

    print(f"\nAll outputs in: {args.outdir}/")


if __name__ == "__main__":
    main()
