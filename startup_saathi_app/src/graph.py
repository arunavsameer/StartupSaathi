"""
graph.py — Topological Sort (Kahn's Algorithm) + Checklist Logic
StartupSaathi | Bharat Bricks Hacks 2026

Implements dependency-aware ordering of compliance tasks so that a startup
always sees tasks in the correct sequence (e.g., COI before GST registration).
"""

from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


def topological_sort(tasks: list[dict]) -> list[dict]:
    """
    Sort tasks in dependency order using Kahn's BFS-based algorithm.

    Each task dict must have:
      - task_id  (str)
      - prereq_ids (list[str]) — ids that must be completed first

    Returns the same tasks sorted so that prerequisites always appear before
    the tasks that depend on them. Tasks with no prerequisites come first.

    If there are tasks whose prerequisites are NOT in the filtered task list
    (e.g., a sector-specific task whose prereq is a universal task), those
    prereq dependencies are satisfied implicitly (not blocked).
    """
    if not tasks:
        return []

    task_map = {t["task_id"]: t for t in tasks}
    task_ids_in_set = set(task_map.keys())

    # Build in-degree count and adjacency list (only for edges within the set)
    in_degree = {tid: 0 for tid in task_ids_in_set}
    adj = defaultdict(list)  # adj[prereq] = [task_that_needs_it, ...]

    for task in tasks:
        for prereq in task.get("prereq_ids", []):
            if prereq in task_ids_in_set:
                # This edge exists within our filtered set
                adj[prereq].append(task["task_id"])
                in_degree[task["task_id"]] += 1
            # If prereq NOT in set, we ignore the edge (task is unblocked)

    # Kahn's: start with all nodes whose in-degree is 0
    queue = deque(
        sorted(
            [tid for tid, deg in in_degree.items() if deg == 0],
            key=lambda tid: task_map[tid].get("task_id", "")
        )
    )

    sorted_ids = []
    while queue:
        current = queue.popleft()
        sorted_ids.append(current)
        for neighbor in sorted(adj[current]):  # sorted for determinism
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_ids) != len(tasks):
        # Cycle detected — fall back to original order
        logger.warning("Cycle detected in task graph — returning original order.")
        return tasks

    return [task_map[tid] for tid in sorted_ids]


def get_available_tasks(sorted_tasks: list[dict], completed_task_ids: set[str]) -> set[str]:
    """
    Return the set of task_ids that the user can currently act on.
    A task is 'available' if ALL of its prerequisites appear in completed_task_ids
    OR the prerequisite is not in the current filtered task set.

    Tasks with no prerequisites are always available.
    """
    task_ids_in_set = {t["task_id"] for t in sorted_tasks}
    available = set()

    for task in sorted_tasks:
        prereqs_in_set = [p for p in task.get("prereq_ids", []) if p in task_ids_in_set]
        if all(p in completed_task_ids for p in prereqs_in_set):
            available.add(task["task_id"])

    return available


def group_by_phase(sorted_tasks: list[dict]) -> dict[str, list[dict]]:
    """
    Group sorted tasks into a dict keyed by lifecycle phase.
    Preserves the sorted order within each phase bucket.

    Returns:
        {
            "incorporation":      [task, ...],
            "post-incorporation": [task, ...],
            "operations":         [task, ...],
        }
    All keys are always present (even if empty) so the UI can render them.
    """
    phases = {
        "incorporation":      [],
        "post-incorporation": [],
        "operations":         [],
    }
    for task in sorted_tasks:
        phase = task.get("phase", "operations")
        if phase in phases:
            phases[phase].append(task)
        else:
            phases["operations"].append(task)  # fallback bucket
    return phases


def estimate_days_remaining(tasks: list[dict], completed_task_ids: set[str]) -> int:
    """Return the sum of est_days for all tasks NOT yet completed."""
    return sum(
        t.get("est_days", 0)
        for t in tasks
        if t["task_id"] not in completed_task_ids
    )
