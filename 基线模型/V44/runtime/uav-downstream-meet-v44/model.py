"""V44 platform entry. Create one policy per mission, then call agent.act(observation)."""
from experiment44 import new_agent
import numpy as np


def load_model(people=64, window_seconds=720, *, edge_search=None, fleet_policy=None):
    if not 1 <= people <= 80 or not 0 < window_seconds < 2400:
        raise ValueError("Invalid V44 population size or personal window")
    if fleet_policy == "terminal_dispatch":
        if edge_search is not None or (people, window_seconds) != (64, 720):
            raise ValueError("Terminal dispatch requires 64 people and a 720-second window")
        from study_terminal44 import sources, unpack, build_terminal
        from experiment44 import HERE, read, sha
        fitted = read(HERE/"terminal-fit.json")
        selected = read(HERE/"terminal-select.json")
        if fitted["source"] != sources() or selected["fit"] != sha(HERE/"terminal-fit.json"):
            raise ValueError("Terminal-dispatch weights and selection must match training")
        return build_terminal(unpack(fitted["variants"][selected["winner"]]))
    if fleet_policy == "early_delivery":
        if edge_search is not None or (people, window_seconds) != (64, 720):
            raise ValueError("Early delivery was trained for 64 people and a 720-second window")
        from study_speed44 import configured, hashes
        from experiment44 import HERE, read, sha
        selected = read(HERE/"speed-select.json")
        weights = read(HERE/"speed-fit.json")
        if selected["fit"] != sha(HERE/"speed-fit.json") or weights["code"] != hashes():
            raise ValueError("Early-delivery selection and weights must match their training run")
        return configured(weights["variants"][selected["winner"]])
    agent = new_agent()
    if fleet_policy is not None:
        from experiment44 import HERE, read
        from fleet_options44 import Fleet, NAMES
        if edge_search is not None or fleet_policy not in (*NAMES, "rule", "learned"):
            raise ValueError("Choose a single supported fleet policy")
        weights = None
        if fleet_policy == "learned":
            data = read(HERE/"fleet-options-fit.json")
            weights = {k: np.asarray(data[k]) for k in ("mean","scale","coef","intercept")}
        agent.deep = Fleet(fleet_policy, weights)
    if edge_search is not None:
        if edge_search == "occupancy":
            from experiment44 import HERE, read
            from occupancy44 import Occupancy
            agent.deep = Occupancy(read(HERE/"occupancy-fit.json")["weights"])
        elif edge_search == "detour_gate":
            from detour_gate44 import DetourGate
            agent.deep = DetourGate()
        elif edge_search in ("all", "one"):
            from edge_search44 import EdgeSearch
            agent.deep = EdgeSearch(edge_search)
        else:
            raise ValueError("Unknown edge search policy")
    agent.deep.known44 = np.zeros(people, dtype=bool)
    agent.deep.belief.personal_window_seconds = window_seconds
    agent.sparse.cohorts.personal_window_seconds = window_seconds
    return agent
