
import data
import for_sets1

# Step to find n^(t) and m^(t)
def find_optimal_pair(w, S_w, S_w_bar, G, previous_t):
    """
    Find the optimal (n^(t), m^(t)) pair for a given w.

    Args:
        w: Current meaning w.
        S_w: Subset S_w for meaning w.
        S_w_bar: Complementary subset S̄_w.
        G: Function to compute G(w, s_n, s_m).
        previous_t: Previously selected subset (t-1).

    Returns:
        Tuple (n_t, m_t) minimizing G(w, s_n, s_m).
    """
    # Initialize variables for the minimum value and corresponding pair
    min_value = float('inf')
    optimal_pair = None

    # Iterate through all combinations of s_n in S_w and s_m in S̄_w
    for s_n in S_w:
        for s_m in S_w_bar:
            # Compute the target function G for the current pair
            current_value = G(w, previous_t, s_n, s_m)
            # Update if a new minimum is found
            if current_value < min_value:
                min_value = current_value
                optimal_pair = (s_n, s_m)

    return optimal_pair



def algorithm_for_encode(W, S, P, Q, loss_fn, distortion_fn, d_fn, max_cost, max_iterations=100):
    """
    Python implementation of Algorithm 2 from the given pseudocode.

    Args:
        W (list): Set of meanings.
        S (list): Set of semantic symbols.
        P (dict): Joint probability distribution over W and S.
        Q (dict): Channel transition probabilities.
        loss_fn (function): Loss function l(s).
        distortion_fn (function): Distortion measure d(w, w_hat).
        d_fn (function): Semantic distance function.
        max_cost (float): Maximum allowed cost Lmax.
        max_iterations (int): Maximum iterations for termination.

    Returns:
        dict: Output results including U^(t), costs, distortions, and T.
    """
    results = {
        'U': [],
        'costs': [],
        'distortions': [],
        'T': None
    }

    # Step 1: Compute subsets S_w_low and S_w_up for each w in W
    subsets_low = {}
    subsets_up = {}
    for w in W:
        subsets_low[w], subsets_up[w] = for_sets1.discover_subsets(w, S , for_sets1.compute_distortion, for_sets1.compute_cost)  # Call Algorithm 1 to get subsets

    # Step 2: Construct a sequence of deterministic source coding schemes U^(t)
    t = 0
    U_t = {}
    costs = []
    distortions = []

    while True:
        # Initialize U^(t)
        U_t[t] = {}

        # Select semantic encoding for each w
        for w in W:
            S_low_w = subsets_low[w] # 下边界子集
            S_up_w = subsets_up[w]
            n_t, m_t = min(((n, m) for n in S_low_w for m in S_up_w),
                           key=lambda nm: G(w, S[nm[0]], S[nm[1]], m=nm[1]))
            U_t[t][w] = (n_t, m_t)

        # Compute total cost and distortion
        total_cost = sum(loss_fn(S[n_t]) for w, (n_t, _) in U_t[t].items())
        total_distortion = sum(distortion_fn(w, U_t[t][w][0]) for w in W)

        costs.append(total_cost)
        distortions.append(total_distortion)

        # Check termination condition
        if total_cost >= max_cost or t >= max_iterations:
            break
        t += 1

    results['U'] = U_t
    results['costs'] = costs
    results['distortions'] = distortions
    results['T'] = t
    return results
