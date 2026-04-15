def near_first_order_resonance(a1,a2,j,tol=0.02):
    """
    Flag proximity to (j+1):j resonance.

    Parameters
    ----------
    tol : fractional tolerance (e.g. 0.01–0.05)

    Notes
    -----
    This is a selection criterion, not a dynamical stability condition.

    It estimates based on an integer j, whether you are within decimal tol
    of j+1/j first-order resonance.

    """
    P_ratio=(a2/a1)**1.5
    target=(j+1)/j
    return abs(P_ratio/target-1)<tol


def classify_with_resonance(mu1,mu2,a1,a2,e1,e2):
    # 1. Orbit crossing → always unstable
    if a1*(1+e1)>=a2*(1-e2):
        return "unstable"

    # 2. Hill spacing
    RH=((mu1+mu2)/3)**(1/3)*(a1+a2)/2
    delta=(a2-a1)/RH

    # 3. Check near resonance (2:1 and 3:2 only for now)
    near_res=(
            near_first_order_resonance(a1,a2,j=1) or
            near_first_order_resonance(a1,a2,j=2)
    )

    if delta<3.5:
        if near_res:
            return "resonant_stable_possible"
        else:
            return "likely_unstable"

    return "stable"