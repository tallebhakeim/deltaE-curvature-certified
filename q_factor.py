"""
INNOVATION axis 3 -- Q(kappa) from a physical magnetic-loss mechanism.

The paper (Sec 2.2, Eq 4) fits the quality factor with a phenomenological
expression and separate damping constants gamma_NM, gamma_M; it observes that a
stronger frequency response correlates with a LOWER Q ("suggesting a magnetic
loss mechanism") but does not derive it.

Here we tie the loss to the SAME magnetoelastic softening that produces the
Delta-E effect.  Under the resonator's AC stress the magnetization re-rotates;
the lag of that rotation dissipates energy.  The magnetoelastic loss compliance
is proportional to the reversible softening compliance dlambda/dsigma times an
effective magnetic damping alpha_m:

    1/Q_mag = alpha_m * E0 * (dlambda/dsigma)_rms      (dimensionless)
    1/Q     = 1/Q0 + 1/Q_mag

Because dlambda/dsigma is largest for the flat, high-Delta-E devices, Q_mag is
smallest there: Q rises with curvature, exactly the anti-correlation the paper
reports (Fig 6).  This converts their Eq 4 fit into a predictive relation and,
fed back, further suppresses the effective response of flat devices toward the
observed ~2 orders of magnitude spread.
"""
from __future__ import annotations
import numpy as np
from materials import FEGAB_E
from deltaE import magnetic_modulus_profile


def softening_compliance(H_arr, profile, model="squire"):
    """NET reversible Delta-E amplitude over the field sweep, expressed as the
    relative modulus swing (Emax-Emin)/E0 of the through-thickness INTEGRATED
    magnetic modulus.  Using the integrated modulus means the same slice-to-slice
    cancellation that suppresses Delta f_r also suppresses the loss, so Q tracks
    the response inversely."""
    Emag = magnetic_modulus_profile(H_arr, profile, model=model)
    return float((Emag.max() - Emag.min()) / FEGAB_E)


def q_factor(H_arr, profile, Q0=2500.0, alpha_m=0.004, model="squire"):
    """
    Quality factor Q of a device with the given stress profile.
    Q0 is the non-magnetic (mechanical/anchor) limit; alpha_m the magnetic
    damping strength.  Returns (Q, one_over_Qmag).
    """
    s = softening_compliance(H_arr, profile, model=model)
    inv_qmag = alpha_m * s
    Q = 1.0 / (1.0 / Q0 + inv_qmag)
    return Q, inv_qmag


def q_vs_curvature(H_arr, profiles, Q0=2500.0, alpha_m=0.004, model="squire"):
    Q = []; soft = []
    for p in profiles:
        q, _ = q_factor(H_arr, p, Q0=Q0, alpha_m=alpha_m, model=model)
        Q.append(q)
        soft.append(softening_compliance(H_arr, p, model=model))
    return np.array(Q), np.array(soft)
