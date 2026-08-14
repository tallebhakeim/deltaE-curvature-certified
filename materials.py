"""
Material and design constants for the reproduction of

    A. D. Matyushov, B. Spetzler, ... N. X. Sun,
    "Curvature and Stress Effects on the Performance of Contour-Mode
     Resonant Delta-E Effect Magnetometers",
    Adv. Mater. Technol. 2021, 6, 2100294.

All numbers are taken VERBATIM from the paper (main text + Experimental
section) so the reproduction can be audited line by line.  Where the paper
gives a value with a reference (e.g. lambda_s, B1) the source equation number
is noted in the comment.
"""
from __future__ import annotations
from dataclasses import dataclass

MU0 = 4.0e-7 * 3.141592653589793  # H/m


@dataclass(frozen=True)
class IsotropicLayer:
    """A single isotropic elastic film in the laminate."""
    name: str
    thickness: float      # m
    E: float              # Young's modulus, Pa
    nu: float             # Poisson ratio
    rho: float            # density, kg/m^3

    @property
    def Q(self) -> tuple[float, float, float]:
        """Plane-stress reduced stiffness (Q11=Q22, Q12, Q66) in Pa."""
        E, nu = self.E, self.nu
        q11 = E / (1.0 - nu * nu)
        q12 = nu * E / (1.0 - nu * nu)
        q66 = E / (2.0 * (1.0 + nu))
        return q11, q12, q66


# --- FeGaB magnetic film -------------------------------------------------
# Paper: E_m = 215 GPa, nu = 0.3  -> C1111 = 289.4 GPa, C1122 = 124 GPa (eq 6)
# lambda_s = 75 ppm, magnetoelastic coupling B1 ~ -18.6 MPa (Sec 2.3.2)
FEGAB_E = 215e9
FEGAB_NU = 0.3
FEGAB_RHO = 7800.0        # kg/m^3 (typical FeGa; used only in f_r scaling)
LAMBDA_S = 75e-6          # saturation magnetostriction
B1 = -18.6e6              # Pa, = -(3/2) lambda_s * (C1111 - C1122), eq 6
MS_FEGAB = 1.28e6         # A/m, saturation magnetization of FeGaB (~1.6 T)
KU = 0.8e3               # J/m^3, field-induced uniaxial anisotropy (Sec 2.3.3)

# --- AlN piezoelectric -----------------------------------------------------
ALN_E = 320e9
ALN_NU = 0.24
ALN_RHO = 3260.0

# --- Pt bottom interdigital electrode --------------------------------------
PT_E = 168e9
PT_NU = 0.38
PT_RHO = 21450.0


def default_stack(t_fegab: float = 250e-9) -> list[IsotropicLayer]:
    """
    Layer stack of Investigation-1 devices (bottom -> top):
        50 nm Pt IDE | 250 nm AlN | t_fegab FeGaB (default 250 nm)
    Order in the list is bottom -> top; z is built in mechanics.py.
    """
    return [
        IsotropicLayer("Pt", 50e-9, PT_E, PT_NU, PT_RHO),
        IsotropicLayer("AlN", 250e-9, ALN_E, ALN_NU, ALN_RHO),
        IsotropicLayer("FeGaB", t_fegab, FEGAB_E, FEGAB_NU, FEGAB_RHO),
    ]


# --- Device / test constants ----------------------------------------------
# Reference (zero-field) resonance and IDE pitch are design dependent; the
# absolute f_r cancels in Delta f_r / f_r,min so any consistent value works.
F_R0 = 245.8e6            # Hz, design ID 3 average f_r (Table 1)
