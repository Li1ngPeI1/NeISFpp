# depolbrdf.py
""" Diffuse depolarization of AnisoPolarIR (backend-agnostic: numpy / torch).

This module is the P2 increment: it brings paper one's *physically constrained*
diffuse depolarization into the polarimetric forward model, which until now had
none.  Everything here is transcribed from

    "A Physically-Constrained Anisotropic Polarimetric BRDF Model"

equations (11)-(14), page 9 -- the same source the reference implementation
``Self_pBRDF/models/pbrdf_model_v5.py`` implements.

What is ported
--------------
    Eq (11)  P_L(theta_i, theta_o, dphi) = rho(n0,k) * sin^2(th_i) * sin^2(th_o) * cos(2 dphi)
    Eq (12)  M_d = [[1, P_L, 0, 0],
                    [P_L, xi_par, 0, 0],
                    [0, 0, xi_perp, 0],
                    [0, 0, 0, xi_circ]]
    Eq (13)  xi_par  = 1 - W
             xi_perp = (1 - W) * sqrt(1 - P_L^2)
             xi_circ = (1 - 2W) * sqrt(1 - P_L^2)
    Eq (14)  W(sigma) = W_max * (1 - exp(-sigma / sigma_0)),  W_max = 1/3, sigma_0 = 0.3

The roughness argument of ``W``
-------------------------------
Paper one calls the argument ``sigma``.  It is *not* a free convention: both
sides of the project identify it with the GGX slope.  In the reference,
``roughness_params`` sets ``sigma_u = alpha_t = sqrt(2/(nu+2))`` and feeds
``sigma_bar = (sigma_u + sigma_v)/2`` to ``depolarization_W``; here the network's
``roughness`` is likewise the GGX slope (see ``anisobrdf``: "the network's
``roughness`` is the GGX slope ``alpha``").  So the port is a straight
substitution

    sigma_bar  ==  (alpha_x + alpha_y) / 2

with **no scale conversion**.  ``sigma_bar`` below is provided as a named helper
because that identification is the one thing a reader might otherwise get wrong.

Two facts worth stating up front
--------------------------------
1. **The degeneration is ``P_L = 0`` *and* ``sigma = 0`` -> ``M_depol = I``** (paper
   one Table 2).  It is *not* ``W = 0`` alone: at ``W = 0`` the diagonal is
   ``(1, 1, sqrt(1-P_L^2), sqrt(1-P_L^2))``, which still attenuates S2/S3.  A port
   that claims "``W = 0`` reproduces the old behaviour bit-for-bit" is wrong.
2. **This couples DoLP to roughness.**  ``anisobrdf``'s header states that in the
   anisotropic module "DoLP and AoP are *exactly independent* of (alpha_x, alpha_y,
   phi_t); only S0 carries the anisotropy".  ``xi_par`` depends on ``W(sigma_bar)``,
   so that independence ends here, by design: roughness-dependent depolarization
   *is* the paper's contribution.  (See also ``diffuse_two_pass_depol``: given a
   ``(s0, s1, 0, 0)`` incident only ``xi_par`` is observable, so in this pipeline
   the coupling enters through the linear channel alone.)

What upstream's diffuse model actually is
-----------------------------------------
The port has to coexist with ``polarutils.calc_diffuse_stokes_full``, so it matters
what that function *is*.  Its three lines factorise as a rank-1 product, and are
exactly equivalent (verified to 1.1e-16 by ``scripts/check_depol_vs_upstream.py``)
to a two-pass Fresnel chain with an **ideal depolarizer** in the middle::

    S_out = Mt(theta_o) . diag(1, 0) . Mt(theta_i) . S_in

Three consequences, none of them obvious from reading the three lines:

* The rank-1 form is *deliberate*, not a coding slip.  ``diag(1, 0)`` is a realisable
  Mueller matrix, and the same factorisation -- with ``s0 = 1`` -- is what the sibling
  ``calc_diffuse_stokes_mitsuba`` uses, the one whose docstring states it is
  "confirmed to produce the same value as Mitsuba".  It is inherited from the classical
  SfP diffuse model (diffuse DoLP ``= (t_s - t_p)/(t_s + t_p)(theta_o)``, from the paper
  cited at ``calc_diffuse_stokes_no_wi``), where only the *exit* interface polarizes.
* So this port is **not** a bug fix.  It swaps a *presupposed ideal* depolarizer for a
  physical, roughness-dependent one -- ``xi_par = 1 - W`` stays in ``[2/3, 1]`` rather
  than being forced to ``0``.  The two models make *opposite* claims about whether the
  internal linear polarization survives, and they already disagree at ``Mdepol = I``
  (S1 differing by 65% even for unpolarized incidence).  Adopting Eq (9) therefore
  changes the forward model; it cannot be a drop-in.
* Replacing the ideal depolarizer with Eq (9)'s is the cleaner way to state paper two's
  contribution: not "add a depolarization term" but "replace the baseline's implicit
  ideal depolarizer with a physical one".

Why the docstrings of the reference must not be trusted
------------------------------------------------------
The reference ``diffuse_mueller`` docstring still describes an older revision
(``xi_par = 1 - P_L^2``, no ``sqrt``).  That older form is **not** physically
admissible: evaluated with the repository's own Cloude test it gives
``lambda_min = -1/12`` at ``P_L = 0``, ``W = 1/3``, and its Cloude boundary is
``1/sqrt(2) = 0.7071``, whereas the paper's stated boundary is ``0.77``.  The
form ported here -- the paper's Eq (13), which the reference *code* implements --
gives ``lambda_min = 0`` there and a boundary of ``0.7698``, i.e. the paper's
number.  ``scripts/test_depolbrdf.py`` re-derives both as a regression check.

Backends
--------
One implementation, dispatched on the type of the first array argument, reusing
``anisobrdf``'s op shim so the two modules cannot drift apart on dtype/device
handling.
"""

import numpy as np

from mymodules.anisobrdf import ops_of, EPS

try:  # torch is optional: the numpy path must work without it
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _cast(o, value, ref):
    """ Coerce ``value`` to ``ref``'s backend / dtype / device **and shape**.

    ``anisobrdf._like`` only handles dtype/device: a python scalar comes back 0-d,
    which then refuses to concatenate against an ``(n, 1)`` array.  Every formula
    below mixes scalar parameters (``W``, ``W_max``, ``sigma_0``) with arrays, so
    the broadcast is folded in here instead of sprinkled at each call site.
    """
    v = o.asarray(value, dtype=getattr(ref, "dtype", None),
                  device=getattr(ref, "device", None))
    return v * o.ones_like(ref)


def _atleast1d(x):
    """ Promote a 0-d value to shape ``(1,)``.

    Only needed where values are *concatenated*: ``cat_last`` rejects 0-d inputs,
    so a scalar ``PL`` would otherwise blow up in ``diffuse_mueller_flat`` while
    the same call with a 1-element array works.  ``reshape`` exists on both
    numpy arrays and torch tensors, so this stays backend-agnostic.
    """
    if hasattr(x, "ndim") and x.ndim == 0:
        return x.reshape(1)
    return x


# Paper one, Eq (14): the saturation bound is *derived*, not tuned.
W_MAX_DEFAULT = 1.0 / 3.0
SIGMA_0_DEFAULT = 0.3


def _exp(x):
    """ Backend-aware exp (the ``anisobrdf`` shim has no exp; keep it local). """
    if torch is not None and isinstance(x, torch.Tensor):
        return torch.exp(x)
    return np.exp(x)


# =====================================================================
# Eq (14): the depolarization factor
# =====================================================================

def depolarization_W(sigma_bar, W_max=W_MAX_DEFAULT, sigma_0=SIGMA_0_DEFAULT):
    """ Roughness-driven depolarization factor, paper one Eq (14).

        W(sigma) = W_max * (1 - exp(-sigma / sigma_0)),   W_max = 1/3, sigma_0 = 0.3

    Monotone increasing in ``sigma_bar``: smooth surfaces do not depolarize
    (``W -> 0``), rough ones saturate at ``W_max``.  ``sigma_bar`` is the mean GGX
    slope ``(alpha_x + alpha_y)/2``; see the module docstring.

    Args:
        sigma_bar: mean roughness (GGX slope), scalar or array.
        W_max (float): saturation bound, paper default 1/3.
        sigma_0 (float): characteristic scale, paper default 0.3.

    Returns:
        W, broadcast to ``sigma_bar``.
    """
    o = ops_of(sigma_bar)
    s = _cast(o, sigma_bar, sigma_bar)
    return _cast(o, W_max, s) * (1.0 - _exp(-s / _cast(o, sigma_0, s)))


def sigma_bar_from_alpha(alpha_x, alpha_y):
    """ ``(alpha_x + alpha_y)/2`` -- the argument Eq (14) expects.

    A named helper because this is the one identification a reader could get
    wrong (see the module docstring): paper one's ``sigma`` is the GGX slope,
    which is what the network's ``roughness`` already is.
    """
    o = ops_of(alpha_x)
    return 0.5 * (_cast(o, alpha_x, alpha_x) + _cast(o, alpha_y, alpha_x))


# =====================================================================
# Eq (11): the diffuse linear polarization degree
# =====================================================================

def rho_material(n0, k):
    """ Material polarization amplitude ``rho(n0, k)``, paper one Sec. 2.4.

        rho = [(n0^2 + k^2 - 1)^2 + 4 k^2] / [(n0^2 + k^2) * (2 + 2 (n0^2 + k^2))]

    Limits quoted in the paper: ``rho -> 1/2`` in the strongly absorbing metal
    limit, and ``rho ~= 0.107`` for glass at ``n0 = 1.5``.  Both are re-checked in
    the test script.
    """
    o = ops_of(n0)
    n0 = _cast(o, n0, n0)
    k = _cast(o, k, n0)
    s = n0 * n0 + k * k
    rho = ((s - 1.0) ** 2 + 4.0 * k * k) / (s * (2.0 + 2.0 * s))
    # rho is a degree of polarisation, so clip to [0, 1] exactly as the reference
    # does.  Without the clip an unphysical eta (n0 < 1, k ~ 0) gives rho >> 1 and
    # P_L leaves the Cloude-admissible range.  The training path never evaluates P_L
    # (only xi_par survives an (s0, s1, 0, 0) incident), so this is a consistency
    # fix, not a change to any trained path.
    return o.clip(rho, 0.0, 1.0)


def polarization_degree(sin_theta_i, sin_theta_o, cos_2delta_phi, rho):
    """ Diffuse linear polarization degree, paper one Eq (11).

        P_L = rho * sin^2(theta_i) * sin^2(theta_o) * cos(2 dphi)

    Callers pass the factors already reduced to sines and the double-angle cosine
    so this stays backend-agnostic and free of any reference-frame convention --
    the azimuth convention lives in the caller, where it can be checked against
    the rotation already in use.
    """
    o = ops_of(rho)
    sti = _cast(o, sin_theta_i, rho)
    sto = _cast(o, sin_theta_o, rho)
    c2p = _cast(o, cos_2delta_phi, rho)
    return _cast(o, rho, rho) * sti * sti * sto * sto * c2p


# =====================================================================
# Eq (12)-(13): the diffuse Mueller matrix
# =====================================================================

def diffuse_depol_diagonal(PL, W):
    """ The three diagonal depolarization components, paper one Eq (13).

        xi_par  = 1 - W
        xi_perp = (1 - W) * sqrt(1 - P_L^2)
        xi_circ = (1 - 2W) * sqrt(1 - P_L^2)

    These are the *only* place ``W`` enters the diffuse Mueller matrix.

    Note ``xi_circ`` cannot change sign: ``W_max = 1/3`` bounds
    ``xi_circ >= 1/3 > 0``.  (The reference docstring claims otherwise; that
    claim belongs to the older revision -- see the module docstring.)

    Returns:
        (xi_par, xi_perp, xi_circ), each broadcast to ``PL``.
    """
    o = ops_of(PL)
    p = _cast(o, PL, PL)
    w = _cast(o, W, p)
    sqrt_p = o.sqrt(o.maximum(1.0 - p * p, 0.0))
    return 1.0 - w, (1.0 - w) * sqrt_p, (1.0 - 2.0 * w) * sqrt_p


def diffuse_mueller_flat(PL, W):
    """ The diffuse Mueller matrix, paper one Eq (12), returned as the 6 non-zero
    components ``(m00, m01, m10, m11, m22, m33)``.

    Shape ``(..., 6)`` rather than ``(..., 4, 4)`` on purpose: the matrix is
    diagonal besides the ``(0,1)/(1,0)`` pair, all four zero rows/columns are
    exact zeros, and every downstream use here is elementwise.  Materialising a
    ``4x4`` would allocate 16 slots to carry 6 numbers and invite accidental
    4x4-by-3 matmuls against the ``(n, 3)`` Stokes tensors this codebase passes
    around.

    Components:
        m00 = 1,  m01 = m10 = P_L,  m11 = xi_par,  m22 = xi_perp,  m33 = xi_circ
    """
    o = ops_of(PL)
    p = _atleast1d(_cast(o, PL, PL))
    xi_par, xi_perp, xi_circ = diffuse_depol_diagonal(p, W)
    one = o.ones_like(p)
    return o.cat_last([one, p, p, xi_par, xi_perp, xi_circ])


def apply_diffuse_mueller_flat(stokes, mueller_flat):
    """ Apply ``diffuse_mueller_flat`` to a Stokes vector ``(..., 4)`` -> ``(..., 4)``.

    Written out rather than as a matmul: the matrix is sparse, and this keeps the
    operation differentiable and backend-agnostic without a 4x4 allocation.  The
    two cross terms use the usual symmetrised form so a non-symmetric input is
    handled correctly.
    """
    o = ops_of(stokes)
    s = stokes
    m00, m01, m10, m11, m22, m33 = (mueller_flat[..., i:i + 1] for i in range(6))
    s0, s1, s2, s3 = s[..., 0:1], s[..., 1:2], s[..., 2:3], s[..., 3:4]
    out0 = m00 * s0 + m01 * s1
    out1 = m10 * s0 + m11 * s1
    out2 = m22 * s2
    out3 = m33 * s3
    return o.cat_last([out0, out1, out2, out3])


# =====================================================================
# Eq (9): the full two-pass chain  S_out = Mt,out . Mdepol . Mt,in . S_in
# =====================================================================

def fresnel_power_transmittance(cos_theta, ref_id, eps=EPS):
    """ Power transmittances ``(ts, tp)`` of one dielectric interface.

    Transcribed verbatim from ``polarutils.calc_ts_tp`` (which is torch-only) so
    the two-pass chain below can be assembled and *checked against upstream*
    without importing torch.  ``calc_ts_tp`` returns *power* (not amplitude)
    coefficients -- the numerator's ``sqrt(n^2 - sin^2)`` is ``n cos(theta_t)``.
    Parity is asserted in ``scripts/check_depol_vs_upstream.py``; keep the two in
    step if either changes.
    """
    o = ops_of(cos_theta)
    c = _cast(o, cos_theta, cos_theta)
    n2 = _cast(o, float(ref_id) ** 2, c)
    sqr = o.sqrt(n2 - (1.0 - c * c) + eps)
    t_s = 4.0 * c * sqr / ((c + sqr) ** 2 + eps)
    t_p = 4.0 * c * sqr * n2 / ((n2 * c + sqr) ** 2 + eps)
    return t_s, t_p


# ---------------------------------------------------------------------
# Upstream's own arithmetic, wherever torch is in play
# ---------------------------------------------------------------------
# The torch path must use *upstream's* functions rather than a transcription of
# them.  A transcription is only "close": ``calc_ts_tp`` writes ``cos_theta ** 2``
# where the obvious transcription writes ``c * c``, and ``normalize_tensor`` uses
# ``torch.norm`` where one would write ``sqrt(sum(v*v))``.  Those differ in the last
# bits, and the rotation amplifies them through the near-degenerate ``sqrt(1 - cos^2)``
# -- measured at ~7e-15 on the ``xi_par = 0`` comparison, i.e. the two paths agreed to
# 14 digits but not to the last one.  Calling upstream's own helpers makes the
# agreement *exact by construction*, which is the same reason
# ``anisobrdf.specular_dg`` calls ``brdfutils`` for its isotropic branch ("this
# wrapper adds a call, not a computation").
#
# The numpy path keeps the transcription, since it must run without torch;
# ``scripts/check_depol_wiring.py`` group E pins the two paths together, so the
# transcription cannot drift unnoticed.


def _T_from_ts_tp(ts, tp):
    """ ``(t+, t-) = ((ts + tp)/2, (ts - tp)/2)``, upstream's own combination. """
    return (ts + tp) / 2.0, (ts - tp) / 2.0


def _transmittances(cos_theta, ref_id, eps):
    """ ``(ts, tp)`` -- upstream's ``calc_ts_tp`` under torch, else the transcription. """
    if torch is not None and isinstance(cos_theta, torch.Tensor):
        from mymodules import polarutils
        return polarutils.calc_ts_tp(cos_theta=cos_theta, ref_id=ref_id)
    return fresnel_power_transmittance(cos_theta, ref_id, eps)


def _cross_basis(a, b):
    """ Row-wise cross product, in upstream's own call form under torch.

    Upstream calls ``torch.cross(a, b)`` without ``dim``, which torch warns about as
    deprecated.  That warning is deliberately kept rather than fixed: passing
    ``dim=-1`` is the same result here but is *our* choice, and the whole point of
    this function is to be upstream's arithmetic verbatim so the ``xi_par = 0``
    comparison lands on equality instead of 14 digits.  Upstream's own line 622
    emits the identical warning.
    """
    if torch is not None and isinstance(a, torch.Tensor):
        return torch.cross(a, b)
    return np.cross(a, b)


def _normalize_basis(v, eps):
    """ Unit-normalise ``(n, 3)`` rows -- upstream's ``normalize_tensor`` under torch.

    Note upstream divides by ``norm + EPS``, not by ``norm``; that regularisation is
    part of what the ``xi_par = 0`` comparison is comparing against.
    """
    if torch is not None and isinstance(v, torch.Tensor):
        from mymodules.tensorutils import normalize_tensor
        return normalize_tensor(v)
    return v / (_norm_like_upstream(ops_of(v), v) + eps)


def _chain_terms(Tpi, Tmi, Tpo, Tmo, s0, s1, xi_par, eps=EPS):
    """ The six products the two-pass chain is built from, in upstream's own order.

    Splitting them out is what lets both callers preserve upstream's association
    term by term, and *that* is what keeps the ``xi_par = 0`` limit as tight as the
    transmittances themselves.  Upstream's rotated outputs are ``t10*cos + t11*cos``
    and ``-t10*sin - t11*sin``; computing ``(t10 + t11)*cos`` instead is the same
    maths and a different rounding.

    Returns ``(t00, t01, t10, t11, c0, c1)``, with

        S0 = t00 + t01 + c0            S1_surf = t10 + t11 + c1

    and both ``c`` exactly ``0.0`` at ``xi_par = 0`` (they carry the whole
    depolarizing correction).
    """
    o = ops_of(s0)
    s0 = _atleast1d(_cast(o, s0, s0))
    s1 = _atleast1d(_cast(o, s1, s0))

    xi = _atleast1d(_cast(o, xi_par, Tpi))  # broadcast onto the (n, 1) geometry
    inner = Tmi * s0 + Tpi * s1             # S1' of the entrance transmission

    t00 = s0 * Tpo * Tpi
    t01 = s1 * Tpo * Tmi
    t10 = s0 * Tmo * Tpi
    t11 = s1 * Tmo * Tmi
    return t00, t01, t10, t11, Tmo * xi * inner, Tpo * xi * inner


def diffuse_surface_stokes(cos_i, cos_o, s0, s1, xi_par, ref_id, eps=EPS):
    """ Surface-frame ``(S0, S1)`` of ``Mt(theta_o) . diag(1, xi_par) . Mt(theta_i) . S_in``.

    The middle entry is passed in directly (rather than a ``W``) so that the
    ``xi_par = 0`` limit -- what upstream computes, and what no ``W`` can reach,
    since ``xi_par = 1 - W >= 2/3`` -- stays *testable*; see
    ``scripts/check_depol_wiring.py``.  At ``xi_par = 0`` this reproduces upstream's
    three lines term for term.

    Unlike :func:`calc_diffuse_stokes_depol` this uses the numpy transcription
    throughout, so it is a pure-numpy reference and is not the training path.
    """
    tsi, tpi = fresnel_power_transmittance(cos_i, ref_id, eps)
    tso, tpo = fresnel_power_transmittance(cos_o, ref_id, eps)
    Tpi, Tmi = _T_from_ts_tp(tsi, tpi)
    Tpo, Tmo = _T_from_ts_tp(tso, tpo)
    t00, t01, t10, t11, c0, c1 = _chain_terms(Tpi, Tmi, Tpo, Tmo, s0, s1, xi_par, eps)
    return t00 + t01 + c0, t10 + t11 + c1


def diffuse_two_pass_depol(cos_i, cos_o, s0, s1, ref_id, PL, W, eps=EPS):
    """ Paper one Eq (9), evaluated in the **surface frame**.

        S_out = Mt,out . diag(1, xi_par, xi_perp, xi_circ) . Mt,in . S_in

    Incident is taken as ``(s0, s1, 0, 0)`` -- the same assumption upstream makes
    ("the incident Stokes vectors already share the same coordinate frames with
    the surface").  Returns ``(S0_out, S1_out)`` in the surface frame; the caller
    rotates S1 into the camera frame exactly as ``polarutils`` does
    (``S1_cam = S1 cos2phi``, ``S2_cam = -S1 sin2phi``).

    **Consequence worth knowing before wiring:** with a ``(s0, s1, 0, 0)``
    incident the depolarizer acts on ``(S0', xi_par S1', xi_perp*0, xi_circ*0)``,
    so **xi_perp and xi_circ are inert here** -- only ``xi_par = 1 - W`` can reach
    the output.  The roughness-dependent cross/circular depolarization the paper
    derives is therefore not observable through this interface; exercising it
    would need incident S2/S3.

    ``PL`` is accepted for call-site symmetry with :func:`diffuse_mueller_flat` but
    does not reach the output, for the reason just given.
    """
    xi_par, _, _ = diffuse_depol_diagonal(PL, W)
    return diffuse_surface_stokes(cos_i, cos_o, s0, s1, xi_par, ref_id, eps)


# =====================================================================
# trainer-facing: upstream's diffuse Stokes with a *physical* depolarizer
# =====================================================================

# Mirrors ``polarutils.REF_ID``.  Hard-coded rather than imported because
# ``polarutils`` imports torch at module scope while this module must stay
# importable -- and its numpy path testable -- without torch.
# ``scripts/check_depol_wiring.py`` asserts the two agree where torch exists.
REF_ID_DEFAULT = 1.5


def sigma_bar_from_params(brdf_params):
    """ ``(alpha_x + alpha_y)/2`` from the trainer's ``brdf_params`` tensor.

    Dispatches on the last-channel count, mirroring ``anisobrdf.specular_dg``:
    ``(..., 1)`` is upstream's isotropic ``alpha``, ``(..., 3)`` is
    ``(alpha_x, alpha_y, phi_t)``.

    The **arithmetic** mean, deliberately not ``anisobrdf``'s
    ``alpha_iso = sqrt(ax*ay)``: paper one defines ``sigma_bar = (sigma_u +
    sigma_v)/2``.  The two coincide whenever ``alpha_x == alpha_y`` -- including at
    initialisation -- and diverge exactly when anisotropy appears, which is the
    one place a silent substitution would go unnoticed.  Hence a named helper.
    """
    n_ch = brdf_params.shape[-1]
    if n_ch == 1:
        return brdf_params
    if n_ch != 3:
        raise ValueError("brdf params must have 1 or 3 channels, got {}".format(n_ch))
    return 0.5 * (brdf_params[..., 0:1] + brdf_params[..., 1:2])


def _dot_last(o, a, b):
    return o.sum_last(a * b)


def _norm_like_upstream(o, v, eps=EPS):
    """ ``sqrt(sum(v*v))``: the numpy-side stand-in for upstream's ``torch.norm``.

    Only the numpy branch of :func:`_normalize_basis` uses this.  It is the same
    value as ``torch.norm(v, dim=1)`` but not necessarily the same floating-point
    reduction order, which is why the torch branch calls upstream's
    ``normalize_tensor`` instead of this.  ``scripts/check_depol_wiring.py`` group E
    bounds the gap between the two paths.
    """
    return o.sqrt(o.sum_last(v * v))


def _normalize_like_upstream(o, v, eps=EPS):
    """ ``v / (||v|| + eps)``, transcribing ``tensorutils.normalize_tensor``.

    Deliberately *not* exact normalisation: bit-identity at ``xi_par = 0`` requires
    upstream's ``+ eps``.
    """
    return v / (_norm_like_upstream(o, v, eps) + eps)


def _sign_from_cond(o, cond):
    """ ``-1`` where ``cond``, else ``+1`` -- upstream's ``mask * (-2) + 1``. """
    if torch is not None and isinstance(cond, torch.Tensor):
        return torch.where(cond, -1.0, 1.0)
    return np.where(cond, -1.0, 1.0)


def calc_diffuse_stokes_depol(wo_wld, wi_wld, n_wld, cam_axis, s0, s1, brdf_params,
                              ref_id=REF_ID_DEFAULT, eps=EPS, xi_scale=1.0):
    """ Upstream's ``calc_diffuse_stokes_full`` with paper one's physical depolarizer.

    Returns ``(S0, S1, S2)`` in exactly upstream's shapes and frame convention: the
    ``(S1, S2)`` pair is the camera-frame rotation of the *surface-frame* ``S1``.
    The only change from upstream is what that surface-frame ``S1`` is::

        upstream : S1_surf = t-o * S0'
        here     : S1_surf = t-o * S0' + t+o * xi_par * S1',   xi_par = 1 - c * W(bar sigma)

    with ``S0`` picking up the matching ``t-o * xi_par * S1'`` term.  This is the
    identity ``upstream == Mt(theta_o) . diag(1, 0) . Mt(theta_i)`` verified in
    ``scripts/check_depol_vs_upstream.py``, with the middle ``0`` replaced by
    ``xi_par`` -- i.e. the baseline's assumed ideal depolarizer swapped for a
    roughness-dependent one.

    ``xi_scale`` is the coefficient ``c`` in ``xi_par = 1 - c * W``.  Paper one is
    ``c = 1``; it is exposed so the coefficient can be learned or swept.  It may be a
    tensor (a 0-d ``nn.Parameter``) and the gradient flows through it.

    Why ``c`` is only allowed to move *up*: within paper one's family the matrix is a
    valid Mueller matrix (Cloude/PSD) only for ``c >= 1`` at the reachable
    ``|P_L| <= ~0.58``; weaker depolarization (``c < 1``) makes it non-realizable.
    So the paper's ``c = 1`` is the *mildest admissible* depolarization, not an
    arbitrary choice.  The admissible upper end is about ``c = 1.75`` before the
    PSD boundary crosses that ``|P_L|``.  See ``scripts/check_depol_realizability.py``.

    ``brdf_params`` is the trainer's ``(..., 1)`` isotropic ``alpha`` or ``(..., 3)``
    ``(alpha_x, alpha_y, phi_t)``; ``sigma_bar`` is derived by
    :func:`sigma_bar_from_params`.  ``s0``/``s1`` carry the colour axis ``(n, 3)``
    while ``xi_par`` is ``(n, 1)``, so the broadcast is the one upstream already
    relies on.

    Under torch this borrows upstream's own ``calc_ts_tp``, ``normalize_tensor`` and
    ``torch.cross`` (see the ``_transmittances``/``_normalize_basis``/``_cross_basis``
    dispatchers and the note above them), so at ``xi_par = 0`` it agrees with
    ``calc_diffuse_stokes_full`` exactly rather than to 14 digits.  The numpy path
    uses the transcription and is not the training path.

    Reached only when the trainer's ``depol_brdf`` flag is on; with it off the
    trainer calls ``polarutils.calc_diffuse_stokes_full`` unchanged, so the default
    path is untouched by construction.
    """
    o = ops_of(wo_wld)

    cos_i = _dot_last(o, wi_wld, n_wld)
    cos_o = _dot_last(o, wo_wld, n_wld)

    # --- camera-frame rotation, mirroring polarutils.calc_diffuse_stokes_full ---
    current_basis = _normalize_basis(_cross_basis(n_wld, wo_wld), eps)
    target_basis = _normalize_basis(_cross_basis(wo_wld, cam_axis), eps)
    cos_phi = _dot_last(o, current_basis, target_basis)
    forward = _dot_last(o, wo_wld, _cross_basis(current_basis, target_basis))
    flip = _sign_from_cond(o, forward < 0.0)
    sin_phi = o.sqrt(1.0 - cos_phi * cos_phi + eps) * flip
    sin_2phi = 2.0 * sin_phi * cos_phi
    cos_2phi = 1.0 - 2.0 * sin_phi * sin_phi

    # --- the chain, with the depolarizing middle entry ---
    tsi, tpi = _transmittances(cos_i, ref_id, eps)
    tso, tpo = _transmittances(cos_o, ref_id, eps)
    Tpi, Tmi = _T_from_ts_tp(tsi, tpi)
    Tpo, Tmo = _T_from_ts_tp(tso, tpo)
    xi_par = 1.0 - xi_scale * depolarization_W(sigma_bar_from_params(brdf_params))
    t00, t01, t10, t11, c0, c1 = _chain_terms(Tpi, Tmi, Tpo, Tmo, s0, s1, xi_par, eps)
    # Each product is rotated on its own, which is upstream's own association.
    return (t00 + t01 + c0,
            t10 * cos_2phi + t11 * cos_2phi + c1 * cos_2phi,
            -(t10 * sin_2phi) - (t11 * sin_2phi) - (c1 * sin_2phi))
