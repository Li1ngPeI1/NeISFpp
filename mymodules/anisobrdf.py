# anisobrdf.py
""" Anisotropic GGX microfacet forward model (backend-agnostic: numpy / torch).

This module is the P1 increment of AnisoPolarIR: it replaces the isotropic
roughness scalar of ``brdfutils.calc_ggx_reflectance_baek_no_fresnel`` with an
anisotropic roughness pair ``(alpha_x, alpha_y)`` and an explicit tangent frame
whose azimuth ``phi_t`` is a per-pixel (learnable) quantity.

Scope (deliberately narrow)
---------------------------
Only the *scalar* factor

    D * G / (4 * |v.n| * |l.n|)

is (re)implemented here, because that is the only place anisotropy enters.  The
polarization part (Fresnel amplitudes, Stokes rotation into the camera frame) is
left entirely to ``polarutils``: ``calc_specular_stokes_*_general`` uses only
``(wo, wi, cam_axis)`` and the complex refractive index, and the anisotropic
factor is multiplied onto the whole Stokes vector by the caller (see
``trainers/trainers_neisf.py``: ``i_spe_s = fs_s * ndl_s * ILLUM_AREA``).
Consequently DoLP and AoP are *exactly independent* of ``(alpha_x, alpha_y,
phi_t)``; only ``S0`` carries the anisotropy.  This was verified against Mitsuba
independently of this code (gate A, test T1).

Two backends, one implementation
--------------------------------
Both numpy and torch are supported from this single file, dispatched on the type
of the first array argument.  This is deliberate: a second copy of the math would
be free to drift, and the isotropic-degeneration property below is only as good
as the fidelity of the transcription it is checked against.  The numpy path is
what ``scripts/test_anisobrdf.py`` exercises (runs without torch); the torch path
is what the training loop will call.

Pass ``alpha_x`` / ``alpha_y`` / ``phi_t`` as scalars or backend arrays; they are
cast to the dtype and device of the geometry, so python floats work with CUDA
tensors too.

Upstream correspondence and the degeneration baseline
-----------------------------------------------------
Every anisotropic function has a named isotropic counterpart transcribed from
upstream, so the isotropic limit is checked against upstream's own expressions
rather than against a re-derivation:

    ndf_ggx_aniso     -> ndf_ggx_iso        (brdfutils.calc_normal_distribution_walter)
    smith_g1_aniso    -> smith_g1_iso       (brdfutils.calc_lambda_heitz)
    smith_g2_aniso    -> smith_g2_iso       (brdfutils.calc_self_masking)
    reflectance_no_fresnel_aniso -> ..._iso (brdfutils.calc_ggx_reflectance_baek_no_fresnel)

Two upstream facts drive the implementation:

1. ``calc_lambda_heitz`` returns the Smith ``G1`` (not Lambda); hence
   ``calc_self_masking`` returns ``G1(v)*G1(l)``, the *separable* Smith term.
   The anisotropic generalisation stays separable for that reason.
2. ``calc_self_masking`` is called with ``roughness_sq = alpha**2``, i.e. the
   network's ``roughness`` is the GGX slope ``alpha``.  The project maps it as
   ``alpha = sqrt(2 / (nu + 2))``.

At ``alpha_x == alpha_y == alpha`` everything below reduces *identically* to the
isotropic form; the residual against upstream's expression is upstream's own
``EPS`` regularisation, predicted analytically by
``scripts/check_aniso_parity_upstream.py`` (alpha >= 0.3: below 2e-5 relative,
and only in the ``theta_h -> 0`` corner).

Two deliberate departures from upstream (both confined to regularisation, both
making our value the *unregularised* one):

1. the tangent frame and the half vector use exact normalisation, where
   upstream's ``normalize_tensor`` divides by ``norm + EPS``.  For a unit-length
   input that shrink is a relative ``EPS``; it also makes the frame non-orthonormal,
   which breaks the exact swap identity at the 1e-7 level.
2. no additive ``EPS`` in the NFD denominator, unlike upstream.  In the
   ``(h.n, h.t, h.b)`` form the bracket is bounded below by 1 for ``alpha <= 1``,
   so the denominator is ``>= pi ax ay > 0`` on its own.

The reference frame for the anisotropy axis comes from the branchless
orthonormal basis of Duff et al. (2017).  It is closed form, and deliberately
*not* built by picking the world axis least aligned with the normal, for two
reasons.  First, that needs an ``argmin``, whose tie-breaking is not guaranteed
to agree between numpy and CUDA, and exact ties in ``(|nx|, |ny|, |nz|)`` are
common on real surfaces rather than measure-zero; a frame that differed between
the reference implementation and the training path would silently change what
``phi_t`` means.  Second, ``argmin`` makes the frame a piecewise-constant
function of the normal, so the gradient of the frame w.r.t. ``n`` is zero almost
everywhere -- the ONB has a well-defined derivative w.r.t. both the normal and
``phi_t``, which matters because all three of ``alpha_x``, ``alpha_y`` and
``phi_t`` are learned.

Verification ladder (see also ``scripts/check_aniso_parity_upstream.py``)
-----------------------------------------------------------------------
1. ``scripts/test_anisobrdf.py``              49 checks, numpy, no torch needed
2. ``scripts/check_aniso_parity_upstream.py`` 21 checks, numpy vs upstream torch
3. ``scripts/check_torch_backends.py``        31 checks, float64 / float32 / autograd
4. ``scripts/check_aniso_wiring.py``          the two trainer-facing functions below,
                                              against upstream, no data needed.  This
                                              is what makes the *training path*
                                              covered rather than the module alone.

Re-running gate A (``scripts/test_mitsuba_convention.py``) after the wiring is a
no-op: it imports only ``mitsuba``, ``numpy`` and upstream ``polarutils``, so it
never exercises this file.  What the wiring *does* make possible is a genuinely
new test -- our anisotropic ``S0`` against Mitsuba with ``alpha_x != alpha_y``
over a range of ``phi_t``.  Gate A's T1 established an invariance property of
Mitsuba itself; it never compared an anisotropic ``S0`` to anything.
"""

import numpy as np

try:  # torch is optional: the numpy path must work without it
    import torch
except ImportError:  # pragma: no cover
    torch = None

EPS = 1e-07

# Plain python float so that it never promotes a numpy array or a torch tensor
# (numpy NEP 50 and torch scalar rules both keep the operand's dtype).
_PI = 3.141592653589793

# np.finfo(np.float32).tiny as a plain python float, same reasoning.
_TINY = 1.1754944e-38


# =====================================================================
# backend shim
# =====================================================================

class _NumpyOps:
    name = "numpy"

    def asarray(self, v, dtype=None, device=None):
        return np.asarray(v, dtype=dtype)

    def ones_like(self, x):
        return np.ones_like(x)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def sum_last(self, x):
        return np.sum(x, axis=-1, keepdims=True)

    def norm_last(self, x):
        return np.linalg.norm(x, axis=-1, keepdims=True)

    def cross_last(self, a, b):
        """ Row-wise cross product of ``(..., 3)`` pairs; needed by depolbrdf's
        transcription of upstream's camera-frame rotation. """
        return np.cross(a, b)

    def cat_last(self, seq):
        """ Join (..., 1) components along the last axis into (..., 3).

        Note this is *concatenate*, not stack: the components are already
        (..., 1) slices, so stacking would insert an extra axis and produce
        (..., 1, 3), which then broadcasts catastrophically against (..., 3).
        """
        return np.concatenate(seq, axis=-1)

    def sqrt(self, x):
        return np.sqrt(x)

    def cos(self, x):
        return np.cos(x)

    def sin(self, x):
        return np.sin(x)

    def abs(self, x):
        return np.abs(x)

    def sign(self, x):
        return np.sign(x)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def copysign(self, a, b):
        return np.copysign(a, b)


class _TorchOps:
    name = "torch"

    def asarray(self, v, dtype=None, device=None):
        return torch.as_tensor(v, dtype=dtype, device=device)

    def ones_like(self, x):
        return torch.ones_like(x)

    def zeros_like(self, x):
        return torch.zeros_like(x)

    def sum_last(self, x):
        return torch.sum(x, dim=-1, keepdim=True)

    def norm_last(self, x):
        return torch.linalg.norm(x, dim=-1, keepdim=True)

    def cross_last(self, a, b):
        """ Row-wise cross product of ``(..., 3)`` pairs; see the numpy version. """
        return torch.cross(a, b, dim=-1)

    def cat_last(self, seq):
        """ Join (..., 1) components along the last axis; see the numpy version. """
        return torch.cat(seq, dim=-1)

    def sqrt(self, x):
        return torch.sqrt(x)

    def cos(self, x):
        return torch.cos(x)

    def sin(self, x):
        return torch.sin(x)

    def abs(self, x):
        return torch.abs(x)

    def sign(self, x):
        return torch.sign(x)

    def clip(self, x, lo, hi):
        return torch.clip(x, lo, hi)

    def _tensorise(self, x, ref):
        """ Coerce a plain scalar to a tensor matching ``ref``'s dtype and device.

        Needed because several ``torch`` binary ops are *stricter* than their
        numpy counterparts: ``torch.maximum`` insists on two tensors where
        ``np.maximum`` happily takes a python float.  Applying this at the shim
        boundary keeps that asymmetry from leaking into the formulas above.
        """
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)

    def maximum(self, a, b):
        """ Element-wise max (numpy signature: accepts scalars on either side). """
        if isinstance(a, torch.Tensor):
            b = self._tensorise(b, a)
        else:
            a = self._tensorise(a, b)
        return torch.maximum(a, b)

    def copysign(self, a, b):
        """ Element-wise copysign (numpy signature: accepts scalars on either side). """
        if isinstance(a, torch.Tensor):
            b = self._tensorise(b, a)
        else:
            a = self._tensorise(a, b)
        return torch.copysign(a, b)


_NP_OPS = _NumpyOps()
_TORCH_OPS = _TorchOps() if torch is not None else None


def ops_of(x):
    """ Backend dispatcher: the ops namespace matching ``x``. """
    if _TORCH_OPS is not None and isinstance(x, torch.Tensor):
        return _TORCH_OPS
    return _NP_OPS


def _like(o, value, ref):
    """ Cast ``value`` to the backend / dtype / device of ``ref``.

    Needed so that python floats (or numpy scalars) can be combined with CUDA
    tensors without a device or dtype promotion surprise.  ``ref`` may itself be
    a python scalar, in which case no dtype is imposed.
    """
    return o.asarray(value, dtype=getattr(ref, "dtype", None),
                     device=getattr(ref, "device", None))


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------

def _dot(a, b):
    """ Dot product over the last axis, returned as (..., 1) like upstream. """
    return ops_of(a).sum_last(a * b)


def _cross(a, b):
    """ Cross product over the last axis (numpy's own ``cross`` is being retired). """
    o = ops_of(a)
    ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3]
    bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3]
    return o.cat_last([ay * bz - az * by,
                       az * bx - ax * bz,
                       ax * by - ay * bx])


def _normalize(v):
    """ Mirror of ``tensorutils.normalize_tensor``: ``v / (norm + EPS)``.

    Used only where upstream's own math is reproduced, so that the isotropic
    limit stays comparable to upstream expression for expression.  Do *not* use
    it for the tangent frame: for a unit-length input it shrinks the vector by a
    relative ``EPS`` and breaks the exact orthonormality the frame identities
    rely on.
    """
    o = ops_of(v)
    return v / (o.norm_last(v) + EPS)


def _normalize_exact(v):
    """ Exact normalisation (no ``+ EPS``), for vectors of known unit length.

    The tangent frame is built from unit vectors, so a plain division keeps it
    exactly orthonormal, which the swap-symmetry identity needs.  Dividing by
    ``norm + EPS`` instead would introduce a spurious ~1e-7 scaling of the
    anisotropy axis, and ``phi_t`` is meant to be a learned quantity.
    """
    o = ops_of(v)
    return v / o.maximum(o.norm_last(v), _TINY)


def _chi(x):
    """ Mirror of ``brdfutils.calc_chi``: 1 where x > 0, else 0. """
    o = ops_of(x)
    return o.clip(o.sign(x), 0.0, 1.0)


# ---------------------------------------------------------------------
# roughness mapping (project convention: nu <-> alpha)
# ---------------------------------------------------------------------

def alpha_from_nu(nu):
    """ GGX slope from the project's ``nu`` convention: alpha = sqrt(2/(nu+2)).

    Same mapping as the Mitsuba validation scripts, so rendered alpha_u/alpha_v
    and the model's (nu, nv) are interchangeable.  Works on scalars, numpy arrays
    and torch tensors alike (the arithmetic stays in the operand's own backend
    and dtype).
    """
    return ops_of(nu).sqrt(2.0 / (nu + 2.0))


def nu_from_alpha(alpha):
    """ Inverse of ``alpha_from_nu``. """
    return 2.0 / (alpha * alpha) - 2.0


# ---------------------------------------------------------------------
# tangent frame
# ---------------------------------------------------------------------

def tangent_frame(normal, phi_t=0.0):
    """ Orthonormal tangent frame ``(t, b)`` about ``normal``, azimuthed by ``phi_t``.

    ``t`` is the anisotropy axis (the ``alpha_x`` direction).  The reference
    frame is the branchless orthonormal basis of Duff et al. (2017), then rotated
    about ``normal`` by ``phi_t``.  Handedness is right-handed:
    ``cross(t, b) == normal``.

    Args:
        normal: surface normals, shape (..., 3).
        phi_t: azimuth of the anisotropy axis, radians (scalar or array).

    Returns:
        (t, b) each of shape (..., 3).
    """
    o = ops_of(normal)
    n = _normalize_exact(normal)

    nx, ny, nz = n[..., 0:1], n[..., 1:2], n[..., 2:3]
    s = o.copysign(o.ones_like(nz), nz)
    a = -1.0 / (s + nz)          # |s + nz| >= 1, so this is always finite
    bb = nx * ny * a

    t0 = _normalize_exact(o.cat_last([
        1.0 + s * nx * nx * a,
        s * bb,
        -s * nx,
    ]))
    b0 = o.cat_last([bb, s + ny * ny * a, -ny])

    phi = _like(o, phi_t, t0)
    t = o.cos(phi) * t0 + o.sin(phi) * b0
    return t, _cross(n, t)


# ---------------------------------------------------------------------
# NDF
# ---------------------------------------------------------------------

def ndf_ggx_aniso(half_vec, normal, t, b, alpha_x, alpha_y):
    """ Anisotropic GGX normal distribution, shape (..., 1).

    ``D = chi(h.n) / (pi ax ay (h.n^2 + h.t^2/ax^2 + h.b^2/ay^2)^2)``

    No additive ``+ EPS`` in the denominator, unlike upstream: in this
    ``h``-frame form the bracket is bounded below by
    ``h.n^2 + h.t^2 + h.b^2 = 1`` for ``alpha <= 1``, so the denominator is
    ``>= pi ax ay > 0`` on its own.  Upstream needs the guard because its
    denominator ``pi cos^4(th)(alpha^2 + tan^2(th))^2`` vanishes as
    ``alpha -> 0`` at ``theta_h -> 0``; that guard biases upstream's ``D`` there,
    which is the entire origin of the residual in the degeneration checks.
    """
    o = ops_of(half_vec)
    ax = _like(o, alpha_x, half_vec)
    ay = _like(o, alpha_y, half_vec)

    h_n = _dot(half_vec, normal)
    h_t = _dot(half_vec, t)
    h_b = _dot(half_vec, b)

    inner = h_n ** 2 + h_t ** 2 / (ax * ax) + h_b ** 2 / (ay * ay)
    return _chi(h_n) / (_PI * ax * ay * inner ** 2)


def ndf_ggx_iso(n_dot_h, roughness_sq):
    """ Upstream's isotropic GGX NDF, transcribed from ``calc_normal_distribution_walter``.

    Kept verbatim (including ``tan2_th = (1-cos2)/(cos2+EPS)`` and the trailing
    ``+ EPS``) so the anisotropic form is measured against upstream's own
    expression rather than against a re-derivation.
    """
    o = ops_of(n_dot_h)
    cos2_th = n_dot_h * n_dot_h
    tan2_th = (1.0 - cos2_th) / (cos2_th + EPS)
    cos4_th = cos2_th * cos2_th
    return roughness_sq * _chi(n_dot_h) / (
        _PI * cos4_th * (roughness_sq + tan2_th) ** 2 + EPS)


# ---------------------------------------------------------------------
# Smith masking
# ---------------------------------------------------------------------

def smith_g1_aniso(v, normal, t, b, alpha_x, alpha_y):
    """ Anisotropic Smith ``G1`` (Heitz 2014), shape (..., 1).

    ``G1(v) = 2 chi(v.n) / (1 + sqrt(1 + (ax^2 v.t^2 + ay^2 v.b^2) / (v.n^2 + EPS)))``

    The scalar under the root is written as ``slope / (v.n^2 + EPS)`` rather than
    via ``tan^2 * (ax^2 cos^2 phi + ay^2 sin^2 phi)``; the two are the same
    expression and the former avoids a ``1 - v.n^2`` division.
    """
    o = ops_of(v)
    ax = _like(o, alpha_x, v)
    ay = _like(o, alpha_y, v)

    v_n = _dot(v, normal)
    v_t = _dot(v, t)
    v_b = _dot(v, b)

    slope = ax * ax * v_t ** 2 + ay * ay * v_b ** 2
    root = o.sqrt(1.0 + slope / (v_n ** 2 + EPS))
    return 2.0 * _chi(v_n) / (1.0 + root)


def smith_g1_iso(x_dot_n, roughness_sq):
    """ Upstream's Smith ``G1``, transcribed from ``calc_lambda_heitz``.

    Note the upstream function is *named* lambda but returns ``1/(1+Lambda)``,
    i.e. ``G1``.
    """
    o = ops_of(x_dot_n)
    cos2_th = x_dot_n * x_dot_n
    tan2_th = (1.0 - cos2_th) / (cos2_th + EPS)
    return 2.0 * _chi(x_dot_n) / (1.0 + o.sqrt(1.0 + roughness_sq * tan2_th))


def smith_g2_aniso(wo, wi, normal, t, b, alpha_x, alpha_y):
    """ Separable Smith ``G2 = G1(wo) * G1(wi)``, matching upstream's ``calc_self_masking``.

    Upstream uses the separable form (not height-correlated), so the anisotropic
    generalisation stays separable to keep the isotropic limit exact.
    """
    return (smith_g1_aniso(wo, normal, t, b, alpha_x, alpha_y)
            * smith_g1_aniso(wi, normal, t, b, alpha_x, alpha_y))


def smith_g2_iso(v_dot_n, l_dot_n, roughness_sq):
    """ Upstream's separable Smith ``G2``, transcribed from ``calc_self_masking``. """
    return (smith_g1_iso(v_dot_n, roughness_sq)
            * smith_g1_iso(l_dot_n, roughness_sq))


# ---------------------------------------------------------------------
# the anisotropic factor that multiplies the Stokes vector
# ---------------------------------------------------------------------

def reflectance_no_fresnel_aniso(wo, wi, normal, alpha_x, alpha_y, phi_t=0.0):
    """ ``D * G / (4 |wo.n| |wi.n|)`` for anisotropic GGX, shape (..., 1).

    Same role as upstream's ``calc_ggx_reflectance_baek_no_fresnel`` (no Fresnel:
    the Fresnel term lives in the Stokes calculation).  Multiply the returned
    scalar onto the *whole* Stokes vector returned by
    ``polarutils.calc_specular_stokes_*_general``.

    Args:
        wo: view directions, shape (..., 3).  Point away from the surface.
        wi: light directions, shape (..., 3).  Point away from the surface.
        normal: surface normals, shape (..., 3).
        alpha_x, alpha_y: GGX slopes along t and b (scalar or array).
        phi_t: azimuth of the anisotropy axis about ``normal`` (scalar or array).

    Returns:
        (..., 1) in the same backend and dtype as ``wo``.
    """
    o = ops_of(wo)
    t, b = tangent_frame(normal, phi_t)
    half_vec = _normalize_exact(wo + wi)

    d_term = ndf_ggx_aniso(half_vec, normal, t, b, alpha_x, alpha_y)
    g_term = smith_g2_aniso(wo, wi, normal, t, b, alpha_x, alpha_y)

    v_n = _dot(wo, normal)
    l_n = _dot(wi, normal)
    return d_term * g_term / (4.0 * o.abs(v_n) * o.abs(l_n) + EPS)


def reflectance_no_fresnel_iso(wo, wi, normal, roughness):
    """ Upstream's ``calc_ggx_reflectance_baek_no_fresnel``, transcribed verbatim.

    Kept expression for expression against upstream (including ``_normalize``'s
    ``+ EPS`` and the additive ``EPS`` in the NDF denominator).  It is the
    *parity target*, not our implementation: comparing it against
    ``reflectance_no_fresnel_aniso`` therefore measures upstream's
    regularisation bias as much as it measures agreement.  Use it as the
    reference only together with the predicted-bias check in
    ``scripts/check_aniso_parity_upstream.py``.
    """
    o = ops_of(wo)
    roughness_sq = roughness ** 2

    v_n = _dot(wo, normal)
    l_n = _dot(wi, normal)
    half_vec = _normalize(wo + wi)
    n_dot_h = _dot(normal, half_vec)

    d_term = ndf_ggx_iso(n_dot_h, roughness_sq)
    g_term = smith_g2_iso(v_n, l_n, roughness_sq)

    return d_term * g_term / (4.0 * o.abs(v_n) * o.abs(l_n) + EPS)


# ---------------------------------------------------------------------
# trainer-facing API
# ---------------------------------------------------------------------
# The two functions below are the *entire* surface that
# ``trainers/trainers_neisf.py`` uses.  Keeping the parameterisation here rather
# than in the trainer has two reasons: it is the same in all four trainers, and
# it can be tested in the sandbox on the numpy path.

def brdf_params_from_raw(raw, is_aniso):
    """ Map the raw ``rough-net`` output to the BRDF parameters.

    Returns ``(alpha_iso, params)``.

    * ``alpha_iso``: shape (..., 1).  An isotropic-equivalent roughness, because
      the consumers that predate the anisotropy increment -- UV texture export,
      ``evaluate``/``compute_metrics`` -- are 1-channel by construction.  With
      ``is_aniso=False`` it *is* the model's roughness.  With ``is_aniso=True``
      it is the geometric mean ``sqrt(ax*ay)``, which is the ``alpha`` that
      reproduces both ``D(h=n) = 1/(pi ax ay)`` and the hemispherical integral of
      the anisotropic NDF.  Stated explicitly because the alternative --
      letting a 3-channel tensor flow into ``reshape_and_scale_all_images`` --
      would be reshaped to (h, w, 3) and silently written out as if it were a
      picture.
    * ``params``: shape (..., 1) for the isotropic case (upstream's own tensor,
      unchanged) or (..., 3) ``(alpha_x, alpha_y, phi_t)``.  This is the argument
      to :func:`specular_dg`.

    Conventions (``raw`` is ``nn.Tanh``-squashed by ``BRDFNet`` and the last
    layer is zero-initialised, so ``raw`` lies in ``(-1, 1)`` and is exactly 0 at
    initialisation):

    * ``alpha = raw * 0.5 + 0.5``, in ``(0, 1)`` -- upstream's own line, verbatim.
      Hence ``alpha_x == alpha_y == 0.5`` at initialisation, i.e. the anisotropic
      model *starts* isotropic, which is what makes the flag-on run comparable to
      the baseline at epoch 1.
    * ``phi_t = raw * pi``, strictly inside ``(-pi, pi)``.

    A single angle channel, not a ``(cos, sin)`` pair.  An earlier note
    recommended the pair to avoid the discontinuity at ``+/- pi``; that concern
    does not apply here, because with a ``tanh`` head the reachable interval is
    *open*, so the parameterisation is continuous, ``phi_t == 0`` at
    initialisation, and the gradient there is non-zero.  A ``(cos, sin)`` pair
    would on the contrary be degenerate exactly at ``(0, 0)``, which is the
    initialisation, leaving ``phi_t`` undefined and its gradient at the origin.
    """
    o = ops_of(raw)
    if not is_aniso:
        alpha = raw * 0.5 + 0.5
        return alpha, alpha

    alpha_x = raw[..., 0:1] * 0.5 + 0.5
    alpha_y = raw[..., 1:2] * 0.5 + 0.5
    phi_t = raw[..., 2:3] * _PI
    alpha_iso = o.sqrt(alpha_x * alpha_y)
    return alpha_iso, o.cat_last([alpha_x, alpha_y, phi_t])


def specular_dg(view_dir, light_dir, normal, params):
    """ ``D * G / (4 |v.n| |l.n|)``, dispatching on the anisotropy.

    ``params`` is (..., 1) isotropic ``alpha`` or (..., 3) ``(alpha_x, alpha_y,
    phi_t)``; see :func:`brdf_params_from_raw`.

    The 1-channel branch calls upstream's
    ``brdfutils.calc_ggx_reflectance_baek_no_fresnel`` unchanged, so with the
    anisotropy flag off the training path is arithmetically identical to
    upstream's: this wrapper adds a call, not a computation.

    ``brdfutils`` is imported lazily because it imports torch, while this module
    must stay importable -- and its numpy path testable -- without torch.
    """
    n_ch = params.shape[-1]
    if n_ch == 1:
        from mymodules import brdfutils
        return brdfutils.calc_ggx_reflectance_baek_no_fresnel(
            view_dir=view_dir, light_dir=light_dir, normal=normal,
            roughness=params)
    if n_ch != 3:
        raise ValueError(
            "brdf params must have 1 or 3 channels, got {}".format(n_ch))
    return reflectance_no_fresnel_aniso(
        view_dir, light_dir, normal,
        params[..., 0:1], params[..., 1:2], params[..., 2:3])
