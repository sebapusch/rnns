import numpy as np
import sys

# ========= ASCII console printer (unchanged from your style) =========
def print_digit(arr: np.ndarray, clear: bool = True) -> None:
    """
    Prints a bipolar (-1/+1) array to the console using ASCII blocks,
    optionally clearing and redrawing the frame in place.
    """
    size = int(len(arr) ** 0.5)
    
    if clear:
        # ANSI escape: move cursor to top-left and clear screen
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()

    for row in range(size):
        for col in range(size):
            i = row * size + col
            sys.stdout.write('█' if arr[i] > 0 else ' ')
        sys.stdout.write('\n')
    sys.stdout.flush()


# =================== Geometry helper utilities =======================
def _grid(size: int):
    """Return pixel-centered coordinate grids X,Y in pixels."""
    a = np.arange(size) + 0.5
    return np.meshgrid(a, a)  # Y, X by default, but we’ll use (Y,X) consistently


def _stroke_line(mask_shape, x0, y0, x1, y1, thickness):
    """
    Anti-aliased-ish binary stroke for a line segment using distance-to-segment.
    x*, y* in pixels. thickness in pixels.
    """
    H, W = mask_shape
    Y, X = _grid(H)  # both HxW
    # Vector from A->P and A->B
    APx, APy = X - x0, Y - y0
    ABx, ABy = x1 - x0, y1 - y0
    AB2 = ABx * ABx + ABy * ABy + 1e-9

    # Project P onto AB, clamp to [0,1]
    t = (APx * ABx + APy * ABy) / AB2
    t = np.clip(t, 0.0, 1.0)

    # Closest point on segment
    Cx = x0 + t * ABx
    Cy = y0 + t * ABy

    # Distance from P to closest point
    dist = np.sqrt((X - Cx) ** 2 + (Y - Cy) ** 2)
    return dist <= (thickness / 2)


def _stroke_rect(mask_shape, x0, y0, x1, y1, thickness):
    """Axis-aligned rectangle outline (stroke), coords in pixels."""
    H, W = mask_shape
    Y, X = _grid(H)
    inside_outer = (X >= x0 - thickness/2) & (X <= x1 + thickness/2) & (Y >= y0 - thickness/2) & (Y <= y1 + thickness/2)
    inside_inner = (X > x0 + thickness/2) & (X < x1 - thickness/2) & (Y > y0 + thickness/2) & (Y < y1 - thickness/2)
    return inside_outer & (~inside_inner)


def _stroke_ellipse(mask_shape, cx, cy, rx, ry, thickness, angle_range=None):
    """
    Ellipse outline by band between two ellipses with radii (rx±t/2, ry±t/2).
    angle_range=(ang_min, ang_max) in radians to draw an arc; full if None.
    """
    H, W = mask_shape
    Y, X = _grid(H)
    dx = X - cx
    dy = Y - cy

    # Normalize to ellipse radii
    f_outer = (dx / (rx + thickness/2 + 1e-9)) ** 2 + (dy / (ry + thickness/2 + 1e-9)) ** 2
    f_inner = (dx / max(rx - thickness/2, 1e-6)) ** 2 + (dy / max(ry - thickness/2, 1e-6)) ** 2

    band = (f_outer <= 1.0) & (f_inner >= 1.0)

    if angle_range is not None:
        ang = np.arctan2(dy, dx)  # [-pi, pi]
        a0, a1 = angle_range
        # Handle wrap-around robustly
        if a0 <= a1:
            ang_mask = (ang >= a0) & (ang <= a1)
        else:
            ang_mask = (ang >= a0) | (ang <= a1)
        band &= ang_mask

    return band


def _fill_circle(mask_shape, cx, cy, r):
    """Filled circle (for nicer terminals, dots/serifs)."""
    H, W = mask_shape
    Y, X = _grid(H)
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


def _compose(size, strokes):
    """
    Combine stroke masks into a single -1/1 image.
    strokes: iterable of boolean masks; any True becomes +1, else -1.
    """
    img = -np.ones((size, size), dtype=int)
    for m in strokes:
        img[m] = 1
    return img


# ===================== Pretty digit generator ========================
def make_digits(size=200, thickness=10):
    """
    Returns a list of 10 flattened bipolar (-1/+1) arrays of shape (size*size,)
    representing digits 0..9, drawn with curved/diagonal strokes to be
    visually distinct and less 'square'.
    """
    H = W = size
    t = max(2, int(thickness))

    # Useful anchors (pixels)
    pad = int(0.12 * size)
    cx, cy = W / 2, H / 2
    rx, ry = (W - 2 * pad) / 2, (H - 2 * pad) / 2

    digits = []

    # ---- 0 : Ellipse outline (rounded zero) ----
    m0 = _stroke_ellipse((H, W), cx, cy, rx, ry, t)
    digits.append(_compose(size, [m0]).flatten())

    # ---- 1 : Slightly slanted vertical with base and top serif ----
    x_mid = W * 0.55
    m1_line = _stroke_line((H, W), x_mid - size * 0.05, pad, x_mid, H - pad, t)
    m1_base = _stroke_line((H, W), x_mid - size * 0.15, H - pad, x_mid + size * 0.15, H - pad, t)
    m1_cap  = _stroke_line((H, W), x_mid - size * 0.12, pad, x_mid + size * 0.05, pad + size * 0.05, t)
    digits.append(_compose(size, [m1_line, m1_base, m1_cap]).flatten())

    # ---- 2 : Top arc + descending diagonal + bottom bar ----
    m2_toparc = _stroke_ellipse((H, W), cx, pad + ry * 0.8, rx, ry * 0.9, t, angle_range=(-0.15*np.pi, 1.15*np.pi))
    m2_diag   = _stroke_line((H, W), W - pad, pad + size * 0.35, pad, H - pad * 0.9, t)
    m2_base   = _stroke_line((H, W), pad, H - pad, W - pad, H - pad, t)
    digits.append(_compose(size, [m2_toparc, m2_diag, m2_base]).flatten())

    # ---- 3 : Two right arcs + middle connector ----
    m3_toparc = _stroke_ellipse((H, W), cx, cy - size * 0.22, rx, ry * 0.8, t, angle_range=(-0.25*np.pi, 1.25*np.pi))
    m3_botarc = _stroke_ellipse((H, W), cx, cy + size * 0.22, rx, ry * 0.8, t, angle_range=(-1.25*np.pi, 0.25*np.pi))
    m3_mid    = _stroke_line((H, W), cx - rx * 0.2, cy, cx + rx * 0.6, cy, t)
    digits.append(_compose(size, [m3_toparc, m3_botarc, m3_mid]).flatten())

    # ---- 4 : Diagonal + vertical right + crossbar ----
    m4_diag = _stroke_line((H, W), pad * 0.8, cy - size * 0.35, W - pad, cy, t)
    m4_vert = _stroke_line((H, W), W - pad, pad * 0.8, W - pad, H - pad, t)
    m4_bar  = _stroke_line((H, W), pad, cy, W - pad, cy, t)
    digits.append(_compose(size, [m4_diag, m4_vert, m4_bar]).flatten())

    # ---- 5 : Top bar + left vertical upper + mid bar + bottom bar + right lower vertical (rounded feel) ----
    m5_top = _stroke_line((H, W), pad, pad, W - pad, pad, t)
    m5_lup = _stroke_line((H, W), pad, pad, pad, cy, t)
    m5_mid = _stroke_line((H, W), pad, cy, W - pad * 0.9, cy, t)
    m5_bot = _stroke_line((H, W), pad, H - pad, W - pad, H - pad, t)
    m5_rlo = _stroke_line((H, W), W - pad, cy, W - pad, H - pad, t)
    digits.append(_compose(size, [m5_top, m5_lup, m5_mid, m5_bot, m5_rlo]).flatten())

    # ---- 6 : Outer ellipse + inner connector + open top-right gap ----
    m6_ell = _stroke_ellipse((H, W), cx, cy + size * 0.02, rx, ry, t)
    m6_mid = _stroke_line((H, W), cx - rx * 0.6, cy, cx + rx * 0.2, cy, t)
    # create a gap in the top-right by NOT adding strokes there; add a short inner vertical to emphasize '6'
    m6_inn = _stroke_line((H, W), pad + t, cy - ry * 0.2, pad + t, cy + ry * 0.45, t)
    digits.append(_compose(size, [m6_ell, m6_mid, m6_inn]).flatten())

    # ---- 7 : Top bar + long diagonal to bottom-right ----
    m7_top = _stroke_line((H, W), pad, pad, W - pad, pad, t)
    m7_diag = _stroke_line((H, W), W - pad, pad, pad * 0.9, H - pad, t)
    digits.append(_compose(size, [m7_top, m7_diag]).flatten())

    # ---- 8 : Two stacked ellipses (upper smaller, lower larger) + slim middle bridge ----
    m8_up = _stroke_ellipse((H, W), cx, cy - size * 0.26, rx * 0.85, ry * 0.65, t)
    m8_dn = _stroke_ellipse((H, W), cx, cy + size * 0.28, rx, ry * 0.75, t)
    m8_mid = _stroke_line((H, W), cx - rx * 0.35, cy, cx + rx * 0.35, cy, max(2, t-2))
    digits.append(_compose(size, [m8_up, m8_dn, m8_mid]).flatten())

    # ---- 9 : Ellipse + right upper vertical + slight top bar (leave lower-left open) ----
    m9_ell = _stroke_ellipse((H, W), cx, cy - size * 0.02, rx, ry, t)
    m9_rup = _stroke_line((H, W), W - pad, cy - ry * 0.5, W - pad, cy + ry * 0.2, t)
    m9_top = _stroke_line((H, W), pad + rx * 0.2, pad, W - pad, pad, t)
    digits.append(_compose(size, [m9_ell, m9_rup, m9_top]).flatten())

    # Optional: small round dots to add character where helpful (e.g., serif on '1')
    dot = _fill_circle((H, W), int(W * 0.55), int(H - pad), max(2, t // 2))
    digits[1] = _compose(size, [digits[1].reshape(size, size) == 1, dot]).flatten()

    return digits

def corrupt_pattern(pattern: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """
    Returns a corrupted copy of a bipolar (-1/+1) pattern.
    
    Parameters
    ----------
    pattern : np.ndarray
        Flattened 1D array of -1/+1 values.
    noise_level : float, default=0.1
        Fraction of elements to flip (0.0 = no noise, 1.0 = full inversion).

    Returns
    -------
    np.ndarray
        A new corrupted pattern of the same shape.
    """
    corrupted = pattern.copy()
    n_flip = int(noise_level * pattern.size)

    flip_indices = np.random.choice(pattern.size, n_flip, replace=False)
    corrupted[flip_indices] *= -1
    return corrupted
