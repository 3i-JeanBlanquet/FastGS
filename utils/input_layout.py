import os

# COLMAP model may sit in the standard 3DGS location or bare at the input root.
_MODEL_CANDIDATES = [os.path.join("sparse", "0"), "sparse", "0", ""]


def find_model_dir(source_path):
    """Return the directory holding cameras/images/points3D (.bin or .txt)."""
    for rel in _MODEL_CANDIDATES:
        d = os.path.join(source_path, rel) if rel else source_path
        if os.path.exists(os.path.join(d, "cameras.bin")) or \
           os.path.exists(os.path.join(d, "cameras.txt")):
            return d
    raise FileNotFoundError(
        "No COLMAP model under {}; tried {}".format(source_path, _MODEL_CANDIDATES))


def resolve_image_path(source_path, images_arg, colmap_name):
    """Resolve a COLMAP image name to a path on disk.

    colmap_name may carry a rig subdirectory ("rig-1,0,0,0/<uuid>.jpg"), which
    must be preserved -- the four cube faces of a station share a basename and
    would otherwise collapse onto one file.
    """
    reading_dir = images_arg if images_arg else "images"
    candidates = [
        os.path.join(source_path, colmap_name),                  # rig dirs at root
        os.path.join(source_path, reading_dir, colmap_name),     # rig dirs under images/
        os.path.join(source_path, reading_dir, os.path.basename(colmap_name)),  # flat
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Image {} not found; tried:\n  {}".format(colmap_name, "\n  ".join(candidates)))


def resolve_depth_path(image_path):
    """Return the depth PNG for an image, or None if there is not one."""
    stem, _ = os.path.splitext(image_path)
    d = os.path.dirname(image_path)
    base = os.path.basename(stem)
    candidates = [
        stem + "_depth.png",                                   # sibling suffix form
        os.path.join(os.path.dirname(d), "depths", base + ".png"),  # images/ + depths/
        os.path.join(d, "..", "depths", base + ".png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.normpath(c)
    return None
