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
        stem + "_depth.png",  # sibling suffix form
    ]

    # Walk up to 2 levels looking for depths/ sibling.
    # Preserve rig subdirectory when present.
    current_dir = d
    rig_subdir = None  # Remember rig directory if we encounter one
    for _ in range(2):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break  # reached root

        dir_name = os.path.basename(current_dir)

        # Track rig directory on first iteration
        if rig_subdir is None and dir_name.startswith("rig-"):
            rig_subdir = dir_name

        # depths is a sibling of parent_dir
        depths_dir = os.path.join(parent_dir, "depths")

        # Try with rig subdirectory preserved
        if rig_subdir is not None:
            candidates.append(os.path.join(depths_dir, rig_subdir, base + ".png"))

        # Try without rig subdirectory
        candidates.append(os.path.join(depths_dir, base + ".png"))

        current_dir = parent_dir

    for c in candidates:
        if os.path.exists(c):
            return os.path.normpath(c)
    return None
