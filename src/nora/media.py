"""Pre-action media shared by the downloader and model runner."""

from pathlib import Path
import shutil

from nora.data import index_rows

MEDIA_FILES = {"frames": "frame_all_prev.jpg", "video": "video_prev.mp4"}
MEDIA_REPO = "open-social-world/EgoNormia"


def media_path(root, clip_id, media):
    if media not in MEDIA_FILES:
        raise ValueError("media must be frames or video")
    if not clip_id or clip_id in {".", ".."} or "/" in clip_id or "\\" in clip_id:
        raise ValueError("unsafe_clip_id")
    root = Path(root).resolve()
    path = (root / clip_id / MEDIA_FILES[media]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("unsafe_media_path")
    return path


def download_media(references, *, output, media="frames"):
    """Download pre-action media for an iterable of reference dictionaries.

    Only clip IDs are used. Choose frames (a montage) or video; existing files
    are reused. Return download/reuse counts and per-clip failures.
    """
    from huggingface_hub import hf_hub_download

    rows = index_rows(references)
    paths = [(key, media_path(output, key, media)) for key in rows]
    failures, downloaded, reused = [], 0, 0
    for key, destination in paths:
        if destination.is_file():
            reused += 1
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            cached = hf_hub_download(MEDIA_REPO, f"video/{key}/{MEDIA_FILES[media]}",
                                     repo_type="dataset")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, temporary)
            temporary.replace(destination)
            downloaded += 1
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            failures.append({"clip_id": key, "stage": "download", "error_type": type(exc).__name__})
    return {"output": str(Path(output)), "media": media, "rows": len(rows),
            "downloaded": downloaded, "reused": reused, "invalid": len(failures),
            "failures": failures}
