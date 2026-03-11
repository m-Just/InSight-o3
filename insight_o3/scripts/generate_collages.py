import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


# Rectangle helpers
def rect_area(rect: Tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((x1 - x0) * (y1 - y0))


def rect_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def expand_rect(rect: Tuple[int, int, int, int], g: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return (max(0, x0 - g), max(0, y0 - g), min(W, x1 + g), min(H, y1 + g))


def clip_rect_to_bounds(rect: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    x0 = max(0, min(W, x0))
    y0 = max(0, min(H, y0))
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    if x1 < x0:
        x1 = x0
    if y1 < y0:
        y1 = y0
    return (x0, y0, x1, y1)


def split_free_rectangles(free_list: List[Tuple[int, int, int, int]],
                          used: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    # Guillotine subdivision with containment cleanup
    ux0, uy0, ux1, uy1 = used
    new_free = []
    for fr in free_list:
        fx0, fy0, fx1, fy1 = fr
        if not rect_intersect(fr, used):
            new_free.append(fr)
            continue
        # split
        if fy0 < uy0:
            new_free.append((fx0, fy0, fx1, uy0))
        if uy1 < fy1:
            new_free.append((fx0, uy1, fx1, fy1))
        left_x1 = min(ux0, fx1)
        left_x0 = fx0
        mid_y0 = max(fy0, uy0)
        mid_y1 = min(fy1, uy1)
        if left_x0 < left_x1 and mid_y0 < mid_y1:
            new_free.append((left_x0, mid_y0, left_x1, mid_y1))
        right_x0 = max(ux1, fx0)
        right_x1 = fx1
        if right_x0 < right_x1 and mid_y0 < mid_y1:
            new_free.append((right_x0, mid_y0, right_x1, mid_y1))
    # remove contained rectangles
    pruned = []
    for i in range(len(new_free)):
        a = new_free[i]
        contained = False
        for j in range(len(new_free)):
            if i == j:
                continue
            b = new_free[j]
            if a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]:
                contained = True
                break
        if not contained and rect_area(a) > 0:
            pruned.append(a)
    return pruned


def align_down_to_grid(v: int, G: int) -> int:
    return (int(v) // G) * G


def sample_canvas_dims(A: float, a_ratio: float, G: int) -> Tuple[int, int]:
    W = math.sqrt(A * a_ratio)
    H = A / W
    Wg = max(G, align_down_to_grid(W, G))
    Hg = max(G, align_down_to_grid(H, G))
    return Wg, Hg


def bbox_area_ratio(bbox: Optional[Tuple[float, float, float, float]], S: float) -> float:
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    if S <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    return float((bw * bh) / S)


def parse_bbox(value) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse bbox from various formats to a tuple(float,float,float,float) or None.
    Accepts: list/tuple of 4 numbers; string like "[x0, y0, x1, y1]" or "x0,y0,x1,y1"; dict with keys.
    """
    if value is None:
        return None
    # tuple/list
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            x0, y0, x1, y1 = value
            return (float(x0), float(y0), float(x1), float(y1))
        except Exception:
            return None
    # dict
    if isinstance(value, dict):
        keys = ["x0", "y0", "x1", "y1"]
        if all(k in value for k in keys):
            try:
                return (float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"]))
            except Exception:
                return None
    # string
    if isinstance(value, str):
        s = value.strip()
        try:
            # remove brackets
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            parts = [p for p in s.replace(" ", "").split(",") if p != ""]
            if len(parts) == 4:
                return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        except Exception:
            return None
    return None


def weighted_sample_indices(idx_arr: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    if len(idx_arr) == 0:
        return []
    w = np.maximum(weights.astype(np.float64), 1e-12)
    w = w / w.sum()
    k = min(k, len(idx_arr))
    choice = rng.choice(idx_arr, size=k, replace=False, p=w)
    # preserve original index dtype (don't cast to int)
    return list(choice.tolist())


def center_crop_for_ratio(src_w: int, src_h: int, target_ratio: float) -> Tuple[int, int, int, int]:
    # Returns crop box (x0,y0,x1,y1) centered, achieving target aspect ratio
    if src_w <= 0 or src_h <= 0 or target_ratio <= 0:
        return (0, 0, src_w, src_h)
    cur_ratio = src_w / max(1.0, src_h)
    if abs(math.log(cur_ratio / target_ratio)) < 1e-6:
        return (0, 0, src_w, src_h)
    if cur_ratio > target_ratio:
        # too wide → crop width
        new_w = int(round(src_h * target_ratio))
        new_w = max(1, min(src_w, new_w))
        x0 = (src_w - new_w) // 2
        return (x0, 0, x0 + new_w, src_h)
    else:
        # too tall → crop height
        new_h = int(round(src_w / target_ratio))
        new_h = max(1, min(src_h, new_h))
        y0 = (src_h - new_h) // 2
        return (0, y0, src_w, y0 + new_h)


class Canvas:
    def __init__(self, width: int, height: int, grid: int = 48, margin_g: int = 48, bg_color=(255, 255, 255)):
        """
        Keep placements in self.placements as list of dicts:
        {layout_id, layout_bbox(x0,y0,x1,y1), layout_area, layout_bbox_ratio, src_* fields, scale_ratio (lambda),
         img_crop_region (in source coords or None), object_bbox, object_bbox_ratio, image_type, r_out}
        """
        self.W = int(width)
        self.H = int(height)
        self.G = int(grid)
        self.margin_g = int(margin_g)
        self.bg_color = tuple(bg_color)
        self.placements: List[Dict[str, Any]] = []
        self.free_rects: List[Tuple[int, int, int, int]] = [(0, 0, self.W, self.H)]

    def add_layouts(self, layouts: List[Dict[str, Any]]):
        for l in layouts:
            # Do NOT re-quantize to grid here to avoid drift; only clip to bounds
            bb = clip_rect_to_bounds(l["layout_bbox"], self.W, self.H)
            l["layout_bbox"] = bb
            self.placements.append(l)
            # Reserve safety margin while carving free space
            used = expand_rect(bb, self.margin_g, self.W, self.H)
            self.free_rects = split_free_rectangles(self.free_rects, used)

    def assign_layout_ids(self, start_from: int = 1):
        # Assign IDs by spatial order: top-to-bottom, left-to-right
        indexed = []
        for idx, p in enumerate(self.placements):
            x0, y0, x1, y1 = map(int, p.get("layout_bbox", (0, 0, 0, 0)))
            indexed.append((y0, x0, idx))
        indexed.sort()
        for rank, (_, _, idx) in enumerate(indexed, start=start_from):
            self.placements[idx]["layout_id"] = int(rank)

    @staticmethod
    def _to_jsonable(v):
        if isinstance(v, (tuple, list)):
            return [Canvas._to_jsonable(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, dict):
            return {str(k): Canvas._to_jsonable(val) for k, val in v.items()}
        return v

    def save_json(self, json_path: str):
        # ensure ids assigned upstream (render assigns deterministically)
        # counts of layouts by type
        total_count = int(len(self.placements))
        core_count = int(sum(1 for p in self.placements if str(p.get("image_type", "")).lower() == "core"))
        alt_count = int(sum(1 for p in self.placements if str(p.get("image_type", "")).lower() == "alt"))
        fill_count = int(sum(1 for p in self.placements if str(p.get("image_type", "")).lower() == "fill"))
        data = {
            "canvas": {
                "width": int(self.W),
                "height": int(self.H),
                "aspect_ratio": float(self.W / max(1, self.H)),
                "grid": int(self.G),
                "margin_g": int(self.margin_g),
                "bg_color": list(self.bg_color),
                "layout_count": total_count,
                "core_count": core_count,
                "alt_count": alt_count,
                "fill_count": fill_count,
            },
            "layouts": [],
        }
        keep_keys = {
            "layout_id", "layout_bbox", "layout_area", "layout_bbox_ratio",
            "src_dataset", "src_idx", "src_img", "scale_ratio", "img_crop_region",
            "object_bbox", "object_bbox_ratio", "image_type", "r_out",
            "canvas_object_bbox", "canvas_object_bbox_ratio", "question", "answer",
        }
        for p in self.placements:
            q = {k: self._to_jsonable(p.get(k, None)) for k in keep_keys}
            data["layouts"].append(q)
        import json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_path: str) -> "Canvas":
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # support both new and legacy formats
        if "canvas" in data:
            canvas_info = data["canvas"]
            W = int(canvas_info.get("width", 0))
            H = int(canvas_info.get("height", 0))
            G = int(canvas_info.get("grid", 1))
            M = int(canvas_info.get("margin_g", 0))
            bg = tuple(canvas_info.get("bg_color", [255, 255, 255]))
            placements = data.get("layouts", [])
        else:
            W = int(data.get("width", 0))
            H = int(data.get("height", 0))
            G = int(data.get("grid", 1))
            M = int(data.get("margin_g", 0))
            bg = tuple(data.get("bg_color", [255, 255, 255]))
            placements = data.get("placements", [])
        c = cls(W, H, grid=G, margin_g=M, bg_color=bg)
        # normalize bbox types
        norm = []
        for p in placements:
            if p.get("layout_bbox") is not None:
                bb = tuple(map(int, p["layout_bbox"]))
                p["layout_bbox"] = bb
            if p.get("img_crop_region") is not None:
                cr = tuple(map(int, p["img_crop_region"]))
                p["img_crop_region"] = cr
            if p.get("object_bbox") is not None:
                ob = tuple(map(float, p["object_bbox"]))
                p["object_bbox"] = ob
            norm.append(p)
        c.add_layouts(norm)
        return c

    def render(self, save_path: Optional[str] = None,
               draw_edges: bool = False,
               edge_color: Optional[Tuple[int, int, int]] = None,
               print_layout: bool = False,
               plot_bbox: bool = False,
               show_label: bool = False,
               edge_by_type: bool = True,
               label_color: Tuple[int, int, int] = (0, 0, 0)):
        img = Image.new("RGB", (self.W, self.H), self.bg_color)
        need_draw = bool(draw_edges or show_label or plot_bbox)
        draw = ImageDraw.Draw(img) if need_draw else None
        # cache font once per render
        font = None
        if show_label:
            try:
                from PIL import ImageFont
                font_size = 20
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except Exception:
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()
            except Exception:
                font = None
        # ensure layout ids are present
        self.assign_layout_ids()
        for p in self.placements:
            x0, y0, x1, y1 = map(int, p["layout_bbox"])
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            src = p.get("src_img", None)
            crop = p.get("img_crop_region", None)
            if src is not None:
                with Image.open(src) as _im:
                    im = _im.convert("RGB")
                    if crop is not None:
                        cx0, cy0, cx1, cy1 = map(int, crop)
                        cx0 = max(0, cx0)
                        cy0 = max(0, cy0)
                        cx1 = min(im.width, cx1)
                        cy1 = min(im.height, cy1)
                        if cx1 > cx0 and cy1 > cy0:
                            im = im.crop((cx0, cy0, cx1, cy1))
                    im = im.resize((w, h), Image.Resampling.LANCZOS)
                    img.paste(im, (x0, y0))
            # record final object bbox on canvas if available
            if p.get("object_bbox") is not None and p.get("_src_S") is not None and p.get("_src_r") is not None:
                try:
                    ob = parse_bbox(p.get("object_bbox"))
                    if ob is None:
                        raise ValueError("invalid bbox")
                    bx0, by0, bx1, by1 = map(float, ob)
                    # source dimensions reconstructed from area and aspect ratio
                    S_src = float(p.get("_src_S"))
                    r_src = float(p.get("_src_r"))
                    src_w = math.sqrt(max(1e-8, S_src * r_src))
                    src_h = math.sqrt(max(1e-8, S_src / r_src))
                    # if cropped, adjust bbox and source dims
                    crop = p.get("img_crop_region", None)
                    if crop is not None:
                        cx0, cy0, cx1, cy1 = map(float, crop)
                        bx0 -= cx0; by0 -= cy0; bx1 -= cx0; by1 -= cy0
                        src_w = max(1.0, cx1 - cx0)
                        src_h = max(1.0, cy1 - cy0)
                    # placed dimensions
                    placed_w = max(1.0, w)
                    placed_h = max(1.0, h)
                    scale_x = placed_w / src_w
                    scale_y = placed_h / src_h
                    fx0 = x0 + int(round(bx0 * scale_x))
                    fy0 = y0 + int(round(by0 * scale_y))
                    fx1 = x0 + int(round(bx1 * scale_x))
                    fy1 = y0 + int(round(by1 * scale_y))
                    # clip to layout bounds to avoid overspill
                    fx0 = max(x0, min(x1, fx0)); fx1 = max(x0, min(x1, fx1))
                    fy0 = max(y0, min(y1, fy0)); fy1 = max(y0, min(y1, fy1))
                    p["canvas_object_bbox"] = (fx0, fy0, fx1, fy1)
                    can_area = max(1, self.W * self.H)
                    p["canvas_object_bbox_ratio"] = round(rect_area((fx0, fy0, fx1, fy1)) / can_area, 5)
                except Exception:
                    p["canvas_object_bbox"] = None
                    p["canvas_object_bbox_ratio"] = None
            if draw_edges and draw is not None:
                if edge_by_type:
                    t = str(p.get("image_type", "")).lower()
                    type_color_map = {
                        "core": (255, 0, 0),
                        "alt": (0, 128, 255),
                        "fill": (0, 200, 0),
                    }
                    c = type_color_map.get(t, (0, 0, 0) if edge_color is None else tuple(edge_color))
                else:
                    c = (0, 0, 0) if edge_color is None else tuple(edge_color)
                # shrink by 1px to ensure edge is visible within bounds
                ex1 = max(x0 + 1, x1 - 1)
                ey1 = max(y0 + 1, y1 - 1)
                draw.rectangle([x0, y0, ex1, ey1], outline=c, width=3)
            if show_label and draw is not None:
                lc = tuple(label_color)
                lid = p["layout_id"]
                text = f"Image {int(lid)}"
                # draw with stroke to reduce blur
                draw.text((x0 + 6, y0 + 6), text, fill=lc, font=font, stroke_width=2, stroke_fill=(255, 255, 255))

            # optionally draw bbox
            if plot_bbox and draw is not None:
                bb = p.get("canvas_object_bbox")
                # if not computed yet, attempt compute quickly
                if bb is None and p.get("object_bbox") is not None and p.get("scale_ratio") is not None:
                    try:
                        ob = parse_bbox(p.get("object_bbox"))
                        if ob is not None:
                            scale = math.sqrt(float(p.get("scale_ratio")))
                            bx0, by0, bx1, by1 = ob
                            bb = (x0 + int(round(bx0 * scale)), y0 + int(round(by0 * scale)), x0 + int(round(bx1 * scale)), y0 + int(round(by1 * scale)))
                            p["canvas_object_bbox"] = bb
                            p["canvas_object_bbox_ratio"] = round(rect_area(bb) / max(1, self.W * self.H), 5)
                    except Exception:
                        bb = None
                if bb is not None:
                    draw.rectangle(list(map(int, bb)), outline=(255, 0, 0), width=5)

        if save_path:
            img.save(save_path)
            # also save json next to image
            try:
                base, _ = os.path.splitext(save_path)
                self.save_json(base + ".json")
            except Exception as _e:
                print("[render] failed to save json:", _e)
        if print_layout:
            try:
                import pandas as _pd
                cols = [
                    "layout_id", "image_type", "src_dataset", "src_idx", "src_img",
                    "layout_bbox", "layout_area", "layout_bbox_ratio", "scale_ratio", "r_out",
                    "src_object_bbox", "src_object_bbox_ratio", "canvas_object_bbox", "canvas_object_bbox_ratio",
                ]
                rows = []
                for p in self.placements:
                    row = {k: p.get(k, None) for k in cols}
                    # map keys for src naming
                    row["src_object_bbox"] = p.get("object_bbox", None)
                    val_ratio = p.get("object_bbox_ratio", None)
                    row["src_object_bbox_ratio"] = round(float(val_ratio), 5) if isinstance(val_ratio, (int, float)) else val_ratio
                    # format numeric to 5 decimals
                    for nk in ("layout_bbox_ratio", "scale_ratio", "r_out", "canvas_object_bbox_ratio"):
                        if row.get(nk) is not None and isinstance(row[nk], (int, float)):
                            row[nk] = round(float(row[nk]), 5)
                    rows.append(row)
                tbl = _pd.DataFrame(rows, columns=cols)
                with _pd.option_context('display.max_colwidth', 256, 'display.width', 200):
                    print(tbl.to_string(index=False))
            except Exception as _e:
                print("[render] table print failed:", _e)
        return img


class CollageGenerator:
    def __init__(self, data_df: pd.DataFrame, rng: Optional[np.random.Generator] = None):
        self.df = data_df
        # Allow rng to be a numpy Generator, an int seed, or None
        if isinstance(rng, (int, np.integer)):
            self.rng = np.random.default_rng(int(rng))
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self._recent = deque(maxlen=64)  # cooling set (optional enhancement B)

    def generate(self, index, **kwargs):
        params = self._default_params()
        params.update(kwargs or {})
        max_attempts = int(params.get("max_attempts", 4))
        # ensure grid equals min_short_edge for tiling without gaps and satisfying min side
        if params.get("min_short_edge"):
            params["grid"] = int(params["min_short_edge"])
        for attempt in range(max_attempts):
            # 1) select target set and canvas
            # selection strategy: random first, after first failed attempt switch to pressure-guided
            sel_params = dict(params)
            sel_params["selection_mode"] = "random" if attempt == 0 else "pressure"
            selection = self._select_targets_and_canvas(self.df, sel_params)
            if selection is None:
                continue
            canvas_W, canvas_H, A_canvas, selected_df, lam_bounds, rho_max_arr, _ = selection
            canvas = Canvas(canvas_W, canvas_H, grid=params["grid"], margin_g=params["margin_g"], bg_color=params["bg_color"])
            # 2) assign lambda within bounds with occupancy control and min-short-edge
            placements = self._plan_target_rectangles(selected_df, A_canvas, lam_bounds, rho_max_arr, params, canvas_W, canvas_H)
            if placements is None:
                # try once by expanding canvas
                A_canvas2 = min(params["A_max"], A_canvas * 1.2)
                a_ratio2 = float(self.rng.uniform(params["a_min"], params["a_max"]))
                canvas_W, canvas_H = sample_canvas_dims(A_canvas2, a_ratio2, params["grid"])
                canvas = Canvas(canvas_W, canvas_H, grid=params["grid"], margin_g=params["margin_g"], bg_color=params["bg_color"])
                placements = self._plan_target_rectangles(selected_df, A_canvas2, lam_bounds, rho_max_arr, params, canvas_W, canvas_H)
                if placements is None:
                    continue
                A_canvas = A_canvas2
            # 3) place targets on canvas by grid-aligned rejection sampling with post-checks
            ok, placed_targets = self._place_rects(canvas_W, canvas_H, placements, params, A_canvas)
            if not ok:
                # one-shot expansion retry
                A_canvas_retry = min(params["A_max"], A_canvas * 1.2)
                if A_canvas_retry > A_canvas + 1e-6:
                    a_ratio_retry = float(self.rng.uniform(params["a_min"], params["a_max"]))
                    canvas_W, canvas_H = sample_canvas_dims(A_canvas_retry, a_ratio_retry, params["grid"])
                    canvas = Canvas(canvas_W, canvas_H, grid=params["grid"], margin_g=params["margin_g"], bg_color=params["bg_color"])
                    placements = self._plan_target_rectangles(selected_df, A_canvas_retry, lam_bounds, rho_max_arr, params, canvas_W, canvas_H)
                    if placements is None:
                        continue
                    ok, placed_targets = self._place_rects(canvas_W, canvas_H, placements, params, A_canvas_retry)
                    if not ok:
                        continue
                    A_canvas = A_canvas_retry
                else:
                    continue
            # attach and update free space
            canvas.add_layouts(placed_targets)
            # validate free rectangles short edge
            if not self._validate_free_rects(canvas, params):
                continue
            # 4) normalize all free rectangles by constraints (no count cap)
            self._normalize_free_space(canvas, params)
            # validate again after normalization

            if not self._validate_free_rects(canvas, params):
                continue
            # 5) assign fill images to free panels (always fill with fallback)
            fill_layouts = self._fill_free_panels(canvas, params)
            if fill_layouts:
                canvas.add_layouts(fill_layouts)
            # ensure no free space remains
            # after add_layouts, free_rects already carved; if any remain, we failed
            if len(canvas.free_rects) > 0:
                # as a safety, try one more pass without splitting
                extra_fills = self._fill_free_panels(canvas, params)
                if extra_fills:
                    canvas.add_layouts(extra_fills)
            if len(canvas.free_rects) > 0:
                continue
            # 6) render and update usage/cooling
            save_dir = params.get("save_path")
            if save_dir:
                # shard into subdirs of 1000 to avoid flat-directory metadata bottleneck
                shard = index // 1000
                shard_dir = os.path.join(str(save_dir), "images", str(shard))
                os.makedirs(shard_dir, exist_ok=True)
                out_file = os.path.join(shard_dir, f"{index}.jpg")
                canvas.render(
                    save_path=out_file,
                    draw_edges=params.get("draw_edges", False),
                    print_layout=params.get("print_layout", False),
                    plot_bbox=params.get("plot_bbox", False),
                    show_label=params.get("show_label", False),
                    edge_by_type=params.get("edge_by_type", True),
                    edge_color=tuple(params.get("edge_color", (0, 0, 0))),
                    label_color=tuple(params.get("label_color", (0, 0, 0))),
                )
            self._bump_usage_counts(placed_targets, fill_layouts)
            for pz in (placed_targets or []):
                self._recent.append(pz.get("_src_index", None))
            for fl in (fill_layouts or []):
                self._recent.append(fl.get("_src_index", None))
            return {"canvas": canvas, "targets": placed_targets, "fills": fill_layouts}
        return None

    # ---- Internal helpers ----
    def _default_params(self) -> Dict[str, Any]:
        return {
            "max_attempts": 5,
            # counts
            "K_min": 4, "K_max": 6,
            "core_min": 2, "core_max": 3,
            # canvas area and aspect
            "A_min": 3e6, "A_max": 1.2e7,  # area: 1732^2 to 3464^2
            "a_min": 0.7, "a_max": 1.6,  # aspect ratio: 0.7 to 1.6
            "bg_color": (255, 255, 255),
            # bbox proportion caps
            "rho_max_default": 2e-4,  # cap of target bbox on canvas
            # lambda bounds
            "lambda_up": 1,  # ratio to scale up image
            "lambda_down_base": 0.22,
            # ratio tolerance
            "tau_target": 0.08,  # small distortion for target
            "tau_fill": 0.5,
            # occupancy
            "beta": 0.9,  # upper bound of total area ratio of target image on canvas
            # placement of target image
            "margin_g": 0,
            "place_max_retry": 50,
            "shrink_step": 0.95,
            # minimum short edge for any placed image (targets and fills)
            "min_short_edge": 224,
            # fill strategy
            "fill_lambda_min": 0.45, "fill_lambda_max": 1.8,
            "fill_area_min_panel": None,  # will default to 2*grid*grid if None
            # panel ratio limits (separate from canvas)
            "panel_ratio_min": 0.4, "panel_ratio_max": 2.5,
            # crop policy for fill fallback
            "crop_min_frac": 0.5,
            # usage weights / cooldown
            # higher alpha gives stronger inverse-usage preference (replaces hard zero-only filter)
            "alpha_usage": 4,
            "cooldown_enable": False,
            # source preprocessing (area cap)
            "src_max_area": 1024 * 1024,
            # columns
            "src_img_col": "image",
            "src_w_col": "img_width",
            "src_h_col": "img_height",
            "bbox_col": "bbox",  # [x0,y0,x1,y1] or None
            "rho_src_col": "bbox_ratio",
            "core_count_col": "core_count",
            "alt_count_col": "alt_count",
            "fill_count_col": "fill_count",
            # dataset default
            "default_dataset": "default",
            # render
            "save_path": None,
            "draw_edges": False,
            "plot_bbox": False,
            "print_layout": False,
            "edge_by_type": True,
            "edge_color": (0, 0, 0),
            "show_label": True,
            "label_color": (0, 0, 0),
            # selection
            "selection_mode": "random",  # 'random' first; on retries you can set to 'pressure'
        }

    def _select_targets_and_canvas(self, df: pd.DataFrame, p: Dict[str, Any]):
        wcol, hcol = p["src_w_col"], p["src_h_col"]
        bbox_col, rho_src_col = p["bbox_col"], p["rho_src_col"]
        img_col = p["src_img_col"]
        # Ensure computed columns exist
        # filter invalid src_img rows upfront
        df = df[(df[img_col].notna()) & (df[img_col] != "")]
        if "area" not in df.columns:
            df["area"] = (df[wcol].astype(float) * df[hcol].astype(float)).astype(float)
        if "ratio" not in df.columns:
            df["ratio"] = (df[wcol].astype(float) / np.maximum(1.0, df[hcol].astype(float))).astype(float)
        if rho_src_col not in df.columns:
            def _rho(row):
                bbox = parse_bbox(row.get(bbox_col, None))
                A = float(row.get("area", 0.0) or 0.0)
                if bbox is not None and A > 0:
                    return bbox_area_ratio(bbox, A)
                return 0.0
            df[rho_src_col] = df.apply(_rho, axis=1).astype(float)

        # target counts
        K = int(self.rng.integers(p["K_min"], p["K_max"] + 1))
        n_core = int(self.rng.integers(p["core_min"], p["core_max"] + 1))
        n_alt = max(0, K - n_core)

        # cap source by area uniformly (preserve aspect): effective area for planning/selection
        cap_area = float(p.get("src_max_area", 1024 * 1024))
        src_w_all = df[wcol].astype(float).to_numpy()
        src_h_all = df[hcol].astype(float).to_numpy()
        S_raw = np.maximum(1.0, src_w_all * src_h_all)
        s_pre_all = np.minimum(1.0, np.sqrt(cap_area / S_raw))
        S_all = (S_raw * (s_pre_all ** 2)).astype(np.float64)
        # persist effective planning scalars
        df["_pre_scale"] = s_pre_all
        df["_area_eff"] = S_all
        # unified lambda_down for all images
        lam_down = np.full(len(df), float(p["lambda_down_base"]), dtype=np.float64)
        df["_lam_down"] = lam_down
        rho_src_all = df[rho_src_col].to_numpy()
        rho_max_default = p["rho_max_default"]
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(rho_src_all > 0, (rho_src_all * lam_down * S_all) / max(rho_max_default, 1e-8), 0.0)
        df["_pressure"] = P
        # compute a simple hardness score in [0, ~1.2]: normalized pressure + small aspect deviation term
        if len(df) > 0:
            Pn = P.copy()
            pmin = float(np.nanmin(Pn)) if np.isfinite(Pn).any() else 0.0
            pmax = float(np.nanmax(Pn)) if np.isfinite(Pn).any() else 1.0
            Pn = (Pn - pmin) / max(1e-8, (pmax - pmin))
            rdev = np.abs(np.log(np.maximum(1e-8, df["ratio"].astype(float).to_numpy())))
            rdev = np.minimum(2.0, rdev) / 2.0  # cap to 1.0
            df["_hardness"] = (Pn + 0.2 * rdev).astype(float)
        else:
            df["_hardness"] = 0.0

        def inv_usage_weights(sub: pd.DataFrame, count_col: str) -> np.ndarray:
            if count_col not in sub.columns:
                return np.ones(len(sub), dtype=np.float64)
            u = sub[count_col].fillna(0).to_numpy(dtype=np.float64)
            return 1.0 / (1.0 + p["alpha_usage"] * u)

        # cooling exclusion
        recent_set = set(self._recent) if p.get("cooldown_enable", True) else set()

        # core candidates: entire pool with inverse-usage weights; no hard zero-only filter
        # so the sampler never deadlocks when zero-count images are exhausted in this chunk
        core_cand = df.copy()
        if p.get("cooldown_enable", True) and len(recent_set) > 0:
            core_cand = core_cand[~core_cand.index.isin(recent_set)].copy()
        if len(core_cand) < n_core:
            return None
        # mixed hardness sampling: pick a portion from hard bucket to avoid tail accumulation
        def sample_mixed(pool: pd.DataFrame, k: int, role_col: str, mode: str) -> List[int]:
            if k <= 0 or len(pool) == 0:
                return []
            arr_idx = pool.index.to_numpy()
            h = pool.get("_hardness", pd.Series(0.0, index=pool.index)).to_numpy(dtype=np.float64)
            if len(h) > 2:
                q2 = float(np.quantile(h, 0.66))
            else:
                q2 = float(h.max())
            hard_mask = (h > q2)
            hard_pool = pool.loc[arr_idx[hard_mask]] if hard_mask.any() else pool.iloc[0:0]
            ez_pool = pool.loc[arr_idx[~hard_mask]] if (~hard_mask).any() else pool.iloc[0:0]
            k_hard = int(round(0.4 * k))
            k_ez = int(k - k_hard)
            def weights(dfsub: pd.DataFrame) -> np.ndarray:
                base = inv_usage_weights(dfsub, role_col)
                if mode == "random" or len(dfsub) == 0:
                    return base
                press = dfsub.get("_pressure", pd.Series(0.0, index=dfsub.index)).to_numpy(dtype=np.float64)
                if len(press) > 1:
                    pr = (press - press.min()) / max(1e-8, (press.max() - press.min()))
                else:
                    pr = np.zeros_like(base)
                return base * (1.0 - 0.4 * pr)
            chosen: List[int] = []
            # sample from easy/mid
            if len(ez_pool) > 0 and k_ez > 0:
                w = weights(ez_pool)
                chosen += weighted_sample_indices(ez_pool.index.to_numpy(), w, k_ez, self.rng)
            # sample from hard
            if len(hard_pool) > 0 and k_hard > 0:
                w = weights(hard_pool)
                # avoid duplicates
                base_idx = [ix for ix in hard_pool.index.to_numpy() if ix not in chosen]
                chosen += weighted_sample_indices(np.array(base_idx, dtype=int), weights(hard_pool.loc[base_idx]), k_hard, self.rng) if len(base_idx) > 0 else []
            # if still short, fill from full pool excluding already chosen
            if len(chosen) < k:
                rem = k - len(chosen)
                rest = [ix for ix in pool.index.to_numpy() if ix not in chosen]
                if len(rest) > 0:
                    w = weights(pool.loc[rest])
                    chosen += weighted_sample_indices(np.array(rest, dtype=int), w, rem, self.rng)
            return chosen[:k]

        if p.get("selection_mode", "random") == "random":
            core_idx = sample_mixed(core_cand.copy(), n_core, p["core_count_col"], mode="random")
        else:
            core_idx = sample_mixed(core_cand.copy(), n_core, p["core_count_col"], mode="pressure")

        if len(core_idx) < n_core:
            return None

        # alt candidates: from remaining pool, usage-aware and low pressure
        remain_pool = df.drop(index=core_idx, errors="ignore").copy()
        if p.get("cooldown_enable", True) and len(recent_set) > 0:
            filtered = remain_pool[~remain_pool.index.isin(recent_set)].copy()
            if len(filtered) > 0:
                remain_pool = filtered
        if len(remain_pool) < n_alt:
            return None

        # use full remaining pool with inverse-usage weights; no hard zero-only filter
        alt_pool = remain_pool

        if p.get("selection_mode", "random") == "random":
            alt_idx = sample_mixed(alt_pool.copy(), n_alt, p["alt_count_col"], mode="random")
        else:
            alt_idx = sample_mixed(alt_pool.copy(), n_alt, p["alt_count_col"], mode="pressure")
        if len(alt_idx) < n_alt:
            return None

        sel_idx = core_idx + alt_idx
        selected_df = df.loc[sel_idx].copy()
        # annotate roles
        role_series = pd.Series(index=selected_df.index, dtype=object)
        role_series.loc[core_idx] = "core"
        role_series.loc[alt_idx] = "alt"
        selected_df["selected_role"] = role_series

        # Feasibility lower bound A_LB from bbox proportion
        rho_src_sel = selected_df[rho_src_col].fillna(0).to_numpy(dtype=np.float64)
        S_sel = selected_df["_area_eff"].to_numpy(dtype=np.float64)
        lam_down_sel = selected_df["_lam_down"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            A_LB_i = np.where(rho_src_sel > 0, (rho_src_sel * lam_down_sel * S_sel) / rho_max_default, 0.0)
        A_LB = float(np.nanmax(A_LB_i)) if len(A_LB_i) > 0 else 0.0
        A_min = max(p["A_min"], A_LB)
        if A_min > p["A_max"]:
            return None

        # Sample canvas area and aspect
        A_canvas = float(self.rng.uniform(A_min, p["A_max"]))
        a_ratio = float(self.rng.uniform(p["a_min"], p["a_max"]))
        W, H = sample_canvas_dims(A_canvas, a_ratio, p["grid"])

        # Build lambda bounds array, and rho_max per image
        lam_up = np.full(len(selected_df), float(p["lambda_up"]), dtype=np.float64)
        lam_down_arr = selected_df["_lam_down"].to_numpy(dtype=np.float64)
        rho_max_arr = np.full(len(selected_df), float(p["rho_max_default"]), dtype=np.float64)

        return (W, H, A_canvas, selected_df, (lam_down_arr, lam_up), rho_max_arr, a_ratio)

    def _plan_target_rectangles(self, selected_df: pd.DataFrame, A_canvas: float,
                                lam_bounds: Tuple[np.ndarray, np.ndarray], rho_max_arr: np.ndarray,
                                p: Dict[str, Any], W: int, H: int):
        rho_src = selected_df[p["rho_src_col"]].fillna(0).to_numpy(dtype=np.float64)
        # use effective area for lambda computations post pre-scale
        S = selected_df["_area_eff"].to_numpy(dtype=np.float64)
        r_src = selected_df["ratio"].to_numpy(dtype=np.float64)
        lam_down, lam_up_base = lam_bounds
        with np.errstate(divide="ignore", invalid="ignore"):
            cap = np.where(rho_src > 0, (A_canvas * rho_max_arr) / np.maximum(1e-8, rho_src * S), np.inf)
        lam_up = np.minimum(lam_up_base, cap)
        # Prioritize bbox cap: if conflict (lam_down > lam_up) for bbox items, relax lam_down to lam_up
        lam_down_eff = np.array(lam_down, dtype=np.float64)
        has_bbox_mask = (rho_src > 0)
        lam_down_eff = np.where(has_bbox_mask, np.minimum(lam_down_eff, lam_up), lam_down_eff)
        feasible = lam_down_eff <= np.minimum(lam_up, np.inf)
        if not np.all(feasible):
            return None
        lam_pref = 0.5 * (lam_down_eff + lam_up)
        lam = np.clip(lam_pref, lam_down_eff, lam_up)

        A_targets = lam * S
        if A_targets.sum() > p["beta"] * A_canvas:
            scale = (p["beta"] * A_canvas) / max(1e-8, A_targets.sum())
            lam_scaled = lam * scale
            lam = np.maximum(lam_down, lam_scaled)
            A_targets = lam * S
            if A_targets.sum() > p["beta"] * A_canvas * 1.02:
                return None

        tau = float(p["tau_target"])
        r_low = r_src * math.exp(-tau)
        r_high = r_src * math.exp(+tau)
        r_out = self.rng.uniform(r_low, r_high)

        # pixel target sizes, then enforce min short edge, then grid-align down
        G = int(p["grid"])
        M = int(p.get("min_short_edge", 224))
        w_f = np.sqrt(np.maximum(1.0, A_targets) * r_out)
        h_f = np.sqrt(np.maximum(1.0, A_targets) / r_out)
        # enforce min short edge by scaling up lambda if possible
        min_side = np.minimum(w_f, h_f)
        sf = np.maximum(1.0, np.maximum(M / np.maximum(min_side, 1e-6), 1.0))
        lam_current = A_targets / np.maximum(1e-8, S)
        lam_needed = lam_current * (sf ** 2)
        # Check against lam upper bound considering bbox caps computed earlier
        with np.errstate(divide="ignore", invalid="ignore"):
            cap2 = np.where(selected_df[p["rho_src_col"]].fillna(0).to_numpy(dtype=np.float64) > 0,
                            (A_canvas * rho_max_arr) / np.maximum(1e-8, selected_df[p["rho_src_col"]].fillna(0).to_numpy(dtype=np.float64) * S),
                            np.inf)
        lam_up2 = np.minimum(lam_bounds[1], cap2)
        if not np.all(lam_needed <= lam_up2 + 1e-9):
            return None
        lam = lam_needed
        A_targets = lam * S
        # occupancy re-check
        if A_targets.sum() > p["beta"] * A_canvas:
            return None
        w = np.maximum(G, (np.floor(w_f / G) * G)).astype(int)
        h = np.maximum(G, (np.floor(h_f / G) * G)).astype(int)
        # after grid align, ensure min side still >= M
        too_small = (np.minimum(w, h) < M)
        if np.any(too_small):
            return None

        placements = []
        for i, (idx, row) in enumerate(selected_df.iterrows()):
            # source fields with fallbacks
            ds_val = row.get("src_dataset", p.get("default_dataset", "default"))
            if pd.isna(ds_val) if isinstance(ds_val, (float, np.floating)) else False:
                ds_val = p.get("default_dataset", "default")
            src_idx_val = row.get("src_idx", None)
            if src_idx_val is None or (isinstance(src_idx_val, (float, np.floating)) and pd.isna(src_idx_val)):
                src_idx_val = int(idx)
            src_img_val = row.get(p["src_img_col"], None)
            if src_img_val is None or (isinstance(src_img_val, float) and pd.isna(src_img_val)) or src_img_val == "":
                continue  # skip invalid source image
            # parse and attach src bbox normalized info for later
            src_bbox = parse_bbox(row.get(p["bbox_col"], None))
            src_bbox_ratio = float(row.get(p["rho_src_col"], 0.0) or 0.0)
            placements.append({
                "layout_bbox": (0, 0, int(w[i]), int(h[i])),
                "layout_area": int(w[i] * h[i]),
                "layout_bbox_ratio": float((w[i] * h[i]) / max(1.0, A_canvas)),
                "src_dataset": ds_val,
                "src_idx": src_idx_val,
                "src_img": src_img_val,
                "scale_ratio": float(lam[i]),  # initial; may shrink in placement
                "img_crop_region": None,
                "object_bbox": src_bbox,
                "object_bbox_ratio": src_bbox_ratio,
                "image_type": row["selected_role"],
                "r_out": float(r_out[i]),
                "_src_index": int(idx),
                "_src_S": float(row["area"]),
                "_src_r": float(row["ratio"]),
                "_pre_scale": float(row.get("_pre_scale", 1.0)),
                "_has_bbox": bool(row.get(p["bbox_col"], None) is not None and row.get(p["rho_src_col"], 0.0) > 0),
                "_lam_down": float(lam_down_eff[i]),
                "_rho_max": float(rho_max_arr[i]),
                "_rho_src": src_bbox_ratio,
                "question": row.get("question", None),
                "answer": row.get("answer", None),
            })
        placements.sort(key=lambda d: -d["layout_area"])
        return placements

    def _place_rects(self, W: int, H: int, placements: List[Dict[str, Any]], p: Dict[str, Any], A_canvas: float):
        G = int(p["grid"])
        margin = int(p["margin_g"])
        retries = int(p["place_max_retry"])
        shrink = float(p["shrink_step"])
        M = int(p.get("min_short_edge", 224))

        # local free rectangles start with the full canvas
        free_rects: List[Tuple[int, int, int, int]] = [(0, 0, W, H)]
        placed = []

        # small helper to align up to grid (keep local to avoid changing global API)
        def _align_up_to_grid(v: int, Gv: int) -> int:
            iv = int(v)
            return ((iv + Gv - 1) // Gv) * Gv

        # Hard-first: bbox-capped first, then larger area
        items = sorted(
            placements,
            key=lambda d: (not bool(d.get("_has_bbox", False)), -int(d.get("layout_area", 0)))
        )
        for item in items:
            w0 = item["layout_bbox"][2] - item["layout_bbox"][0]
            h0 = item["layout_bbox"][3] - item["layout_bbox"][1]
            cur_w, cur_h = int(w0), int(h0)
            ok = False
            cur_item = dict(item)

            # try to place with at most `retries` size adjustments
            for attempt in range(retries):
                # grid-quantize current size and enforce minimum short edge
                cur_w = max(G, align_down_to_grid(cur_w, G))
                cur_h = max(G, align_down_to_grid(cur_h, G))
                if min(cur_w, cur_h) < M:
                    cur_w = max(cur_w, align_down_to_grid(M, G))
                    cur_h = max(cur_h, align_down_to_grid(M, G))
                if cur_w > W or cur_h > H:
                    # shrink if larger than canvas
                    cur_w = max(G, align_down_to_grid(int(cur_w * shrink), G))
                    cur_h = max(G, align_down_to_grid(int(cur_h * shrink), G))
                    continue

                # lambda checks depend only on area, not position
                new_area = int(cur_w * cur_h)
                src_S = float(cur_item["_src_S"])
                new_lam = float(new_area / max(1.0, src_S))
                lam_down_i = float(cur_item["_lam_down"])
                # If has bbox, relax the lam_down guard to prioritize bbox-cap; allow extra shrinking
                if new_lam < lam_down_i - 1e-9 and not bool(cur_item.get("_has_bbox", False)):
                    # need to shrink less aggressively; try a gentler decrease
                    cur_w = max(G, align_down_to_grid(int(cur_w * shrink), G))
                    cur_h = max(G, align_down_to_grid(int(cur_h * shrink), G))
                    continue
                if cur_item["_has_bbox"]:
                    rho_src = float(cur_item["_rho_src"])
                    rho_max = float(cur_item["_rho_max"])
                    if (rho_src * new_lam * src_S) / max(1.0, A_canvas) > rho_max + 1e-12:
                        # area too big for bbox cap → shrink
                        cur_w = max(G, align_down_to_grid(int(cur_w * shrink), G))
                        cur_h = max(G, align_down_to_grid(int(cur_h * shrink), G))
                        continue

                # collect free rects and compute slack (area slack) for those that fit
                candidates = []
                best_s_fit = 0.0
                best_fr_for_shrink = None
                for fr in free_rects:
                    fx0, fy0, fx1, fy1 = fr
                    fw, fh = fx1 - fx0, fy1 - fy0
                    if cur_w <= fw and cur_h <= fh:
                        slack = (fw * fh) - (cur_w * cur_h)
                        candidates.append((fr, slack))
                    # track best shrink-to-fit factor
                    s_fit = min(fw / max(1e-8, cur_w), fh / max(1e-8, cur_h))
                    if s_fit > best_s_fit:
                        best_s_fit = s_fit
                        best_fr_for_shrink = fr
                if not candidates:
                    # targeted shrink-to-fit if possible
                    if best_s_fit > 0.0 and best_s_fit < 1.0 and best_fr_for_shrink is not None:
                        jitter = float(self.rng.uniform(0.97, 1.00))
                        s_use = max(0.0, best_s_fit * jitter)
                        tw = max(G, align_down_to_grid(int(cur_w * s_use), G))
                        th = max(G, align_down_to_grid(int(cur_h * s_use), G))
                        if min(tw, th) >= M:
                            new_area2 = int(tw * th)
                            new_lam2 = float(new_area2 / max(1.0, float(cur_item["_src_S"])))
                            if new_lam2 >= float(cur_item["_lam_down"]) - 1e-9:
                                cur_w, cur_h = tw, th
                                # retry with new size
                                continue
                    # fallback gentle shrink
                    cur_w = max(G, align_down_to_grid(int(cur_w * shrink), G))
                    cur_h = max(G, align_down_to_grid(int(cur_h * shrink), G))
                    continue

                # best-fit by minimal area slack; keep slight randomness among top-3
                candidates.sort(key=lambda t: t[1])
                topk = candidates[:min(3, len(candidates))]
                pick_idx = int(self.rng.integers(0, len(topk))) if len(topk) > 1 else 0
                fr, _ = topk[pick_idx]

                fx0, fy0, fx1, fy1 = fr
                # feasible top-left ranges
                x0_min = _align_up_to_grid(fx0, G)
                y0_min = _align_up_to_grid(fy0, G)
                x0_max = align_down_to_grid(fx1 - cur_w, G)
                y0_max = align_down_to_grid(fy1 - cur_h, G)
                if x0_min > x0_max or y0_min > y0_max:
                    # should not happen if we filtered correctly; try next attempt
                    cur_w = max(G, align_down_to_grid(int(cur_w * shrink), G))
                    cur_h = max(G, align_down_to_grid(int(cur_h * shrink), G))
                    continue

                # unbiased random sampling of positions, optionally include center with p=0.3
                steps_x = max(1, (x0_max - x0_min) // G + 1)
                steps_y = max(1, (y0_max - y0_min) // G + 1)
                n_samples = int(min(16, steps_x * steps_y))
                tried = set()
                samples = []
                if float(self.rng.random()) < 0.3:
                    cx = align_down_to_grid((fx0 + fx1 - cur_w) // 2, G)
                    cy = align_down_to_grid((fy0 + fy1 - cur_h) // 2, G)
                    cx = int(min(max(cx, x0_min), x0_max))
                    cy = int(min(max(cy, y0_min), y0_max))
                    samples.append((cx, cy))
                for _t in range(n_samples):
                    rx_idx = int(self.rng.integers(0, steps_x))
                    ry_idx = int(self.rng.integers(0, steps_y))
                    rx0 = x0_min + rx_idx * G
                    ry0 = y0_min + ry_idx * G
                    samples.append((rx0, ry0))

                placed_here = False
                for x0, y0 in samples:
                    key = (x0, y0)
                    if key in tried:
                        continue
                    tried.add(key)
                    x1, y1 = x0 + cur_w, y0 + cur_h
                    cand = (x0, y0, x1, y1)
                    cand = clip_rect_to_bounds(cand, W, H)
                    if rect_area(cand) != cur_w * cur_h:
                        continue
                    # accept this placement
                    cur_item["layout_bbox"] = cand
                    cur_item["layout_area"] = int(cur_w * cur_h)
                    cur_item["scale_ratio"] = float(new_lam)
                    placed.append(cur_item)
                    # carve and prune slivers
                    used = expand_rect(cand, margin, W, H)
                    free_rects = split_free_rectangles(free_rects, used)
                    free_rects = [r for r in free_rects if rect_area(r) > 0 and min(r[2] - r[0], r[3] - r[1]) >= M]
                    ok = True
                    placed_here = True
                    break
                if placed_here:
                    break

                # if we failed to place at this size, shrink and retry
                cur_w = max(G, align_down_to_grid(int(cur_w * shrink), G))
                cur_h = max(G, align_down_to_grid(int(cur_h * shrink), G))

            if not ok:
                return False, None

        return True, placed

    def _normalize_free_space(self, canvas: Canvas, p: Dict[str, Any]):
        # Split free rectangles until each satisfies min short edge and ratio bounds (no count cap)
        grid = int(p["grid"])
        min_short = int(p.get("min_short_edge", 224))
        min_area = int(p["fill_area_min_panel"] or (2 * grid * grid))
        rmin, rmax = float(p["panel_ratio_min"]), float(p["panel_ratio_max"])  # e.g., 0.4 ~ 2.5

        queue: List[Tuple[int, int, int, int]] = list(canvas.free_rects)
        result: List[Tuple[int, int, int, int]] = []
        guard = 0
        while len(queue) > 0 and guard < 5000:
            guard += 1
            areas = np.array([rect_area(R) for R in queue], dtype=np.int64)
            i = int(areas.argmax())
            R = queue.pop(i)
            x0, y0, x1, y1 = R
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue
            if min(w, h) < min_short:
                # discard unusable slivers
                continue
            rR = w / max(1, h)
            if rmin <= rR <= rmax:
                result.append(R)
                continue
            # split along long side to move ratio toward bounds
            horizontal = (w < h)
            split_ok = False
            for _ in range(24):
                pcut = float(np.clip(self.rng.beta(2.0, 2.0), 0.3, 0.7))
                if horizontal:
                    cut = int(align_down_to_grid(h * pcut, grid))
                    if cut <= 0 or cut >= h:
                        continue
                    R1 = (x0, y0, x1, y0 + cut)
                    R2 = (x0, y0 + cut, x1, y1)
                else:
                    cut = int(align_down_to_grid(w * pcut, grid))
                    if cut <= 0 or cut >= w:
                        continue
                    R1 = (x0, y0, x0 + cut, y1)
                    R2 = (x0 + cut, y0, x1, y1)
                ok = True
                for Rt in (R1, R2):
                    aw = Rt[2] - Rt[0]
                    ah = Rt[3] - Rt[1]
                    if aw * ah < min_area or min(aw, ah) < min_short:
                        ok = False
                        break
                if not ok:
                    continue
                queue.append(R1)
                queue.append(R2)
                split_ok = True
                break
            if not split_ok:
                # cannot usefully split further; accept as-is
                result.append(R)
        canvas.free_rects = result

    def _fill_free_panels(self, canvas: Canvas, p: Dict[str, Any]) -> List[Dict[str, Any]]:
        if len(canvas.free_rects) == 0:
            return []
        # dynamic role: fills are sampled from the remaining pool not used as targets in this canvas
        used_src_idx = set(
            ix for ix in (pl.get("_src_index", None) for pl in canvas.placements) if ix is not None
        )
        img_col = p["src_img_col"]
        # also deduplicate by image path already used on this canvas (targets or previously added fills)
        used_img_paths = set(
            str(pl.get("src_img")) for pl in canvas.placements if pl.get("src_img") is not None
        )
        fill_df = self.df.drop(index=list(used_src_idx), errors="ignore").copy()
        # require src_img validity
        fill_df = fill_df[(fill_df[img_col].notna()) & (fill_df[img_col] != "")]
        # exclude any rows whose image path has already been used on this canvas
        if len(used_img_paths) > 0:
            fill_df = fill_df[~fill_df[img_col].isin(used_img_paths)]
        if len(fill_df) == 0:
            return []
        if "area" not in fill_df.columns:
            fill_df["area"] = fill_df[p["src_w_col"]].astype(float) * fill_df[p["src_h_col"]].astype(float)
        if "ratio" not in fill_df.columns:
            fill_df["ratio"] = fill_df[p["src_w_col"]].astype(float) / np.maximum(1.0, fill_df[p["src_h_col"]].astype(float))
        # pre-scale cap by area for fills (planning only)
        cap_area = float(p.get("src_max_area", 1024 * 1024))
        fw_all = fill_df[p["src_w_col"]].astype(float).to_numpy()
        fh_all = fill_df[p["src_h_col"]].astype(float).to_numpy()
        S_raw = np.maximum(1.0, fw_all * fh_all)
        f_pre = np.minimum(1.0, np.sqrt(cap_area / S_raw))
        fill_df["_pre_scale"] = f_pre
        fill_df["_area_eff"] = (S_raw * (f_pre ** 2))
        alpha = float(p["alpha_usage"])
        use_counts = fill_df.get(p["fill_count_col"], pd.Series(0, index=fill_df.index)).fillna(0).to_numpy(dtype=np.float64)
        w_usage = 1.0 / (1.0 + alpha * use_counts)
        S = fill_df["_area_eff"].to_numpy(dtype=np.float64)
        r_src = fill_df["ratio"].to_numpy(dtype=np.float64)
        tau = float(p["tau_fill"])
        lam_min, lam_max = float(p["fill_lambda_min"]), float(p["fill_lambda_max"])
        crop_min_frac = float(p.get("crop_min_frac", 0.5))

        # cooling exclusion
        recent_set = set(self._recent) if p.get("cooldown_enable", True) else set()

        panels = sorted(canvas.free_rects, key=lambda R: -rect_area(R))
        layouts = []
        used_indices = set()
        for R in panels:
            A_R = float(rect_area(R))
            # enforce min short edge for panels
            wR = R[2] - R[0]
            hR = R[3] - R[1]
            if min(wR, hR) < int(p.get("min_short_edge", 224)):
                continue
            rR = wR / max(1.0, hR)
            S_lo = A_R / lam_max
            S_hi = A_R / lam_min
            r_lo = rR * math.exp(-tau)
            r_hi = rR * math.exp(+tau)
            mask = (S >= S_lo) & (S <= S_hi) & (r_src >= r_lo) & (r_src <= r_hi)
            idx_arr = fill_df.index.to_numpy()[mask]
            if p.get("cooldown_enable", True):
                idx_arr = np.array([ix for ix in idx_arr if ix not in recent_set], dtype=int)
            # avoid duplicate image paths within the same canvas
            if len(idx_arr) > 0 and len(used_img_paths) > 0:
                idx_arr = np.array([
                    ix for ix in idx_arr
                    if str(fill_df.at[ix, img_col]) not in used_img_paths
                ], dtype=int)
            if len(idx_arr) == 0:
                # relax ratio
                r_lo = rR * math.exp(-tau * 1.6)
                r_hi = rR * math.exp(+tau * 1.6)
                mask = (S >= S_lo) & (S <= S_hi) & (r_src >= r_lo) & (r_src <= r_hi)
                idx_arr = fill_df.index.to_numpy()[mask]
                if p.get("cooldown_enable", True):
                    idx_arr = np.array([ix for ix in idx_arr if ix not in recent_set], dtype=int)
                if len(idx_arr) > 0 and len(used_img_paths) > 0:
                    idx_arr = np.array([
                        ix for ix in idx_arr
                        if str(fill_df.at[ix, img_col]) not in used_img_paths
                    ], dtype=int)
            if len(idx_arr) == 0:
                # fallback: consider large images with crop + resize
                all_idx_arr = fill_df.index.to_numpy()
                if p.get("cooldown_enable", True):
                    all_idx_arr = np.array([
                        ix for ix in all_idx_arr
                        if ix not in used_indices and ix not in recent_set and str(fill_df.at[ix, img_col]) not in used_img_paths
                    ], dtype=int)
                else:
                    all_idx_arr = np.array([
                        ix for ix in all_idx_arr
                        if ix not in used_indices and str(fill_df.at[ix, img_col]) not in used_img_paths
                    ], dtype=int)
                if len(all_idx_arr) == 0:
                    continue
                # score by usage primarily, then ratio closeness
                cand_w = w_usage[fill_df.index.get_indexer(all_idx_arr)]
                cand_r = r_src[fill_df.index.get_indexer(all_idx_arr)]
                ratio_dist = np.abs(np.log(np.maximum(1e-8, cand_r) / max(1e-8, rR)))
                score = cand_w / (1.0 + 0.8 * ratio_dist)
                # pick top-N and then sample among them to keep randomness
                N = int(min(64, len(all_idx_arr)))
                top_idx = np.argsort(-score)[:N]
                pool = all_idx_arr[top_idx]
                choice = weighted_sample_indices(pool, cand_w[top_idx], 1, self.rng)
                if not choice:
                    continue
                j = choice[0]
                row = fill_df.loc[j]
                src_w = int(row.get(p["src_w_col"], 0) or 0)
                src_h = int(row.get(p["src_h_col"], 0) or 0)
                if src_w <= 0 or src_h <= 0:
                    continue
                # choose a crop area within bounds and >= crop_min_frac of source
                S_src = float(max(1.0, row["area"]))
                A_min = max(A_R / lam_max, crop_min_frac * S_src)
                A_max = min(S_src, A_R / lam_min)
                if A_min > A_max:
                    # cannot obtain sufficient crop; skip
                    continue
                # aim for the largest feasible crop to minimize resizing
                A_c = A_max
                crop_box = center_crop_for_ratio(src_w, src_h, rR)
                # adjust crop to requested area by scaling around center if needed
                cx0, cy0, cx1, cy1 = crop_box
                cur_w = max(1, cx1 - cx0)
                cur_h = max(1, cy1 - cy0)
                # shrink crop proportionally if larger than target crop dims
                if cur_w * cur_h > A_c:
                    scale = math.sqrt(A_c / (cur_w * cur_h))
                    new_w = max(1, int(cur_w * scale))
                    new_h = max(1, int(cur_h * scale))
                    dx = (cur_w - new_w) // 2
                    dy = (cur_h - new_h) // 2
                    cx0 += dx; cy0 += dy; cx1 -= dx; cy1 -= dy
                crop = (int(cx0), int(cy0), int(cx1), int(cy1))
                used_indices.add(j)
                if row.get(img_col, None) is not None:
                    used_img_paths.add(str(row.get(img_col)))
                lam = float(A_R / max(1.0, A_c))
                lam = float(np.clip(lam, lam_min, lam_max))
                layouts.append({
                    "layout_bbox": R,
                    "layout_area": int(A_R),
                    "layout_bbox_ratio": float(A_R / max(1.0, canvas.W * canvas.H)),
                    "src_dataset": row.get("src_dataset", p.get("default_dataset", "default")),
                    "src_idx": int(row.name) if pd.notna(row.name) else None,
                    "src_img": row.get(p["src_img_col"], None),
                    "scale_ratio": lam,
                    "img_crop_region": crop,
                    "object_bbox": row.get(p["bbox_col"], None),
                    "object_bbox_ratio": float(row.get(p["rho_src_col"], 0.0)),
                    "image_type": "fill",
                    "r_out": float(rR),
                    "_src_index": int(j),
                    "_src_S": float(row["area"]),
                    "_src_r": float(row["ratio"]),
                    "_has_bbox": False,
                    "question": row.get("question", None),
                    "answer": row.get("answer", None),
                })
                continue
            idx_arr = np.array([
                ix for ix in idx_arr
                if ix not in used_indices and str(fill_df.at[ix, img_col]) not in used_img_paths
            ], dtype=int)
            if len(idx_arr) == 0:
                continue
            w = w_usage[fill_df.index.get_indexer(idx_arr)]
            choice = weighted_sample_indices(idx_arr, w, 1, self.rng)
            if not choice:
                continue
            j = choice[0]
            used_indices.add(j)
            row = fill_df.loc[j]
            # mark this image path as used on canvas to prevent duplicates
            if row.get(img_col, None) is not None:
                used_img_paths.add(str(row.get(img_col)))
            lam = float(A_R / max(1.0, row["area"]))
            lam = float(np.clip(lam, lam_min, lam_max))

            # optional: ratio-matching center-crop (enhancement D)
            src_w = int(row.get(p["src_w_col"], 0) or 0)
            src_h = int(row.get(p["src_h_col"], 0) or 0)
            crop = None
            if src_w > 0 and src_h > 0:
                crop = center_crop_for_ratio(src_w, src_h, rR)

            layouts.append({
                "layout_bbox": R,
                "layout_area": int(A_R),
                "layout_bbox_ratio": float(A_R / max(1.0, canvas.W * canvas.H)),
                "src_dataset": row.get("src_dataset", p.get("default_dataset", "default")),
                "src_idx": int(row.name) if pd.notna(row.name) else None,
                "src_img": row.get(p["src_img_col"], None),
                "scale_ratio": lam,
                "img_crop_region": crop,
                "object_bbox": row.get(p["bbox_col"], None),
                "object_bbox_ratio": float(row.get(p["rho_src_col"], 0.0)),
                "image_type": "fill",
                "r_out": float(rR),
                "_src_index": int(j),
                "_src_S": float(row["area"]),
                "_src_r": float(row["ratio"]),
                "_has_bbox": False,
                "question": row.get("question", None),
                "answer": row.get("answer", None),
            })
        return layouts

    def _validate_free_rects(self, canvas: Canvas, p: Dict[str, Any]) -> bool:
        if not canvas.free_rects:
            return True
        M = int(p.get("min_short_edge", 224))
        for R in canvas.free_rects:
            w = R[2] - R[0]
            h = R[3] - R[1]
            if w == 0 or h == 0:
                continue
            if min(w, h) < M:
                return False
        return True

    def _bump_usage_counts(self, targets: Optional[List[Dict[str, Any]]], fills: Optional[List[Dict[str, Any]]]):
        if targets:
            for t in targets:
                idx = t.get("_src_index", None)
                typ = t.get("image_type", None)
                if idx is not None and typ in ("core", "alt"):
                    col = "core_count" if typ == "core" else "alt_count"
                    try:
                        self.df.loc[idx, col] = int((self.df.loc[idx, col] or 0)) + 1
                    except Exception:
                        cur = self.df.loc[idx, col]
                        self.df.loc[idx, col] = 1 if pd.isna(cur) else int(cur) + 1
        if fills:
            for f in fills:
                idx = f.get("_src_index", None)
                if idx is not None:
                    col = "fill_count"
                    try:
                        self.df.loc[idx, col] = int((self.df.loc[idx, col] or 0)) + 1
                    except Exception:
                        cur = self.df.loc[idx, col]
                        self.df.loc[idx, col] = 1 if pd.isna(cur) else int(cur) + 1


def summarize_usage_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute usage distributions for core, alt, and fill.
    Returns a dict with:
      - overview: DataFrame with per-role summary stats
      - histograms: dict(role -> DataFrame with columns ['count', 'num_images'])
    """
    roles = {
        "core": "core_count",
        "alt": "alt_count",
        "fill": "fill_count",
    }
    overview_rows: List[Dict[str, Any]] = []
    histograms: Dict[str, pd.DataFrame] = {}
    for role, col in roles.items():
        if col not in df.columns:
            continue
        s = df[col].fillna(0).astype(int)
        total = int(len(s))
        used = int((s > 0).sum())
        unused = int((s == 0).sum())
        row = {
            "role": role,
            "total_images": total,
            "used_images": used,
            "unused_images": unused,
            "sum_count": int(s.sum()),
            "mean": float(s.mean()) if total > 0 else 0.0,
            "median": float(s.median()) if total > 0 else 0.0,
            "min": int(s.min()) if total > 0 else 0,
            "max": int(s.max()) if total > 0 else 0,
            "p90": float(s.quantile(0.90)) if total > 0 else 0.0,
            "p99": float(s.quantile(0.99)) if total > 0 else 0.0,
        }
        overview_rows.append(row)
        hist_df = (
            s.value_counts()
            .sort_index()
            .rename_axis("count")
            .reset_index(name="num_images")
        )
        histograms[role] = hist_df
    overview_cols = [
        "role", "total_images", "used_images", "unused_images",
        "sum_count", "mean", "median", "min", "max", "p90", "p99",
    ]
    overview_df = pd.DataFrame(overview_rows, columns=overview_cols)
    print(overview_df)
    for role in ("core", "alt", "fill"):
        hist = histograms.get(role)
        if hist is not None:
            print(f"hist {role} (head):")
            print(hist.head())
    return {"overview": overview_df, "histograms": histograms}


def _scan_existing_indices(save_dir: Optional[str]) -> Tuple[int, int]:
    """
    Scan save_dir/images/** for existing JPG files named as integer indices.
    Returns (max_index_found, num_images_found). If none found, returns (-1, 0).
    """
    if not save_dir:
        return -1, 0
    images_root = os.path.join(str(save_dir), "images")
    if not os.path.isdir(images_root):
        return -1, 0
    max_idx = -1
    count = 0
    for root, dirs, files in os.walk(images_root):
        for fn in files:
            if not fn.lower().endswith(".jpg"):
                continue
            name = os.path.splitext(fn)[0]
            try:
                idx = int(name)
                count += 1
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                continue
    return max_idx, count


def _find_latest_counts_csv(save_dir: Optional[str]) -> Optional[str]:
    """
    Heuristic to find the latest counts CSV to resume from within save_dir.
    Priority:
      1) final_counts.csv
      2) counts_latest.csv
      3) checkpoints/*.csv with newest mtime
    """
    if not save_dir:
        return None
    root = str(save_dir)
    cand = os.path.join(root, "final_counts.csv")
    if os.path.isfile(cand):
        return cand
    cand = os.path.join(root, "counts_latest.csv")
    if os.path.isfile(cand):
        return cand
    ckpt_dir = os.path.join(root, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    latest_path = None
    latest_mtime = -1.0
    try:
        for fn in os.listdir(ckpt_dir):
            if not fn.lower().endswith(".csv"):
                continue
            p = os.path.join(ckpt_dir, fn)
            try:
                mt = os.path.getmtime(p)
            except Exception:
                continue
            if mt > latest_mtime:
                latest_mtime = mt
                latest_path = p
    except Exception:
        return None
    return latest_path


def _merge_counts(base_df: pd.DataFrame, counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge counts from counts_df into base_df, keyed strictly by 'uid'.
    Overwrites/creates core/alt/fill counts in base_df.
    """
    if counts_df is None or len(counts_df) == 0:
        return base_df
    if "uid" not in base_df.columns or "uid" not in counts_df.columns:
        return base_df
    right_cols = [c for c in ("core_count", "alt_count", "fill_count") if c in counts_df.columns]
    if not right_cols:
        return base_df
    right = counts_df[["uid"] + right_cols].copy()
    # ensure dtype consistency
    try:
        right["uid"] = right["uid"].astype(str)
    except Exception:
        pass
    base_df = base_df.copy()
    try:
        base_df["uid"] = base_df["uid"].astype(str)
    except Exception:
        pass
    merged = base_df.merge(right, on="uid", how="left", suffixes=("", "_resume"))
    for c in ("core_count", "alt_count", "fill_count"):
        rc = c + "_resume"
        if rc in merged.columns:
            merged[c] = merged[rc].where(merged[rc].notna(), merged.get(c, 0))
            merged.drop(columns=[rc], inplace=True)
        else:
            if c not in merged.columns:
                merged[c] = 0
    return merged


def _save_checkpoint(cur_df: pd.DataFrame, save_dir: Optional[str], produced_total: int, next_index: int):
    """
    Save rolling checkpoints: counts_latest.csv, checkpoints/counts_round_{produced_total}.csv,
    and resume_state.json with basic state (produced_total, next_index).
    """
    if not save_dir:
        return
    try:
        os.makedirs(str(save_dir), exist_ok=True)
        ckpt_dir = os.path.join(str(save_dir), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        latest_csv = os.path.join(str(save_dir), "counts_latest.csv")
        round_csv = os.path.join(ckpt_dir, f"counts_round_{produced_total}.csv")
        # save aggregated counts by uid only
        cols = [c for c in ("core_count", "alt_count", "fill_count") if c in cur_df.columns]
        if "uid" in cur_df.columns and cols:
            counts_df = cur_df[["uid"] + cols].copy()
            try:
                counts_df["uid"] = counts_df["uid"].astype(str)
            except Exception:
                pass
            for c in ("core_count", "alt_count", "fill_count"):
                if c not in counts_df.columns:
                    counts_df[c] = 0
            counts_df = counts_df.groupby("uid", as_index=False)[["core_count", "alt_count", "fill_count"]].sum()
            counts_df.to_csv(latest_csv, index=False)
            counts_df.to_csv(round_csv, index=False)
        else:
            cur_df.to_csv(latest_csv, index=False)
            cur_df.to_csv(round_csv, index=False)
        # state JSON
        import json as _json
        state_path = os.path.join(str(save_dir), "resume_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            _json.dump({"produced_total": int(produced_total), "next_index": int(next_index)}, f)
    except Exception as _e:
        print("[checkpoint] failed:", _e)


def _reconstruct_counts_from_json(save_dir: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Fallback: reconstruct core/alt/fill counts by scanning saved JSONs in save_dir/images/**.
    Returns a DataFrame with columns ['image','core_count','alt_count','fill_count'] or None if no jsons.
    """
    if not save_dir:
        return None
    images_root = os.path.join(str(save_dir), "images")
    if not os.path.isdir(images_root):
        return None
    import json as _json
    counts: Dict[str, Dict[str, int]] = {}
    for root, _, files in os.walk(images_root):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = _json.load(f)
            except Exception:
                continue
            layouts = data.get("layouts") or data.get("placements") or []
            for it in layouts:
                img_path = it.get("src_img") or it.get("image")
                typ = str(it.get("image_type", "")).lower()
                if not img_path or typ not in ("core", "alt", "fill"):
                    continue
                d = counts.setdefault(str(img_path), {"core_count": 0, "alt_count": 0, "fill_count": 0})
                if typ == "core":
                    d["core_count"] += 1
                elif typ == "alt":
                    d["alt_count"] += 1
                elif typ == "fill":
                    d["fill_count"] += 1
    if not counts:
        return None
    rows = []
    for k, v in counts.items():
        rows.append({"image": k, **v})
    return pd.DataFrame(rows, columns=["image", "core_count", "alt_count", "fill_count"])


def prepare_resume(df: pd.DataFrame, save_dir: Optional[str], resume_counts_path: Optional[str] = None) -> Tuple[pd.DataFrame, int]:
    """
    External resume preparation usable by both single- and multi-process flows.
    - Merge prior counts from provided path or discover latest in save_dir
    - Scan existing images to compute next index to avoid collisions
    Returns (df_with_counts_merged, next_index)
    """
    cur_df = df.copy()
    # merge counts
    counts_path = resume_counts_path or _find_latest_counts_csv(save_dir)
    loaded_counts = False
    if counts_path and os.path.isfile(counts_path):
        try:
            resume_df = pd.read_csv(counts_path)
            # harmonize uid column: if missing, try map from image->uid using current df
            if "uid" not in resume_df.columns:
                if "image" in resume_df.columns and "image" in cur_df.columns and "uid" in cur_df.columns:
                    mapper = cur_df[["image", "uid"]].drop_duplicates()
                    resume_df = resume_df.merge(mapper, on="image", how="left")
                # after mapping, keep only rows with uid
                if "uid" in resume_df.columns:
                    # aggregate by uid in case multiple rows map to same uid
                    keep_cols = [c for c in ("core_count", "alt_count", "fill_count") if c in resume_df.columns]
                    if keep_cols:
                        agg_df = resume_df[["uid"] + keep_cols].dropna(subset=["uid"]).copy()
                        try:
                            agg_df["uid"] = agg_df["uid"].astype(str)
                        except Exception:
                            pass
                        resume_df = agg_df.groupby("uid", as_index=False)[keep_cols].sum()
            # ensure uid dtype
            if "uid" in cur_df.columns:
                try:
                    cur_df["uid"] = cur_df["uid"].astype(str)
                except Exception:
                    pass
            if "uid" in resume_df.columns:
                try:
                    resume_df["uid"] = resume_df["uid"].astype(str)
                except Exception:
                    pass
            cur_df = _merge_counts(cur_df, resume_df)
            loaded_counts = True
            print(f"[resume] loaded counts from {counts_path}")
        except Exception as _e:
            print("[resume] failed to load counts:", _e)
    # Fallback: reconstruct from jsons if no counts were loaded or totals are zero
    try:
        totals = sum(int(cur_df.get(c, 0).fillna(0).sum()) for c in ("core_count", "alt_count", "fill_count") if c in cur_df.columns)
    except Exception:
        totals = 0
    if (not loaded_counts or totals == 0) and save_dir:
        recon = _reconstruct_counts_from_json(save_dir)
        if recon is not None and len(recon) > 0:
            cur_df = _merge_counts(cur_df, recon)
            print(f"[resume] reconstructed counts from jsons: {len(recon)} sources")
    # scan images for next index
    max_idx, num_found = _scan_existing_indices(save_dir)
    next_index = (max_idx + 1) if max_idx >= 0 else 0
    print(f"[resume] found {num_found} existing images, next index {next_index}")
    return cur_df, next_index


def generate_collage(df: pd.DataFrame, n: int, seed: int = 42, save_dir: Optional[str] = None, start_idx: int = 0, show_progress: bool = True, **kwargs):
    generator = CollageGenerator(df, rng=seed)
    results = []
    params = dict(kwargs)
    params["save_path"] = save_dir
    _iter = range(n)
    _iter = tqdm(_iter) if show_progress else _iter
    for i in _iter:
        index = start_idx + i
        out = generator.generate(index, **params)
        if out is not None:
            results.append(out)
    return results


def _collage_worker(args):
    sub_df, num, proc_seed, proc_save_dir, proc_start_idx, wkwargs = args
    outs = generate_collage(
        sub_df,
        n=num,
        seed=int(proc_seed),
        save_dir=proc_save_dir,
        start_idx=int(proc_start_idx),
        show_progress=False,
        **wkwargs,
    )
    produced = len(outs)
    return sub_df, produced


def generate_collage_multi_process(
    df: pd.DataFrame,
    total_count: int = 100,
    per_round_per_proc: int = 5,
    n_proc: int = 4,
    seed: int = 42,
    save_dir: Optional[str] = None,
    checkpoint_each_round: bool = True,
    auto_resume: bool = True,
    **kwargs,
):
    """
    Multi-round, multi-process generation driven by total target samples and per-round per-process quota.
    Args:
      - total_count: total number of collages to generate overall
      - per_round_per_proc: in each round, each process generates this many collages
      - n_proc: number of worker processes per round
    Procedure per round:
      1) Shuffle and balance-split the DataFrame into n_proc chunks (prefer unseen cores; lower alt/fill counts).
      2) Each process generates per_round_per_proc (or fewer on the last round) and returns its updated df chunk.
      3) Aggregate, reshuffle, and continue until total_count is reached.
    Returns the final mixed DataFrame with updated usage counts.
    """
    from multiprocessing import Pool

    rng = np.random.default_rng(seed)

    # global index base for file naming (caller can pass via start_idx)
    global_start_idx = int(kwargs.pop("start_idx", 0) or 0)
    # determine remaining target and optionally adjust start index by scanning existing outputs
    remaining_target = int(total_count)
    if auto_resume and save_dir:
        try:
            max_idx0, num_found0 = _scan_existing_indices(save_dir)
            if max_idx0 >= 0:
                global_start_idx = max(global_start_idx, max_idx0 + 1)
            remaining_target = max(0, int(total_count) - int(num_found0))
            if num_found0 > 0:
                print(f"[auto_resume] found {num_found0} existing images, next start_idx {global_start_idx}, remaining target {remaining_target}")
        except Exception as _e:
            print("[auto_resume] scan failed:", _e)

    def balance_and_split(cur_df: pd.DataFrame) -> List[pd.DataFrame]:
        # Two modes: 'weighted' fast split by weights, or 'lexi' greedy (core>alt>fill)
        balance_mode = kwargs.get("balance_mode", "weighted")
        core_used = (cur_df.get("core_count", 0).fillna(0) > 0).astype(int)
        alt_used = (cur_df.get("alt_count", 0).fillna(0) > 0).astype(int)
        fill_used = (cur_df.get("fill_count", 0).fillna(0) > 0).astype(int)
        if balance_mode == "weighted":
            # priority weights: core >> alt >> fill, then slight preference for lower total usage
            total_usage = (
                cur_df.get("core_count", 0).fillna(0).astype(int)
                + cur_df.get("alt_count", 0).fillna(0).astype(int)
                + cur_df.get("fill_count", 0).fillna(0).astype(int)
            )
            w = (
                core_used.astype(np.int64) * 10_000_000
                + alt_used.astype(np.int64) * 10_000
                + fill_used.astype(np.int64) * 10
                - total_usage.astype(np.int64)
            )
            order = np.argsort(-w.to_numpy())  # descending by weight
            cur_bal = cur_df.iloc[order].reset_index(drop=True)
            # round-robin split to distribute heavy rows evenly
            return [cur_bal.iloc[i::n_proc].copy() for i in range(n_proc)]
        else:
            # Greedy lexicographic balancing (core > alt > fill)
            V = np.stack([core_used.to_numpy(), alt_used.to_numpy(), fill_used.to_numpy()], axis=1).astype(np.int64)
            order = np.argsort(-V.sum(axis=1))
            total = V.sum(axis=0)
            avg = total / max(1, n_proc)
            target_size = len(cur_df) / max(1, n_proc)
            sums = np.zeros((n_proc, 3), dtype=np.float64)
            sizes = np.zeros(n_proc, dtype=np.int64)
            indices_per_chunk = [[] for _ in range(n_proc)]
            size_penalty = 0.05
            for idx in order:
                v = V[idx]
                best_k = 0
                best_tuple = None
                for k in range(n_proc):
                    s_new = sums[k] + v
                    size_new = sizes[k] + 1
                    d_core = abs(s_new[0] - avg[0])
                    d_alt = abs(s_new[1] - avg[1])
                    d_fill = abs(s_new[2] - avg[2])
                    d_size = abs(size_new - target_size) * size_penalty
                    tup = (d_core, d_alt, d_fill, d_size)
                    if best_tuple is None or tup < best_tuple:
                        best_tuple = tup
                        best_k = k
                sums[best_k] += v
                sizes[best_k] += 1
                indices_per_chunk[best_k].append(idx)
            chunks = []
            for k in range(n_proc):
                idxs = indices_per_chunk[k]
                idxs.sort()
                chunks.append(cur_df.iloc[idxs].copy())
            return chunks

    cur_df = df.copy()
    processed_assigned = 0
    produced_total = 0
    pbar = tqdm(total=remaining_target, disable=(n_proc <= 1))
    # create a single persistent pool reused across rounds
    pool = None
    if n_proc > 1:
        pool = Pool(processes=n_proc)
    # compute number of rounds based on remaining work
    while produced_total < remaining_target:
        # shuffle
        cur_df = cur_df.sample(frac=1.0, random_state=int(rng.integers(0, 1 << 30))).reset_index(drop=True)
        # split
        chunks = balance_and_split(cur_df)
        # how many remain to produce (respect remaining_target when auto-resuming)
        remain = remaining_target - produced_total
        # target this round total: up to n_proc * per_round_per_proc
        this_round_total = min(remain, n_proc * per_round_per_proc)
        # per-process counts for this round (fill procs left-to-right)
        per = [0] * n_proc
        for i in range(this_round_total):
            per[i % n_proc] += 1
        # compute start offsets to avoid filename collisions across procs/rounds
        round_base = global_start_idx + processed_assigned
        offsets = []
        acc = 0
        for i in range(n_proc):
            offsets.append(round_base + acc)
            acc += per[i]

        args_list = []
        for i in range(n_proc):
            args_list.append(
                (
                    chunks[i],
                    per[i],
                    int(rng.integers(0, 1 << 30)),
                    save_dir,
                    offsets[i],
                    kwargs,
                )
            )
        if pool is not None:
            subs = pool.map(_collage_worker, args_list)
        else:
            subs = [_collage_worker(args_list[0])]
        # aggregate
        sub_dfs = [sd for (sd, _cnt) in subs]
        produced_this_round = int(sum(_cnt for (_sd, _cnt) in subs))
        cur_df = pd.concat(sub_dfs, ignore_index=True)
        # reshuffle to mix for next round
        cur_df = cur_df.sample(frac=1.0, random_state=int(rng.integers(0, 1 << 30))).reset_index(drop=True)
        produced_total += produced_this_round
        processed_assigned += this_round_total
        if pbar is not None:
            pbar.update(produced_this_round)
        # periodic checkpoint
        if checkpoint_each_round and save_dir:
            _save_checkpoint(cur_df, save_dir, produced_total, global_start_idx + processed_assigned)
    if pbar is not None:
        pbar.close()
    # ensure pool is properly closed after all rounds
    if pool is not None:
        pool.close()
        pool.join()
    summarize_usage_distribution(cur_df)

    # recount and summarize before saving final df
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_csv = os.path.join(str(save_dir), "final_counts.csv")
        cur_df.to_csv(out_csv, index=False)
    return cur_df

def load_and_filter_csvs(csv_files, required_columns=None):
    """
    Load CSV files and filter to keep only specified columns.
    
    Args:
        csv_files: str or list of str - path(s) to CSV file(s)
        required_columns: list of str - columns to keep. Default: 
            ['image', 'question',  'answer', 'img_width', 'img_height', 'bbox', 'bbox_ratio']
    
    Returns:
        pd.DataFrame with only the required columns
    """
    if required_columns is None:
        required_columns = [
            'image', 'question', 'answer',
            'img_width', 'img_height', 'bbox', 'bbox_ratio'
        ]

    # Convert single file to list
    if isinstance(csv_files, str):
        csv_files = [csv_files]

    dfs = []
    for csv_file in csv_files:
        print(f"Loading {csv_file}...")
        df = pd.read_csv(csv_file)

        # Keep only required columns that exist in the dataframe
        existing_columns = [col for col in required_columns if col in df.columns]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns in {csv_file}: {missing_columns}")

        df_filtered = df[existing_columns].copy()
        dfs.append(df_filtered)

    # Concatenate all dataframes
    if len(dfs) == 1:
        result_df = dfs[0]
    else:
        result_df = pd.concat(dfs, ignore_index=True)

    for col in ("core_count", "alt_count", "fill_count"):
        if col not in result_df.columns:
            result_df[col] = 0
    # ensure stable uid column
    result_df["uid"] = result_df.index.astype(str)
    print(f"Combined dataframe shape: {result_df.shape}")
    print(f"Columns: {list(result_df.columns)}")

    return result_df


if __name__ == "__main__":
    # Input CSV format — each row is one source image:
    #   image       (required) absolute path to the image file
    #   img_width   (required) pixel width
    #   img_height  (required) pixel height
    #   bbox        (required) "[x0, y0, x1, y1]" object bounding box in pixels
    #   bbox_ratio  (required) bbox area / image area
    #   question    (required) QA question, passed through to output JSON
    #   answer      (required) QA answer, passed through to output JSON
    #
    # Example rows:
    #   image,img_width,img_height,bbox,bbox_ratio,question,answer
    #   /data/img/car.jpg,1024,768,"[120,80,400,350]",0.093,Where is the red car?,In the lot
    #   /data/img/tree.jpg,800,600,"[100,50,700,550]",0.875,What kind of tree?,Oak tree
    #   /data/img/dog.jpg,640,480,"[50,30,300,400]",0.284,What breed is the dog?,Golden retriever
    
    csv_files = [
     "/scratch/ywxzml3j/yaolewei/data/vsearch_collage/kc_filtered_resized_ann.csv",
     "/scratch/ywxzml3j/yaolewei/data/vsearch_collage/viscot_natural_image_filtered_ann.csv",
     "/scratch/ywxzml3j/yaolewei/data/vsearch_collage/vstar_attr_filtered_ann.csv",
     "/scratch/ywxzml3j/yaolewei/data/vsearch_collage/vstar_spatial_filtered_ann.csv"
    ]

    df = load_and_filter_csvs(csv_files,
                              required_columns=["image", "question", "answer", "img_width", "img_height", "bbox", "bbox_ratio"])

    num_of_samples = 24000
    save_dir = "/scratch/ywxzml3j/likaican/viscot+vstar_target_1_24k"
    seed = 42
    params = {
        "draw_edges": True,
        "print_layout": False,
        "plot_bbox": False,
        "edge_by_type": False,
        "edge_color": (0, 0, 0),
        "show_label": True,
        "label_color": (0, 0, 0),
        "K_min": 4, "K_max": 6,
        "core_min": 1, "core_max": 1,
        "max_attempts": 10
    }

    print("Generating collages, saving to ", save_dir)
    df, next_idx = prepare_resume(df, save_dir)

    generate_collage_multi_process(
        df,
        num_of_samples,
        per_round_per_proc=16,
        n_proc=64,
        save_dir=save_dir,
        seed=seed,
        checkpoint_each_round=True,
        auto_resume=True,
        start_idx=next_idx,
        **params,
    )


