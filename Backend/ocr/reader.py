from paddleocr import PaddleOCR
import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import re

# Initialize OCR once (very important)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

# Poppler path (Windows)
POPPLER_PATH = r"C:\Users\Prabh Singh\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"


# ---------------------------
# Preprocessing (improves accuracy)
# ---------------------------
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    return processed


# ---------------------------
# Convert PDF to image
# ---------------------------
def pdf_to_image(pdf_path):
    pages = convert_from_path(
        pdf_path,
        dpi=300,                 # VERY IMPORTANT for correct scaling
        poppler_path=POPPLER_PATH
    )

    # First page only
    page = pages[0]

    img = np.array(page)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


# ---------------------------
# Group detection results by visual line (same y), then merge adjacent boxes that are part of the same word
# ---------------------------
def _bbox_rect(bbox):
    """Return (left, top, right, bottom) from 4-point bbox."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs), min(ys), max(xs), max(ys))


def _group_by_line(items, line_tolerance_ratio=0.6):
    """Group (bbox, text, conf) items by line. Same line if vertical overlap / height is within tolerance."""
    if not items:
        return []
    # Sort by center-y then center-x
    def center_y(bbox):
        r = _bbox_rect(bbox)
        return (r[1] + r[3]) / 2

    items_sorted = sorted(items, key=lambda x: (center_y(x[0]), _bbox_rect(x[0])[0]))
    lines = []
    current_line = [items_sorted[0]]
    _, ty1, _, ty2 = _bbox_rect(items_sorted[0][0])
    line_height = ty2 - ty1

    for i in range(1, len(items_sorted)):
        bbox, text, conf = items_sorted[i]
        left, top, right, bottom = _bbox_rect(bbox)
        cy = (top + bottom) / 2
        # Same line if vertical center is within tolerance of current line's vertical span
        if abs(cy - (ty1 + ty2) / 2) <= line_tolerance_ratio * max(line_height, bottom - top):
            current_line.append((bbox, text, conf))
            ty1 = min(ty1, top)
            ty2 = max(ty2, bottom)
            line_height = ty2 - ty1
        else:
            lines.append(current_line)
            current_line = [(bbox, text, conf)]
            ty1, ty2 = top, bottom
            line_height = ty2 - ty1
    if current_line:
        lines.append(current_line)
    return lines


def _merge_adjacent_fragments(line_items, gap_ratio=0.75, max_gap_px=18):
    """
    Merge consecutive fragments on the same line when the gap suggests same word
    (e.g. "direc" + "tion" -> "direction"). Merge when:
    - gap is tiny (<= max_gap_px), or
    - gap is small relative to box size AND both fragments are alphabetic (no space between words).
    """
    if not line_items:
        return []
    merged = []
    for bbox, text, conf in line_items:
        left, top, right, bottom = _bbox_rect(bbox)
        w = right - left
        if not merged:
            merged.append((left, top, right, bottom, text, conf))
            continue
        prev_left, prev_top, prev_right, prev_bottom, prev_text, prev_conf = merged[-1]
        prev_w = prev_right - prev_left
        gap = left - prev_right
        tiny_gap = gap <= max_gap_px
        small_relative = gap <= gap_ratio * min(w, prev_w)
        looks_like_one_word = (
            prev_text and text
            and prev_text[-1].isalpha() and text[0].isalpha()
        )
        if tiny_gap or (small_relative and looks_like_one_word):
            merged[-1] = (
                prev_left, min(prev_top, top), right, max(prev_bottom, bottom),
                prev_text + text, (prev_conf + conf) / 2
            )
        else:
            merged.append((left, top, right, bottom, text, conf))
    return merged


def _merge_same_word_second_pass(merged_line_items):
    """
    Second pass: merge any consecutive (left, top, right, bottom, text, conf) where
    combined text has no space and is one word (letters + optional trailing punct).
    Catches cases where gap was just over threshold (e.g. "direc" + "tion").
    """
    if len(merged_line_items) <= 1:
        return merged_line_items
    out = [merged_line_items[0]]
    for i in range(1, len(merged_line_items)):
        left, top, right, bottom, text, conf = merged_line_items[i]
        prev_left, prev_top, prev_right, prev_bottom, prev_text, prev_conf = out[-1]
        combined = prev_text + text
        # One word: no space, and (all letters) or (letters + trailing punctuation only)
        stripped = combined.rstrip(".,;:!?\"'")
        if " " not in combined and stripped.isalpha():
            out[-1] = (
                prev_left, min(prev_top, top), right, max(prev_bottom, bottom),
                combined, (prev_conf + conf) / 2
            )
        else:
            out.append((left, top, right, bottom, text, conf))
    return out


# ---------------------------
# Merge tokens that should be one word (e.g. "C" + "++" -> "C++", "Node" + ".js" -> "Node.js")
# ---------------------------
def _should_merge_with_previous(prev_word: str, curr_word: str) -> bool:
    if not prev_word or not curr_word:
        return False
    if re.match(r"^[^\w]+$", curr_word):
        return True
    if curr_word.startswith(".") and len(curr_word) > 1:
        return True
    if prev_word.endswith(".") and curr_word.isalpha():
        return True
    return False


def _get_logical_word_spans(text: str):
    """Split text by spaces, then merge symbol-only tokens with previous. Returns list of (start_idx, end_idx, word)."""
    tokens = text.split()
    if not tokens:
        return []
    char_idx = 0
    spans = []
    for token in tokens:
        start = text.find(token, char_idx)
        if start < 0:
            break
        end = start + len(token)
        char_idx = end
        spans.append((start, end, token))
    merged = []
    for start, end, word in spans:
        if merged and _should_merge_with_previous(merged[-1][2], word):
            prev_start, prev_end, prev_word = merged.pop()
            merged.append((prev_start, end, prev_word + word))
        else:
            merged.append((start, end, word))
    return merged


# ---------------------------
# Main OCR Function (mode: "words" = word-by-word boxes, "lines" = one box per line)
# ---------------------------
def extract_text_boxes(file_path, mode="words"):

    ext = os.path.splitext(file_path)[1].lower()

    # Load correct image
    if ext == ".pdf":
        image = pdf_to_image(file_path)
    else:
        image = cv2.imread(file_path)

    if image is None:
        return {
            "boxes": [],
            "width": 0,
            "height": 0
        }

    original_height, original_width = image.shape[:2]

    processed = preprocess(image)

    results = ocr.ocr(processed)

    boxes = []

    if results:
        if mode == "lines":
            # One box per PaddleOCR result (line-level)
            for line in results[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                if confidence <= 0.6 or not text:
                    continue
                left, top, right, bottom = _bbox_rect(bbox)
                w = int(right - left)
                h = int(bottom - top)
                if w <= 0 or h <= 0:
                    continue
                boxes.append({
                    "text": text.strip(),
                    "x": int(left),
                    "y": int(top),
                    "width": w,
                    "height": h
                })
        else:
            # mode == "words": word-by-word with grouping and merging
            items = []
            for line in results[0]:
                bbox, text, confidence = line[0], line[1][0], line[1][1]
                if confidence <= 0.6 or not text:
                    continue
                items.append((bbox, text.strip(), confidence))

            line_groups = _group_by_line(items)
            for line_items in line_groups:
                merged = _merge_adjacent_fragments(line_items)
                merged = _merge_same_word_second_pass(merged)
                for left, top, right, bottom, text, conf in merged:
                    w = int(right - left)
                    h = int(bottom - top)
                    if w <= 0 or h <= 0:
                        continue
                    word_spans = _get_logical_word_spans(text)
                    if len(word_spans) <= 1:
                        boxes.append({
                            "text": text,
                            "x": int(left),
                            "y": int(top),
                            "width": w,
                            "height": h
                        })
                    else:
                        text_len = len(text)
                        for start_idx, end_idx, word in word_spans:
                            frac_start = start_idx / text_len
                            frac_end = end_idx / text_len
                            x = int(left + w * frac_start)
                            ww = int(w * (frac_end - frac_start))
                            if ww <= 0:
                                continue
                            boxes.append({
                                "text": word,
                                "x": x,
                                "y": int(top),
                                "width": ww,
                                "height": h
                            })

    return {
        "boxes": boxes,
        "width": original_width,
        "height": original_height
    }




# def extract_text_style(image, box):
#     x = box["x"]
#     y = box["y"]
#     w = box["width"]
#     h = box["height"]

#     # Crop word area
#     word_crop = image[y:y+h, x:x+w]

#     # Font size estimate
#     font_scale = h / 30.0

#     # Estimate thickness
#     thickness = max(1, int(h / 25))

#     # Estimate text color (dark pixels only)
#     gray = cv2.cvtColor(word_crop, cv2.COLOR_BGR2GRAY)
#     mean_color = cv2.mean(word_crop)
#     text_color = (
#         int(mean_color[0]),
#         int(mean_color[1]),
#         int(mean_color[2])
#     )


#     text_pixels = cv2.bitwise_and(word_crop, word_crop, mask=mask)
#     mean_color = cv2.mean(text_pixels, mask=mask)

#     text_color = (
#         int(mean_color[0]),
#         int(mean_color[1]),
#         int(mean_color[2])
#     )

#     return {
#         "font_scale": font_scale,
#         "thickness": thickness,
#         "color": text_color
#     }


def detect_text_in_box(image, box):
    """Run OCR on the selected box only; returns the complete detected text in that region. Uses both raw and preprocessed crop for best accuracy."""
    x = int(box["x"])
    y = int(box["y"])
    w = int(box["width"])
    h = int(box["height"])
    if w <= 0 or h <= 0:
        return ""

    pad = max(8, min(w, h) // 4)
    y1 = max(0, y - pad)
    x1 = max(0, x - pad)
    y2 = min(image.shape[0], y + h + pad)
    x2 = min(image.shape[1], x + w + pad)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    def _run_and_join(img_crop):
        res = ocr.ocr(img_crop)
        if not res or not res[0]:
            return "", 0.0
        parts = []
        total_conf = 0.0
        n = 0
        for line in res[0]:
            text = (line[1][0] or "").strip()
            if text:
                parts.append(text)
                total_conf += line[1][1]
                n += 1
        avg_conf = total_conf / n if n else 0
        return " ".join(parts).strip(), avg_conf

    text_proc, conf_proc = _run_and_join(preprocess(crop))
    text_raw, conf_raw = _run_and_join(crop)
    # Prefer result with higher confidence; if similar, prefer longer (more complete)
    if conf_raw > conf_proc and text_raw:
        return text_raw
    if text_proc:
        return text_proc
    return text_raw if text_raw else ""


def remove_text(image, box):
    """Remove only pixels that look like text (dark), so icons and graphics in the box are preserved."""
    x = int(box["x"])
    y = int(box["y"])
    w = int(box["width"])
    h = int(box["height"])
    if w <= 0 or h <= 0:
        return image

    y1, y2 = max(0, y), min(image.shape[0], y + h)
    x1, x2 = max(0, x), min(image.shape[1], x + w)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Treat dark pixels as text (gray < 220 catches black and dark blue/grey text); icons stay
    _, text_mask_crop = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    # Also mask dark-colored pixels (e.g. blue links) that may be mid-gray
    b, g, r = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    dark = (b < 220) & (g < 220) & (r < 220)
    text_mask_crop = np.maximum(text_mask_crop, (dark.astype(np.uint8) * 255))
    kernel = np.ones((2, 2), np.uint8)
    text_mask_crop = cv2.dilate(text_mask_crop, kernel)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = text_mask_crop
    cleaned = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return cleaned

def extract_text_style(image, box):
    """Extract font scale, thickness, and color from the actual text pixels in the box so replacement matches design, weight, and height."""
    x = int(box["x"])
    y = int(box["y"])
    w = int(box["width"])
    h = int(box["height"])

    if w <= 0 or h <= 0:
        return {"font_scale": 1, "thickness": 1, "color": (0, 0, 0), "content_height": h}

    word_crop = image[y : y + h, x : x + w]
    if word_crop is None or word_crop.size == 0:
        return {"font_scale": 1, "thickness": 1, "color": (0, 0, 0), "content_height": h}

    gray = cv2.cvtColor(word_crop, cv2.COLOR_BGR2GRAY)
    _, thresh_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    text_region = cv2.bitwise_and(word_crop, word_crop, mask=thresh_mask)
    mean_color = cv2.mean(text_region, mask=thresh_mask)
    b, g, r = int(mean_color[0]), int(mean_color[1]), int(mean_color[2])
    if (b, g, r) == (0, 0, 0) or (b > 240 and g > 240 and r > 240):
        b = int(np.median(word_crop[:, :, 0]))
        g = int(np.median(word_crop[:, :, 1]))
        r = int(np.median(word_crop[:, :, 2]))
    if (b, g, r) == (0, 0, 0):
        b, g, r = 40, 40, 40
    text_color = (b, g, r)

    # Use actual text pixel bounds for height (not full box) so we match real letter height
    text_pixels = np.where(thresh_mask > 0)
    if len(text_pixels[0]) > 0 and len(text_pixels[1]) > 0:
        content_top = int(np.min(text_pixels[0]))
        content_bottom = int(np.max(text_pixels[0]))
        content_left = int(np.min(text_pixels[1]))
        content_right = int(np.max(text_pixels[1]))
        content_height = max(1, content_bottom - content_top + 1)
        content_width = max(1, content_right - content_left + 1)
        content_area = content_height * content_width
        fill_ratio = (thresh_mask[content_top : content_bottom + 1, content_left : content_right + 1].sum() / 255.0) / max(1, content_area)
    else:
        content_height = h
        fill_ratio = 0.2

    # OpenCV simplex ~21px height at scale 1; scale so our content height matches
    font_scale = max(0.35, content_height / 22.0)
    # Base thickness from content height; boost if text looks bold (high fill)
    base_thickness = max(1, min(3, int(content_height / 18)))
    if fill_ratio > 0.35:
        base_thickness = min(3, base_thickness + 1)
    thickness = base_thickness

    return {
        "font_scale": font_scale,
        "thickness": thickness,
        "color": text_color,
        "content_height": content_height,
    }


def replace_text(image, box, new_text, style):
    """Draw new_text in the box using the same apparent font size, weight, and color as the original."""
    x = box["x"]
    y = box["y"]
    w = box["width"]
    h = box["height"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = style["font_scale"]
    thickness = style["thickness"]

    (text_width, text_height), baseline = cv2.getTextSize(
        new_text, font, font_scale, thickness
    )

    # Only shrink if new text would overflow the box width (keep same style otherwise)
    if text_width > w and w > 0:
        font_scale = font_scale * (w / max(1, text_width))
        font_scale = max(0.25, font_scale)
        (text_width, text_height), baseline = cv2.getTextSize(
            new_text, font, font_scale, thickness
        )

    # Vertical center: putText uses baseline (bottom-left), so baseline_y = y + (h + text_height) / 2
    text_y = y + int((h + text_height) / 2)
    text_x = x

    cv2.putText(
        image,
        new_text,
        (text_x, text_y),
        font,
        font_scale,
        style["color"],
        thickness,
        cv2.LINE_AA,
    )
    return image




