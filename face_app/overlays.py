"""Lightweight Canvas annotations; never run inference or modify camera pixels."""
from __future__ import annotations

from tkinter import font as tkfont


MAX_OVERLAY_AGE = 0.5  # Hide old positions while keeping the live preview moving.


def project_bounds(bounds, frame_size, image_rect):
    """Map camera pixels into the displayed image, including letterbox offsets."""
    left, top, right, bottom = bounds
    source_width, source_height = frame_size
    x, y, width, height = image_rect
    return (x + left * width / source_width, y + top * height / source_height,
            x + right * width / source_width, y + bottom * height / source_height)


class FaceOverlay:
    def __init__(self, canvas, font=None):
        self.canvas = canvas
        self.font = font if font is not None else tkfont.Font(root=canvas, family="TkDefaultFont", size=12, weight="bold")
        self.label_height = self.font.metrics("linespace") + 8
        self.items = []
        self.visible_count = 0
        self.rendered_key = None

    def clear(self):
        for group in self.items[:self.visible_count]:
            for item in group:
                self.canvas.itemconfigure(item, state="hidden")
        self.visible_count = 0
        self.rendered_key = None

    def _label(self, face, available_width):
        if face.status == "known":
            text, color = face.name or "Без имени", "#44d07b"
        elif face.status == "ambiguous":
            text, color = "Неоднозначное совпадение", "#f4ba54"
        else:
            text, color = "Неизвестный человек", "#ff6b6b"
        text = " ".join(text.split())
        # Shorten long names with a logarithmic number of font measurements.
        if self.font.measure(text) > available_width:
            low, high = 0, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if self.font.measure(text[:middle] + "…") <= available_width:
                    low = middle
                else:
                    high = middle - 1
            text = text[:low] + "…" if available_width >= self.font.measure("…") else ""
        return text, color

    def render(self, result, frame, image_rect, now):
        if (result is None or not result.faces or result.frame_size != (frame.image.shape[1], frame.image.shape[0])
                or result.sequence > frame.sequence or result.captured_at > frame.captured_at
                or not 0 <= now - result.captured_at <= MAX_OVERLAY_AGE):
            self.clear()
            return
        key = (result, image_rect)
        if key == self.rendered_key:
            return
        x, y, width, height = image_rect
        if width < 16 or height < self.label_height:
            self.clear()
            return
        faces = [face for face in result.faces if face.bounds is not None]
        while len(self.items) < len(faces):
            self.items.append((self.canvas.create_rectangle(0, 0, 0, 0, width=3, state="hidden"),
                               self.canvas.create_rectangle(0, 0, 0, 0, width=0, state="hidden"),
                               self.canvas.create_text(0, 0, anchor="nw", font=self.font, fill="#101010", state="hidden")))
        for face, (box, badge, caption) in zip(faces, self.items):
            left, top, right, bottom = project_bounds(face.bounds, result.frame_size, image_rect)
            text, color = self._label(face, width - 12)
            badge_width = min(width, self.font.measure(text) + 12)
            badge_x = max(x, min(left, x + width - badge_width))
            badge_y = top - self.label_height if top - self.label_height >= y else top
            badge_y = max(y, min(badge_y, y + height - self.label_height))
            self.canvas.coords(box, left, top, right, bottom)
            self.canvas.itemconfigure(box, outline=color, state="normal")
            self.canvas.coords(badge, badge_x, badge_y, badge_x + badge_width, badge_y + self.label_height)
            self.canvas.itemconfigure(badge, fill=color, state="normal")
            self.canvas.coords(caption, badge_x + 6, badge_y + 4)
            self.canvas.itemconfigure(caption, text=text, state="normal")
        for group in self.items[len(faces):self.visible_count]:
            for item in group:
                self.canvas.itemconfigure(item, state="hidden")
        self.visible_count = len(faces)
        self.rendered_key = key
