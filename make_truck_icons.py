"""Generate the truck icon sprite used by the dashboard map.

    python make_truck_icons.py

Writes assets/truck_sprite/sprite{,@2x}.{png,json}: a MapLibre sprite sheet with
one side-view truck per state colour, facing right ("-r") and left ("-l").
Dash serves the assets/ folder, and app.py points the basemap style at it.
"""
import json
import os

from PIL import Image, ImageDraw

OUT = os.path.join("assets", "truck_sprite")
STATE_COLORS = {          # keep in sync with the dashboard legend
    "full": "#6a1b9a",      # driving with bikes
    "empty": "#757575",     # driving empty
    "handling": "#ef6c00",  # loading / unloading
    "idle": "#212121",
}
W, H = 40, 26             # icon size at pixelRatio 1


def draw_truck(color: str, scale: int) -> Image.Image:
    """Side-view box truck facing right, with a white outline for contrast."""
    img = Image.new("RGBA", (W * scale, H * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def box(x0, y0, x1, y1, fill, r=1.5):
        d.rounded_rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                            radius=r * scale, fill=fill, outline="white", width=max(1, scale))

    def wheel(cx, cy, r):
        d.ellipse([(cx - r) * scale, (cy - r) * scale, (cx + r) * scale, (cy + r) * scale],
                  fill="#111111", outline="white", width=max(1, scale))
        d.ellipse([(cx - r / 2.5) * scale, (cy - r / 2.5) * scale,
                   (cx + r / 2.5) * scale, (cy + r / 2.5) * scale], fill="#bbbbbb")

    box(2, 3, 25, 19, color, r=2)              # cargo box
    box(25, 8, 37, 19, color, r=2.5)           # cab
    box(28, 10, 35, 14, "#dbe9f5", r=1)        # window
    wheel(9, 20, 4)
    wheel(30, 20, 4)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    for scale, suffix in ((1, ""), (2, "@2x")):
        icons = {}
        for state, color in STATE_COLORS.items():
            right = draw_truck(color, scale)
            icons[f"truck-{state}-r"] = right
            icons[f"truck-{state}-l"] = right.transpose(Image.FLIP_LEFT_RIGHT)
        sheet = Image.new("RGBA", (W * scale * len(icons), H * scale), (0, 0, 0, 0))
        index = {}
        for i, (name, img) in enumerate(icons.items()):
            sheet.paste(img, (i * W * scale, 0))
            entry = {"x": i * W * scale, "y": 0, "width": W * scale,
                     "height": H * scale, "pixelRatio": scale}
            # Plotly may look icons up as "<symbol>-15" (maki naming); serve both.
            index[name] = index[f"{name}-15"] = entry
        sheet.save(os.path.join(OUT, f"sprite{suffix}.png"))
        with open(os.path.join(OUT, f"sprite{suffix}.json"), "w") as f:
            json.dump(index, f, indent=1)
    print(f"wrote {len(icons)} icons to {OUT}/")


if __name__ == "__main__":
    main()
