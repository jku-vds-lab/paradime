"""A simple script to extract the colors of the rects in an svg-file.
"""

import json
import re
import xml.dom.minidom

import numpy as np

PALETTE_SVG_IN = "palette.svg"
PALETTE_JSON_OUT = "palette.json"

RECT_DTYPE = [("x", float), ("y", float), ("fill", "U7")]
NAME_DTYPE = [("x", float), ("y", float), ("name", "U15")]


def make_palette(in_file: str, out_file: str) -> None:

    svg = xml.dom.minidom.parse(in_file)

    # extract rect elements, coordinates and fills
    rect_vals = []
    for rect in svg.getElementsByTagName("rect"):
        match = re.match(r"fill:([^;]*)", rect.getAttribute("style"))
        rect_vals.append(
            (
                float(rect.getAttribute("x")),
                float(rect.getAttribute("y")),
                match.group(1) if match else "#ffffff",
            )
        )
    rect_data = np.array(rect_vals, dtype=RECT_DTYPE)
    # sort fills by y first, then by x
    rect_data.sort(order=["y", "x"])

    # extract text (i.e., tspan) elements and coordinates
    name_vals = []
    for tspan in svg.getElementsByTagName("tspan"):
        name_vals.append(
            (
                float(tspan.getAttribute("x")),
                float(tspan.getAttribute("y")),
                tspan.childNodes[0].nodeValue,
            )
        )
    name_data = np.array(name_vals, dtype=NAME_DTYPE)
    # sort names by y first, then by x
    name_data.sort(order=["y", "x"])

    palette = dict(zip(name_data["name"], rect_data["fill"]))

    with open(out_file, "w") as f:
        json.dump(palette, f)


if __name__ == "__main__":
    make_palette(PALETTE_SVG_IN, PALETTE_JSON_OUT)
