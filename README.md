# lex-lutor
WIP application for 3D Lookup Table Manipulation.
This is a technical prototype with fixed layout and no undo.
The code is a mess (and so the commit log), as, currently, this project is just for curiosity. If it is useful and I have time and motivation, it will be rewritten from scratch with a (hopefully) better architecture and proper packaging. However, contributions are always welcome ;)

# Usage
Start main.py
For now, usage is heavily shortcut based.
First, load an image and an LUT or create a new LUT. **Large LUT will lead to slow performance! For now, try size 9 to 11**

Manipulation works as follows:
- RGB master curve is in the curve tab. Internally, the curve is always applied before each subsequent transformation of individual LUT nodes.

## Selection
- Select either individual nodes by either:
   - clicking them in the LUT -> shift + click to select multiple.
   - double clicking on the image to select all nodes affecting the color of selected pixel -> shift+double click on different pixels to expand selection; ctrl + double click to select only the node affecting the clicked pixel most.
- Hovering over an LUT node shows effect on image pixels. holding shift and then hovering over a node shows effect of currently selected nodes + hovered node.
- Space bar toggles permanent selection preview in the image preview. Due to a bug, a click somewhere in the 3d lut view must be performed always before toggling is possible
- In the selection panel, the slider can be used to create a smooth expansion of the selection around the selected LUT nodes. left handle is core radius and right handle is outer radius along the corresponding axis.

## Manipulation
- while hovering over the 3D LUT view, press one of the following keys: [h,s,v,l,r,g,b] to shift the selected nodes along the corresponding axis. left mouse click to apply, right so cancel.
- like above, but having CTRL pressed while key press scales the nodes' position along the given axis, with the selection center of gravity as center. Use this to reduce or increase the difference of selected nodes along an axis.
- Press n to reset selected nodes.
