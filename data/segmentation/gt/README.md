# Project Title

## Overview
This document includes important details about how the ground truth to graph works. 

To perform conversion of annotated floorplan to graph for CAB, change the variable svg_file_path in graph_generation to  './CAB_Floor_E_Annotated.svg' and run 
```
python3 graph_generation.py
```

### Room and Corridor Nodes
- Rooms and corridors are segmented into multiple nodes. For example, corridor `10.003` is split into:
  - `10.003A`
  - `10.003B`
  - `10.003C`
  - `10.003D`
- When referencing a part of a room or corridor that ends with a letter (A, B, C, D), such as `10.003A`, it refers back to the main room or corridor, e.g., `10.003`.
- Splitting big rooms or long corridors into multiple nodes help with distance calculation and room connections.

### Distance Calculation
- **Rooms, Doors, and Connectors**: The path distance for rooms, doors, and connectors is calculated using the width of their respective bounding box.
- **Stairs**: For stairs, the length of the box is used to determine the path distance.
- **Pixel-to-Meter Conversion**: The total distance calculated in pixels is converted to meters by dividing by 15, as **15 pixels** are equivalent to **1 meter**.

## Implementation Notes
- Ensure that all pathfinding logic and distance calculations incorporate these assumptions and naming conventions.