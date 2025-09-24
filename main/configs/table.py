"""
Shared table geometry constants for consistent scaling across modules.

Units:
- centimeters (cm) for vision homography and JSON outputs
- meters (m) for solver/pygame world coordinates
- millimeters (mm) derived for robot payload conversion
"""

# Real table dimensions
TABLE_W_CM: float = 73.5
TABLE_H_CM: float = 37.5

# Meters
TABLE_W_M: float = TABLE_W_CM / 100.0
TABLE_H_M: float = TABLE_H_CM / 100.0
TABLE_M = (TABLE_W_M, TABLE_H_M)

# Millimeters (integer-friendly height used in robot payload Y flip)
TABLE_H_MM: float = TABLE_H_CM * 10.0  # 37.5 cm → 375 mm

