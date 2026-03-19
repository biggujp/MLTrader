from config import *

def build_grid(price, spacing):
    grid = []
    
    for i in range(1, GRID_LEVELS+1):
        grid.append(price - i * spacing)
        grid.append(price + i * spacing)
    
    return grid

def get_position_size(balance, spacing):
    risk = balance * 0.01
    size = risk / spacing
    return max(0.001, min(size, 0.02))