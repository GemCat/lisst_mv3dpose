import matplotlib.pyplot as plt

LIMBS = [
        (0, 1), (0, 15), (0, 14), (15, 17), (14, 16),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (2, 8), (5, 11), (8, 11),
        (8, 9), (9, 10), (10, 21), (21, 22), (22, 23),
        (11, 12), (12, 13), (13, 18), (18, 19), (19, 20)
    ]

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def colors(n_tracks):
    if n_tracks > 11:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        colors = [
            'tab:blue',
            'tab:orange',
            'tab:green',
            'tab:red',
            'tab:purple',
            'red',
            'blue',
            'green',
            'navy',
            'maroon',
            'darkgreen'
        ]
    return colors

