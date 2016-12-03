#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties



cmap = plt.get_cmap('gist_rainbow')
colors = cmap(np.linspace(0, 1.0, 7))

# no CV
x = [0, 1]
y = [0, 1]
plt.plot(x, y, '--', label="Random", linewidth=2.0)
plt.axis('scaled')
plt.axis([0, 1, 0, 1])

plt.plot(1 - 0.646647, 0.665666, 'o', label="M:1, ED: 0.486455, AUC: 0.661004", color=colors[0])
plt.plot(1 - 0.434434, 0.504254, 'o', label="M:2, ED: 0.752083, AUC: 0.459012, F:5", color=colors[1])
plt.plot(1 - 0.434434, 0.504254, 'o', label="M:2, ED: 0.752083, AUC: 0.459830, F:7", color=colors[2])
plt.plot(1 - 0.425425, 0.505828, 'o', label="M:2, ED: 0.757854, AUC: 0.456713, F:10", color=colors[3])
plt.plot(1 - 0.626627, 0.672222, 'o', label="M:3, ED: 0.496836, AUC: 0.642389, F:5", color=colors[4])
plt.plot(1 - 0.622623, 0.685022, 'o', label="M:3, ED: 0.491553, AUC: 0.648800, F:7", color=colors[5])
plt.plot(1 - 0.622623, 0.686600, 'o', label="M:3, ED: 0.490544, AUC: 0.658971, F:10", color=colors[6])

plt.legend(loc=0, fancybox=True, shadow=True)
plt.title("Best Performances of Methods without Cross-Validation")

# CV
plt.figure()
plt.plot(x, y, '--', label="Random", linewidth=2.0)
plt.axis('scaled')
plt.axis([0, 1, 0, 1])

plt.plot(1 - 0.689690, 0.655265, 'o', label="M:1, ED: 0.463826, AUC: 0.638763", color=colors[0])
plt.plot(1 - 0.596597, 0.655334, 'o', label="M:2, ED: 0.530593, AUC: 0.626058, F:5", color=colors[1])
plt.plot(1 - 0.597598, 0.654305, 'o', label="M:2, ED: 0.530502, AUC: 0.634837, F:7", color=colors[2])
plt.plot(1 - 0.597598, 0.654216, 'o', label="M:2, ED: 0.530560, AUC: 0.629073, F:10", color=colors[3])
plt.plot(1 - 0.641642, 0.776754, 'o', label="M:3, ED: 0.422208, AUC: 0.708210, F:5", color=colors[4])
plt.plot(1 - 0.670671, 0.715756, 'o', label="M:3, ED: 0.435031, AUC: 0.714058, F:7", color=colors[5])
plt.plot(1 - 0.659660, 0.728020, 'o', label="M:3, ED: 0.435666, AUC: 0.706612, F:10", color=colors[6])

plt.legend(loc=0, fancybox=True, shadow=True)
plt.title("Best Performances of Methods with Cross-Validation")


# no CV
plt.figure()
plt.plot(x, y, '--', label="Random", linewidth=2.0)
plt.axis('scaled')
plt.axis([0, 1, 0, 1])
plt.plot(1 - 0.719720, 0.671462, 'o', label="M:1, ED: 0.431850, AUC: 0.692795", color=colors[0])
plt.plot(1 - 0.521522, 0.500671, 'o', label="M:2, ED: 0.691572, AUC: 0.520200, F:5", color=colors[1])
plt.plot(1 - 0.521522, 0.500538, 'o', label="M:2, ED: 0.691667, AUC: 0.520140, F:7", color=colors[2])
plt.plot(1 - 0.522523, 0.500446, 'o', label="M:2, ED: 0.691042, AUC: 0.521033, F:10", color=colors[3])
plt.plot(1 - 0.769770, 0.677865, 'o', label="M:3, ED: 0.395951, AUC: 0.727321, F:5", color=colors[4])
plt.plot(1 - 0.742743, 0.675361, 'o', label="M:3, ED: 0.414212, AUC: 0.724421, F:7", color=colors[5])
plt.plot(1 - 0.764765, 0.688727, 'o', label="M:3, ED: 0.390162, AUC: 0.719585, F:10", color=colors[6])
plt.legend(loc='lower left', fancybox=True, shadow=True)
plt.title("Best Performances of Methods without Cross-Validation for 3 Follow-Ups")


# CV
plt.figure()
plt.plot(x, y, '--', label="Random", linewidth=2.0)
plt.axis('scaled')
plt.axis([0, 1, 0, 1])
plt.plot(1 - 0.753754, 0.642422, 'o', label="M:1, ED: 0.434165, AUC: 0.657766", color=colors[0])
plt.plot(1 - 0.720721, 0.595293, 'o', label="M:2, ED: 0.491716, AUC: 0.652802, F:5", color=colors[1])
plt.plot(1 - 0.720721, 0.595293, 'o', label="M:2, ED: 0.491716, AUC: 0.649057, F:7", color=colors[2])
plt.plot(1 - 0.720721, 0.595293, 'o', label="M:2, ED: 0.491716, AUC: 0.647783, F:10", color=colors[3])
plt.plot(1 - 0.740741, 0.682292, 'o', label="M:3, ED: 0.410065, AUC: 0.740186, F:5", color=colors[4])
plt.plot(1 - 0.840841, 0.645124, 'o', label="M:3, ED: 0.388933, AUC: 0.769279, F:7", color=colors[5])
plt.plot(1 - 0.840841, 0.645422, 'o', label="M:3, ED: 0.388660, AUC: 0.767256, F:10", color=colors[6])
plt.legend(loc='lower left', fancybox=True, shadow=True)
plt.title("Best Performances of Methods with Cross-Validation for 3 Follow-Ups")



plt.show()