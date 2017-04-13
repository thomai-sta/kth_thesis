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

mar_size = 8

plt.plot(1 - 0.646647, 0.665666, 'o', label="M:1, ED: 0.486", color=colors[0], markersize=mar_size)
plt.plot(1 - 0.434434, 0.504254, 'v', label="M:2, ED: 0.752, F:5", color=colors[1], markersize=mar_size)
plt.plot(1 - 0.434434, 0.504254, '^', label="M:2, ED: 0.752, F:7", color=colors[2], markersize=mar_size)
plt.plot(1 - 0.425425, 0.505828, '<', label="M:2, ED: 0.758, F:10", color=colors[3], markersize=mar_size)
plt.plot(1 - 0.626627, 0.672222, '>', label="M:3, ED: 0.497, F:5", color=colors[4], markersize=mar_size)
plt.plot(1 - 0.622623, 0.685022, 's', label="M:3, ED: 0.492, F:7", color=colors[5], markersize=mar_size)
plt.plot(1 - 0.622623, 0.686600, "*", label="M:3, ED: 0.491, F:10", color=colors[6], markersize=mar_size)

plt.legend(loc=0, fancybox=True, shadow=True)
plt.title("Best Performances of Blind Experiments")

# CV
plt.figure()
plt.plot(x, y, '--', label="Random", linewidth=2.0)
plt.axis('scaled')
plt.axis([0, 1, 0, 1])

plt.plot(1 - 0.689690, 0.655265, 'o', label="M:1, ED: 0.464", color=colors[0], markersize=mar_size)
plt.plot(1 - 0.596597, 0.655334, 'v', label="M:2, ED: 0.531, F:5", color=colors[1], markersize=mar_size)
plt.plot(1 - 0.597598, 0.654305, '^', label="M:2, ED: 0.531, F:7", color=colors[2], markersize=mar_size)
plt.plot(1 - 0.597598, 0.654216, '<', label="M:2, ED: 0.531, F:10", color=colors[3], markersize=mar_size)
plt.plot(1 - 0.641642, 0.776754, '>', label="M:3, ED: 0.422, F:5", color=colors[4], markersize=mar_size)
plt.plot(1 - 0.670671, 0.715756, 's', label="M:3, ED: 0.435, F:7", color=colors[5], markersize=mar_size)
plt.plot(1 - 0.659660, 0.728020, '*', label="M:3, ED: 0.436, F:10", color=colors[6], markersize=mar_size)

plt.legend(loc=0, fancybox=True, shadow=True)
plt.title("Best Performances of Semi-Blind Experiments")


# no CV
plt.figure()
plt.plot(x, y, '--', label="Random", linewidth=2.0)
plt.axis('scaled')
plt.axis([0, 1, 0, 1])
plt.plot(1 - 0.719720, 0.671462, 'o', label="M:1, ED: 0.432", color=colors[0], markersize=mar_size)
plt.plot(1 - 0.521522, 0.500671, 'v', label="M:2, ED: 0.692, F:5", color=colors[1], markersize=mar_size)
plt.plot(1 - 0.521522, 0.500538, '^', label="M:2, ED: 0.692, F:7", color=colors[2], markersize=mar_size)
plt.plot(1 - 0.522523, 0.500446, '<', label="M:2, ED: 0.691, F:10", color=colors[3], markersize=mar_size)
plt.plot(1 - 0.769770, 0.677865, '>', label="M:3, ED: 0.396, F:5", color=colors[4], markersize=mar_size)
plt.plot(1 - 0.742743, 0.675361, 's', label="M:3, ED: 0.414, F:7", color=colors[5], markersize=mar_size)
plt.plot(1 - 0.764765, 0.688727, '*', label="M:3, ED: 0.39,  F:10", color=colors[6], markersize=mar_size)
plt.legend(loc='lower left', fancybox=True, shadow=True)
plt.title("Best Performances of Blind Experiments for 3 Follow-Ups")


# CV
plt.figure()
plt.plot(x, y, '--', label="Random", linewidth=2.0)
plt.axis('scaled')
plt.axis([0, 1, 0, 1])
plt.plot(1 - 0.753754, 0.642422, 'o', label="M:1, ED: 0.434", color=colors[0], markersize=mar_size)
plt.plot(1 - 0.720721, 0.595293, 'v', label="M:2, ED: 0.492, F:5", color=colors[1], markersize=mar_size)
plt.plot(1 - 0.720721, 0.595293, '^', label="M:2, ED: 0.492, F:7", color=colors[2], markersize=mar_size)
plt.plot(1 - 0.720721, 0.595293, '<', label="M:2, ED: 0.492, F:10", color=colors[3], markersize=mar_size)
plt.plot(1 - 0.740741, 0.682292, '>', label="M:3, ED: 0.41,  F:5", color=colors[4], markersize=mar_size)
plt.plot(1 - 0.840841, 0.645124, 's', label="M:3, ED: 0.389, F:7", color=colors[5], markersize=mar_size)
plt.plot(1 - 0.840841, 0.645422, '*', label="M:3, ED: 0.389, F:10", color=colors[6], markersize=mar_size)
plt.legend(loc='lower left', fancybox=True, shadow=True)
plt.title("Best Performances of Semi-Blind Experiments for 3 Follow-Ups")



plt.show()