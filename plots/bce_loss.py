from math import log

import numpy as np
import plotly.express as px

sigma = .001
def bce(gt, pred):
    return -1 * gt * log(max(pred,sigma), 2) - (1 - gt) * log(1 - max(pred,sigma), 2)


# print(bce(1, .1))
# print(bce(0, .9))
def focal_loss(p, y):
    if y == 1:
        p_t = p
    else:
        p_t = 1-p
    return -1* log(max(p_t, sigma), 2)

x = np.linspace(0, 1, num=100)
print(x)
fig = px.line(x=x, y=[bce(i, 0) for i in x])

fig.show()
