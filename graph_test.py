import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
y = np.sin(x)
# fig = plt.figure()
# plt.plot(x, y)
# for label in fig.get_ticklabels()[::2]:
#     label.set_visible(False)
# plt.show()

plt.scatter(x, y)
ax = plt.gca()
temp = ax.xaxis.get_ticklabels()
temp = list(set(temp) - set(temp[::2]))
for label in temp:
    label.set_visible(False)
plt.show()

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
# fig = plt.figure()
# x = np.linspace(-2, 2, 10)
# y = np.exp(x)
# plt.plot(x, y)
# plt.xlabel("$\bf{y=e^{x}}$")
# spacing = 0.100
# fig.subplots_adjust(bottom=spacing)
# plt.show()