import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker


#@ dataset class distribution
data_distribution = [
    ['Surprised', 0.2027, 0.1408, 0.0989, 0.1108, 0.1051, 0.0777],
    ['Fear', 0.0587, 0.1502, 0.0967, 0.1425, 0.0228, 0.0115], 
    ['Disgust', 0.1440, 0.1361, 0.0703, 0.0152, 0.0585, 0.0446], 
    ['Happy', 0.1627, 0.1502, 0.2131, 0.2522, 0.3890, 0.3397], 
    ['Sad', 0.0640, 0.1408, 0.1714, 0.1674, 0.1615, 0.1100], 
    ['Angry', 0.1067, 0.1408, 0.1901, 0.1388, 0.0574, 0.0404], 
    ['Neutral', 0.2613, 0.1408, 0.1593, 0.1730, 0.2056, 0.3760]
]
methods = ["Expression Categories", "CK+", "JAFFE", "SFEW2.0", "FER2013", "RAF-DB", "ExpW"]


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax1.set_ylabel('Proportion')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)


df = pd.DataFrame(data_distribution, columns=methods)
df.plot(x=methods[0], y=methods[1:], kind="bar",figsize=(9,8), ax=ax1, rot=360, grid=True, colormap='Set2_r', edgecolor='k', width=0.8)
ax1.grid(linestyle='--')
ax1.grid(axis='x')
ax1.legend(bbox_to_anchor=(0., 1.02, 1., 0.2), loc='lower left', ncol=3, mode="expand", borderaxespad=0., frameon=False)

plt.show()

