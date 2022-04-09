import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



width = []
height = []
# with open("/home/zjw/wh.txt", "r") as f:
with open("/home/zjw/wh_remo_data_layer.txt", "r") as f:
# with open("/home/zjw/wh_mini_hand.txt", "r") as f:
    for line in f:
        w, h = line.strip().split()
        width.append(w)
        height.append(h)

df = pd.DataFrame({"width": width,
                   "height": height})
plt.subplot(121)
sns.distplot(width)
sns.distplot(height)
plt.show()

