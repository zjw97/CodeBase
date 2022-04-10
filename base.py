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

def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels