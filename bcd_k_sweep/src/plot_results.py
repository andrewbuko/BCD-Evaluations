import json
import matplotlib.pyplot as plt

with open("results_bcd.json", "r") as f:
    data = json.load(f)

def xy(block):
    xs, ys = [], []
    for p in block["points"]:
        xs.append(p["k"])
        ys.append(p["rate"])
    return xs, ys

o = data["openai"]
g = data["gemini"]

ox, oy = xy(o)
gx, gy = xy(g)

plt.figure()
plt.plot(ox, oy, marker="o", label=o["model"])
plt.plot(gx, gy, marker="o", label=g["model"])
plt.axhline(0.5, linewidth=1)
plt.xlabel("k (prefix length)")
plt.ylabel("rate")
plt.title("r(k) pilot")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("rk_plot.png", dpi=200)
print("Saved rk_plot.png")