from librairie import *

dic_pixel_banlance = {}

for idx, Class in enumerate(CLASS_COLORS.keys()):

    percent = (classe_pixel_frequence[idx] * 100).item()

    dic_pixel_banlance[Class] = np.round(
        percent,
        4
    ).item()


fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)

width = dic_pixel_banlance.values()

plt.barh(
    dic_pixel_banlance.keys(),
    width,
    align="center"
)

plt.xlabel("Pourcentage (%)")
plt.title("Répartition des pixels par classe")


plt.subplot(1, 2, 2)

plt.pie(
    dic_pixel_banlance.values(),
    autopct="%1.1f%%",
    labels=dic_pixel_banlance.keys()
)

plt.legend(
    labels=dic_pixel_banlance.keys(),
    loc="right",
    bbox_to_anchor=(0.05, 0.2)
)

plt.title("Répartition des pixels")

plt.tight_layout()
plt.show()

plt.close(fig)



for i in os.listdir("data_project"):

    print(f"Directory - {i}")

    for j in os.listdir(os.path.join("data_project", i)):

        path = os.path.join(
            "data_project",
            i,
            j
        )

        print(f"├── {j} : {len(os.listdir(path))}")

