import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def dibuja_validacion_cruzada():
    plt.figure(figsize=(12, 2))
    plt.title("Validacion cruzada")
    axes = plt.gca()
    axes.set_frame_on(False)

    n_grupos = 5
    n_muestras = 25

    n_muestras_por_grupos = n_muestras/float(n_grupos)

    for i in range(n_grupos):
        colores = ["w"] * n_grupos
        colores[i] = "grey"
        bars = plt.barh(y=range(n_grupos), width=[n_muestras_por_grupos - 0.1] * n_grupos,
            left=i * n_muestras_por_grupos, height=.6, color=colores, hatch="//",
            edgecolor='k', align='edge')
    axes.invert_yaxis()
    axes.set_xlim(0, n_muestras + 1)
    plt.ylabel("Interaciones CV")
    plt.xlabel("Puntos de datos")
    plt.xticks(np.arange(n_muestras_por_grupos / 2., n_muestras,
                         n_muestras_por_grupos),
               ["Grupos %d" % x for x in range(1, n_grupos + 1)])
    plt.yticks(np.arange(n_grupos) + .3,
               ["Division %d" % x for x in range(1, n_grupos + 1)])
    plt.legend([bars[0], bars[4]], ['Datos de entrenamiento', 'Datos de prueba'],
               loc=(1.05, 0.4), frameon=False)


def dibuja_validacion_cruzada_estratificada():
    fig, both_axes = plt.subplots(2, 1, figsize=(12, 5))
    axes = both_axes[0]
    axes.set_title("Validacion cruzada estandar con etiquetas de clase clasificadas")

    axes.set_frame_on(False)

    n_grupos = 3
    n_muestras = 150

    n_muestras_por_grupo = n_muestras / float(n_grupos)

    for i in range(n_grupos):
        colors = ["w"] * n_grupos
        colors[i] = "grey"
        axes.barh(y=range(n_grupos), width=[n_muestras_por_grupo - 1] *
                  n_grupos, left=i * n_muestras_por_grupo, height=.6,
                  color=colors, hatch="//", edgecolor='k', align='edge')

    axes.barh(y=[n_grupos] * n_grupos, width=[n_muestras_por_grupo - 1] *
              n_grupos, left=np.arange(3) * n_muestras_por_grupo, height=.6,
              color="w", edgecolor='k', align='edge')

    axes.invert_yaxis()
    axes.set_xlim(0, n_muestras + 1)
    axes.set_ylabel("Iteraciones CV")
    axes.set_xlabel("Puntos de datos")
    axes.set_xticks(np.arange(n_muestras_por_grupo / 2.,
                              n_muestras, n_muestras_por_grupo))
    axes.set_xticklabels(["Grupo %d" % x for x in range(1, n_grupos + 1)])
    axes.set_yticks(np.arange(n_grupos + 1) + .3)
    axes.set_yticklabels(
        ["Division %d" % x for x in range(1, n_grupos + 1)] + ["Etiquetas de clase"])
    for i in range(3):
        axes.text((i + .5) * n_muestras_por_grupo, 3.5, "Clase %d" %
                  i, horizontalalignment="center")

    ax = both_axes[1]
    ax.set_title("Validacion cruzada estratificada")
    ax.set_frame_on(False)
    ax.invert_yaxis()
    ax.set_xlim(0, n_muestras + 1)
    ax.set_ylabel("IteracionesCV ")
    ax.set_xlabel("Puntos de datos")

    ax.set_yticks(np.arange(n_grupos + 1) + .3)
    ax.set_yticklabels(
        ["Division %d" % x for x in range(1, n_grupos + 1)] + ["Etiquetas de clase"])

    n_subsplit = n_muestras_por_grupo / 3.
    for i in range(n_grupos):
        bar_prueba = ax.barh(
            y=[i] * n_grupos, width=[n_subsplit - 1] * n_grupos,
            left=np.arange(n_grupos) * n_muestras_por_grupo + i * n_subsplit,
            height=.6, color="grey", hatch="//", edgecolor='k', align='edge')

    w = 2 * n_subsplit - 1
    ax.barh(y=[0] * n_grupos, width=[w] * n_grupos, left=np.arange(n_grupos)
            * n_muestras_por_grupo + (0 + 1) * n_subsplit, height=.6, color="w",
            hatch="//", edgecolor='k', align='edge')
    ax.barh(y=[1] * (n_grupos + 1), width=[w / 2., w, w, w / 2.],
            left=np.maximum(0, np.arange(n_grupos + 1) * n_muestras_por_grupo -
                            n_subsplit), height=.6, color="w", hatch="//",
            edgecolor='k', align='edge')
    bar_entrenamiento = ax.barh(y=[2] * n_grupos, width=[w] * n_grupos,
                            left=np.arange(n_grupos) * n_muestras_por_grupo,
                            height=.6, color="w", hatch="//", edgecolor='k',
                            align='edge')

    ax.barh(y=[n_grupos] * n_grupos, width=[n_muestras_por_grupo - 1] *
            n_grupos, left=np.arange(n_grupos) * n_muestras_por_grupo, height=.6,
            color="w", edgecolor='k', align='edge')

    for i in range(3):
        ax.text((i + .5) * n_muestras_por_grupo, 3.5, "Clase %d" %
                i, horizontalalignment="center")
    ax.set_ylim(4, -0.1)
    plt.legend([bar_entrenamiento[0], bar_prueba[0]], [
               'Datos de entrenamiento', 'Datos de prueba'], loc=(1.05, 1), frameon=False)

    fig.tight_layout()

def dibuja_validacion_cruzada_aleatoria():
    from sklearn.model_selection import ShuffleSplit
    plt.figure(figsize=(10, 2))
    plt.title("Validacion cruzada aleatoria con 10 points"
              ", num_entrenamiento=5, num_prueba= 2, n_division=4")

    axes = plt.gca()
    axes.set_frame_on(False)

    n_grupos = 10
    n_muestras = 10
    n_iter = 4
    n_muestras_por_grupo = 1

    ss = ShuffleSplit(n_splits=4, train_size=5, test_size=2, random_state=43)
    mascara_bool = np.zeros((n_iter, n_muestras))
    for i, (entrenamiento, prueba) in enumerate(ss.split(range(10))):
        mascara_bool[i, entrenamiento] = 1
        mascara_bool[i, prueba] = 2

    for i in range(n_grupos):
        colores = ["grey" if x == 2 else "white" for x in mascara_bool[:, i]]

        cajas = axes.barh(y=range(n_iter), width=[1 - 0.1] * n_iter,
                          left=i * n_muestras_por_grupo, height=.6, color=colores,
                          hatch="//", edgecolor='k', align='edge')
        for j in np.where(mascara_bool[:, i] == 0)[0]:
            cajas[j].set_hatch("")

    axes.invert_yaxis()
    axes.set_xlim(0, n_muestras + 1)
    axes.set_ylabel("Interaciones CV")
    axes.set_xlabel("Puntos de datos")
    axes.set_xticks(np.arange(n_muestras) + .5)
    axes.set_xticklabels(np.arange(1, n_muestras + 1))
    axes.set_yticks(np.arange(n_iter) + .3)
    axes.set_yticklabels(["Division %d" % x for x in range(1, n_iter + 1)])
    plt.legend([cajas[1], cajas[0], cajas[2]], [
               "Conjunto de entrenamiento", "Conjunto de prueba", "No seleccionados"], loc=(1, .3))
    plt.tight_layout()

def dibuja_grupo_kfold():
    from sklearn.model_selection import GroupKFold
    grupos = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]

    plt.figure(figsize=(10, 2))
    plt.title("GroupKFold")

    axes = plt.gca()
    axes.set_frame_on(False)

    n_grupos = 12
    n_muestras = 12
    n_iter = 3
    n_muestras_por_grupo = 1

    cv = GroupKFold(n_splits=3)
    mascara_bool = np.zeros((n_iter, n_muestras))
    for i, (entrenamiento, prueba) in enumerate(cv.split(range(12), groups=grupos)):
        mascara_bool[i, entrenamiento] = 1
        mascara_bool[i, prueba] = 2

    for i in range(n_grupos):
        colores = ["grey" if x == 2 else "white" for x in mascara_bool[:, i]]
        cajas = axes.barh(y=range(n_iter), width=[1 - 0.1] * n_iter,
                          left=i * n_muestras_por_grupo, height=.6, color=colores,
                          hatch="//", edgecolor="k", align='edge')
        for j in np.where(mascara_bool[:, i] == 0)[0]:
            cajas[j].set_hatch("")

    axes.barh(y=[n_iter] * n_grupos, width=[1 - 0.1] * n_grupos,
              left=np.arange(n_grupos) * n_muestras_por_grupo, height=.6,
              color="w", edgecolor='k', align="edge")

    for i in range(12):
        axes.text((i + .5) * n_muestras_por_grupo, 3.5, "%d" %
                  grupos[i], horizontalalignment="center")

    axes.invert_yaxis()
    axes.set_xlim(0, n_muestras + 1)
    axes.set_ylabel("Iteraciones CV ")
    axes.set_xlabel("Puntos de datos")
    axes.set_xticks(np.arange(n_muestras) + .5)
    axes.set_xticklabels(np.arange(1, n_muestras + 1))
    axes.set_yticks(np.arange(n_iter + 1) + .3)
    axes.set_yticklabels(
        ["Division %d" % x for x in range(1, n_iter + 1)] + ["Grupos"])
    plt.legend([cajas[0], cajas[1]], ["Conjunto de entrenamiento", "Conjunto de prueba"], loc=(1, .3))
    plt.tight_layout()

def dibuja_division_tres_grupos():
    plt.figure(figsize=(15, 1))
    axis = plt.gca()
    cajas = axis.barh([0, 0, 0], [11.9, 2.9, 4.9], left=[0, 12, 15], color=[
                     'white', 'grey', 'grey'], hatch="//", edgecolor='k',
                     align='edge')
    cajas[2].set_hatch(r"")
    axis.set_yticks(())
    axis.set_frame_on(False)
    axis.set_ylim(-.1, .8)
    axis.set_xlim(-0.1, 20.1)
    axis.set_xticks([6, 13.3, 17.5])
    axis.set_xticklabels(["Conjunto de entrenamiento", "Conjunto de validacion",
                          "Conjunto de prueba"], fontdict={'fontsize': 15})
    axis.tick_params(length=0, labeltop=True, labelbottom=False)
    axis.text(6, -.3, "Ajuste del modelo",
              fontdict={'fontsize': 11}, horizontalalignment="center")
    axis.text(13.3, -.3, "Seleccion de parametros",
              fontdict={'fontsize': 11}, horizontalalignment="center")
    axis.text(17.5, -.3, "Evaluacion",
              fontdict={'fontsize': 11}, horizontalalignment="center")
