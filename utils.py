

"""
Programa que recoge funciones auxiliares necesarias
"""
#Librerías
import matplotlib.pyplot as plt


# Función de ploter
def plot(COST, ACC):
    """
    Crea una gráfica de doble eje Y, el eje izquierdo será 
    para la pérdida y el eje derecho para la precisión.
    La función twinx permite superponer dos gráficos que 
    tienen escalas distintas pero comparten el mismo eje X
    con las épocas
    """
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('pérdida total', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y',color=color)

    fig.tight_layout()
    plt.show()
