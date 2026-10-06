import numpy as np

print("NEURONA ARTIFICIAL PARA RIEGO DE PLANTAS")

X = np.array([
    [80, 18],
    [70, 22],
    [65, 28],
    [55, 25],
    [50, 32],
    [40, 30],
    [35, 25],
    [30, 32],
    [20, 35],
    [10, 38]
], dtype=float)

y = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1]
], dtype=float)

escala = np.array([100, 50], dtype=float)

X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def entrenar(tasa_aprendizaje, epocas):

    rng = np.random.default_rng(7)

    pesos = rng.normal(size=(2, 1))

    sesgo = 0.0

    for epoca in range(epocas):

        z = X_normalizado @ pesos + sesgo

        probabilidades = sigmoide(z)

        error = probabilidades - y

        gradiente_z = (
            2
            * error
            * probabilidades
            * (1 - probabilidades)
            / len(X_normalizado)
        )

        gradiente_pesos = X_normalizado.T @ gradiente_z

        gradiente_sesgo = np.sum(gradiente_z)

        pesos -= tasa_aprendizaje * gradiente_pesos

        sesgo -= tasa_aprendizaje * gradiente_sesgo

    z_final = X_normalizado @ pesos + sesgo

    probabilidades_finales = sigmoide(z_final)

    respuestas = (probabilidades_finales >= 0.5).astype(int)

    error_final = np.mean(
        (probabilidades_finales - y) ** 2
    )

    return pesos, sesgo, error_final, probabilidades_finales, respuestas


tasa_aprendizaje = 0.5

epocas = 10000

pesos, sesgo, error_final, probabilidades, respuestas = entrenar(
    tasa_aprendizaje,
    epocas
)

print("\nRESULTADOS DEL ENTRENAMIENTO")

print("Peso de humedad:", pesos[0, 0])

print("Peso de temperatura:", pesos[1, 0])

print("Sesgo:", sesgo)

print("Error final:", error_final)


print("\nDATOS DE ENTRENAMIENTO")

print("Humedad | Temperatura | Esperado | Probabilidad | Respuesta")

for entrada, esperado, probabilidad, respuesta in zip(
    X,
    y.ravel(),
    probabilidades.ravel(),
    respuestas.ravel()
):

    print(
        f"{entrada[0]:.0f}% | "
        f"{entrada[1]:.0f} C | "
        f"{int(esperado)} | "
        f"{probabilidad:.4f} | "
        f"{respuesta}"
    )


correctas = np.sum(respuestas.ravel() == y.ravel())

print("\nRespuestas correctas:", correctas, "/ 10")


print("\nPREDICCIONES NUEVAS")

nuevos_datos = np.array([
    [75, 30],
    [45, 34],
    [25, 22],
    [50, 25],
    [30, 40]
], dtype=float)

nuevos_normalizados = nuevos_datos / escala

probabilidades_nuevas = sigmoide(
    nuevos_normalizados @ pesos + sesgo
)

respuestas_nuevas = (
    probabilidades_nuevas >= 0.5
).astype(int)


for entrada, probabilidad, respuesta in zip(
    nuevos_datos,
    probabilidades_nuevas.ravel(),
    respuestas_nuevas.ravel()
):

    if respuesta == 1:
        decision = "REGAR"
    else:
        decision = "NO REGAR"

    print(
        f"{entrada[0]:.0f}% humedad, "
        f"{entrada[1]:.0f} C -> "
        f"{probabilidad:.4f} -> "
        f"{decision}"
    )


print("\nEXPERIMENTOS")

experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas epocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Mas epocas", 20000, 0.5),
    ("Tasa pequena", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0)
]

condicion = np.array([[45, 34]], dtype=float)

condicion_normalizada = condicion / escala


for nombre, numero_epocas, tasa in experimentos:

    (
        pesos_exp,
        sesgo_exp,
        error_exp,
        probabilidades_exp,
        respuestas_exp
    ) = entrenar(tasa, numero_epocas)

    correctas_exp = np.sum(
        respuestas_exp.ravel() == y.ravel()
    )

    probabilidad_45_34 = sigmoide(
        condicion_normalizada @ pesos_exp + sesgo_exp
    )[0, 0]

    print(
        nombre,
        "| epocas:", numero_epocas,
        "| tasa:", tasa,
        "| error:", f"{error_exp:.6f}",
        "| correctas:", f"{correctas_exp}/10",
        "| P(45%,34 C):", f"{probabilidad_45_34:.4f}"
    )


print("\nPRUEBA DE UMBRALES")

for umbral in [0.4, 0.5, 0.6]:

    respuestas_umbral = (
        probabilidades >= umbral
    ).astype(int)

    correctas_umbral = np.sum(
        respuestas_umbral.ravel() == y.ravel()
    )

    print(
        "Umbral:",
        umbral,
        "| Correctas:",
        f"{correctas_umbral}/10"
    )


assert np.array_equal(
    respuestas,
    y.astype(int)
)

print("\nVERIFICACION COMPLETADA")

print("La neurona clasifico correctamente los 10 datos.")
