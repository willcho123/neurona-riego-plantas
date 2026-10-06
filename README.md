Neurona artificial para el riego de plantas
Descripción

Este proyecto implementa una neurona artificial en Python capaz de determinar si una planta necesita ser regada utilizando dos variables de entrada:

Humedad del suelo, expresada como porcentaje entre 0 y 100.

Temperatura ambiental, expresada en grados Celsius.

La salida de la neurona es binaria:

0: no regar.

1: regar.

Los datos utilizados son didácticos y no representan una recomendación agronómica para una especie real.

El proyecto fue desarrollado utilizando Python, NumPy y uv.

Objetivo

El objetivo es comprender el funcionamiento de una neurona artificial y el papel de:

Las entradas.

Los pesos.

El sesgo.

La normalización.

La función sigmoide.

La tasa de aprendizaje.

Las épocas de entrenamiento.

El umbral de clasificación.

Tecnologías utilizadas

Python 3.12

NumPy

uv

Estructura del proyecto
neurona-riego-plantas/
├── main.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
└── .venv/


La carpeta .venv no debe publicarse en GitHub.

Instalación

Se requiere tener Python y uv instalados.

Para instalar las dependencias del proyecto:

uv sync

Ejecución

El programa puede ejecutarse mediante:

uv run python main.py

Datos de entrenamiento

Se utilizaron diez ejemplos:

Caso	Humedad	Temperatura	Salida
1	80 %	18 °C	0
2	70 %	22 °C	0
3	65 %	28 °C	0
4	55 %	25 °C	0
5	50 %	32 °C	0
6	40 %	30 °C	1
7	35 %	25 °C	1
8	30 %	32 °C	1
9	20 %	35 °C	1
10	10 %	38 °C	1
Normalización

Las variables tienen escalas diferentes. La humedad puede llegar aproximadamente a 100 %, mientras que la temperatura se trabaja aproximadamente hasta 50 °C.

Por esta razón se utilizó:

escala = np.array([100, 50], dtype=float)
X_normalizado = X / escala


La humedad se divide entre 100 y la temperatura entre 50.

La misma escala se utiliza posteriormente para las condiciones nuevas. Esto permite que las dos entradas tengan valores en una escala cercana a 0 y 1.

La variable X_normalizado se utiliza durante el entrenamiento en:

z = X_normalizado @ pesos + sesgo


y:

gradiente_pesos = X_normalizado.T @ gradiente_z


El arreglo original X se conserva para mostrar los datos en sus unidades originales y facilitar la interpretación de los resultados.

Función de activación

La neurona utiliza la función sigmoide:

def sigmoide(z):
    return 1 / (1 + np.exp(-z))


La sigmoide transforma la suma ponderada en una probabilidad entre 0 y 1.

Después se utiliza un umbral de 0.5:

respuesta = (probabilidad >= 0.5).astype(int)


Si la probabilidad es mayor o igual a 0.5, la salida es 1 y se decide regar. Si es menor que 0.5, la salida es 0 y se decide no regar.

Entrenamiento base

La configuración inicial fue:

Tasa de aprendizaje: 0.5

Épocas: 10000

Los resultados obtenidos fueron:

Parámetro	Resultado
Peso de humedad	-19.503197
Peso de temperatura	2.310111
Sesgo	7.402159
Error final	0.019645
Respuestas correctas	10/10
Interpretación de los pesos

El peso correspondiente a la humedad fue:

-19.503197


Su signo negativo indica que, según los datos utilizados, una mayor humedad disminuye la probabilidad de que la neurona indique que es necesario regar.

El peso correspondiente a la temperatura fue:

2.310111


Su signo positivo indica que, según los datos utilizados, una mayor temperatura aumenta la probabilidad de que la neurona indique que es necesario regar.

Estos resultados son coherentes con la lógica planteada para este ejercicio didáctico.

Resultados de los datos de entrenamiento
Humedad	Temperatura	Esperado	Probabilidad	Respuesta
80 %	18 °C	0	0.0006	0
70 %	22 °C	0	0.0053	0
65 %	28 °C	0	0.0183	0
55 %	25 °C	0	0.1025	0
50 %	32 °C	0	0.2951	0
40 %	30 °C	1	0.7285	1
35 %	25 °C	1	0.8496	1
30 %	32 °C	1	0.9539	1
20 %	35 °C	1	0.9941	1
10 %	38 °C	1	0.9993	1

La neurona obtuvo:

10/10 respuestas correctas

Predicciones con condiciones nuevas

Después del entrenamiento se probaron cinco condiciones nuevas.

Humedad	Temperatura	Probabilidad	Decisión
75 %	30 °C	0.0029	NO REGAR
45 %	34 °C	0.5490	REGAR
25 %	22 °C	0.9719	REGAR
50 %	25 °C	0.2325	NO REGAR
30 %	40 °C	0.9677	REGAR

La condición de 45 % de humedad y 34 °C obtuvo una probabilidad de 0.5490, por lo que supera el umbral de 0.5 y la neurona determina que se debe regar.

Experimentos con los parámetros

Se realizaron ocho experimentos modificando las épocas o la tasa de aprendizaje.

Experimento	Épocas	Tasa	Error final	Correctas	P(45 %, 34 °C)
Prueba base	10000	0.5	0.019645	10/10	0.5490
Pocas épocas	100	0.5	0.165623	10/10	0.5217
Cantidad intermedia	1000	0.5	0.067481	10/10	0.5793
Más épocas	20000	0.5	0.011705	10/10	0.5263
Tasa pequeña	10000	0.01	0.130048	10/10	0.5388
Tasa moderada	10000	0.1	0.049728	10/10	0.5853
Tasa alta	10000	1.0	0.011705	10/10	0.5263
Tasa muy alta	10000	2.0	0.006566	10/10	0.5082
Análisis de los experimentos
1. ¿Por qué fue necesario normalizar la humedad y la temperatura?

Fue necesario porque las variables utilizan escalas diferentes. La humedad puede llegar hasta 100, mientras que la temperatura se trabaja aproximadamente hasta 50.

La normalización permite trabajar con valores en escalas similares y evita que una variable tenga una influencia excesiva debido únicamente a su magnitud numérica.

2. ¿En qué operaciones se utilizó X_normalizado y para qué se conservó X?

X_normalizado se utilizó en la suma ponderada y en el cálculo del gradiente de los pesos.

Se conservó X para mostrar los datos originales en porcentajes de humedad y grados Celsius, facilitando la interpretación de los resultados.

3. ¿Qué ocurrió al utilizar solamente 100 épocas?

Con 100 épocas el error fue 0.165623, considerablemente mayor que el obtenido con 10000 épocas, que fue 0.019645.

Sin embargo, en este conjunto de datos la neurona todavía clasificó correctamente los 10 ejemplos. Esto demuestra que acertar la clasificación binaria no significa necesariamente que el modelo tenga el menor error posible.

4. ¿Más épocas siempre produjeron una mejora importante?

No siempre.

Pasar de 10000 a 20000 épocas redujo el error de 0.019645 a 0.011705, por lo que hubo una mejora, pero no fue tan grande como la mejora obtenida al pasar de pocas épocas a una cantidad mayor.

Esto muestra que después de cierto punto las mejoras pueden ser cada vez menores.

5. ¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña?

Con una tasa de 0.01, el error fue 0.130048.

El aprendizaje fue más lento y la neurona necesitó muchas actualizaciones para acercarse a una solución con menor error.

6. ¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta?

En este conjunto de datos, las tasas de 1.0 y 2.0 lograron reducir bastante el error.

La tasa de 1.0 obtuvo un error de 0.011705 y la tasa de 2.0 obtuvo 0.006566.

Sin embargo, una tasa alta no garantiza que un modelo siempre mejore. En otros problemas puede provocar oscilaciones o inestabilidad. En este caso particular la neurona logró converger correctamente.

7. ¿Qué representa el signo del peso correspondiente a la humedad?

El peso de humedad fue negativo. Esto representa una relación inversa en los datos utilizados: cuando aumenta la humedad, disminuye la probabilidad de que la neurona indique que se debe regar.

8. ¿Qué representa el signo del peso correspondiente a la temperatura?

El peso de temperatura fue positivo. Esto representa una relación directa en los datos utilizados: cuando aumenta la temperatura, aumenta la probabilidad de que la neurona indique que se debe regar.

9. ¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?

La función sigmoide produce una probabilidad continua entre 0 y 1. Para tomar una decisión binaria se necesita establecer un límite.

En este proyecto se utiliza 0.5. Una probabilidad igual o superior a 0.5 se convierte en 1 y una probabilidad menor se convierte en 0.

10. ¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?

Esta neurona es solamente un modelo didáctico. Utiliza dos variables y diez ejemplos artificiales, por lo que no representa todas las condiciones que afectan las necesidades reales de una planta.

En un sistema real también podrían ser importantes la especie de planta, humedad ambiental, radiación solar, tipo de suelo, lluvia, estación del año, velocidad del viento, profundidad de las raíces y características del sistema de riego.

Por lo tanto, las predicciones de este programa no deben utilizarse como una recomendación agronómica real.

Prueba adicional con umbrales

Se utilizaron los mismos pesos aprendidos y solamente se modificó el umbral.

Umbral	Respuestas correctas
0.4	10/10
0.5	10/10
0.6	10/10

Modificar el umbral no cambia los pesos porque los pesos solamente se modifican durante el entrenamiento mediante los gradientes.

El umbral se aplica después del entrenamiento para convertir las probabilidades en decisiones binarias.

En estos diez datos concretos no hubo cambios en las clasificaciones al utilizar los tres umbrales porque las probabilidades obtenidas quedaron suficientemente alejadas de los límites de decisión relevantes.

Conclusiones

La neurona artificial logró aprender una relación entre la humedad del suelo, la temperatura y la necesidad de riego en los datos didácticos proporcionados.

La normalización permitió trabajar con ambas variables en escalas comparables. El peso negativo de la humedad refleja que una mayor humedad reduce la necesidad de riego, mientras que el peso positivo de la temperatura indica una mayor probabilidad de riego cuando aumenta la temperatura.

Los experimentos mostraron que la cantidad de épocas y la tasa de aprendizaje afectan el error y las probabilidades obtenidas, aunque en este conjunto de datos todos los experimentos lograron clasificar correctamente los diez ejemplos.

El proyecto demuestra de forma práctica cómo una neurona artificial puede utilizar entradas, pesos, sesgo, función sigmoide y aprendizaje mediante gradientes para resolver un problema de clasificación binaria.