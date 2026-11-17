from collections import Counter
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizador = CountVectorizer()
modelo = MultinomialNB()

conceptos_prueba = ["Nómina", "Regalo de abuela", "Vendí zapatos", "Pan", "Videojuego", "Guagua"]

def cargar_movimientos(ruta):
    with open(ruta, encoding="utf-8", newline="") as archivo:
        filas = csv.DictReader(archivo)
        entrenamiento = {"conceptos": [],
                         "categorias": []}

        for fila in filas:
            categoria = fila["categoria"]
            concepto = fila["concepto"]
            if concepto not in conceptos_prueba:
                entrenamiento["conceptos"].append(concepto)
                entrenamiento["categorias"].append(categoria)

    return entrenamiento 

entrenamiento = cargar_movimientos("movimientos.csv")

def elegir_categoria_referencia(categorias_entrenamiento):
    frecuencias = Counter(categorias_entrenamiento)
    candidatas = []
    frecuencia_maxima = max(frecuencias.values())
    for frecuencia in frecuencias.items():
        if frecuencia[1] == frecuencia_maxima:
            candidatas.append(frecuencia[0])

    candidatas_ordenadas = sorted(candidatas)
    return candidatas_ordenadas[0]

def calcular_exactitud(categoria_predicha, categorias_prueba):
    contador_aciertos = 0

    for categoria in categorias_prueba:
        if categoria == categoria_predicha:
            contador_aciertos += 1

    proporcion = contador_aciertos / len(categorias_prueba)

    return proporcion

predicha = elegir_categoria_referencia(entrenamiento["categorias"])

matriz = vectorizador.fit_transform(entrenamiento["conceptos"])

modelo.fit(matriz, entrenamiento["categorias"])

matriz_prueba = vectorizador.transform(conceptos_prueba)

print(modelo.predict(matriz_prueba))