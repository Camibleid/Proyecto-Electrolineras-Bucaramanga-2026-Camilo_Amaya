from __future__ import annotations
import networkx as nx
import random
import csv
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import TYPE_CHECKING, Any

np: Any = None
RandomForestClassifier: Any = None
RandomForestRegressor:  Any = None
train_test_split:       Any = None
accuracy_score:         Any = None
mean_absolute_error:    Any = None
r2_score:               Any = None
classification_report:  Any = None
LabelEncoder:           Any = None

try:
    import numpy as np                                               # type: ignore[assignment]
    from sklearn.ensemble import (RandomForestClassifier,            # type: ignore[assignment]
                                  RandomForestRegressor)
    from sklearn.model_selection import train_test_split             # type: ignore[assignment]
    from sklearn.metrics import (accuracy_score,                     # type: ignore[assignment]
                                 classification_report,
                                 mean_absolute_error,
                                 r2_score)
    from sklearn.preprocessing import LabelEncoder                   # type: ignore[assignment]
    ML_DISPONIBLE = True
except BaseException:
    ML_DISPONIBLE = False
ELECTROLINERAS = {
    0: "Homecenter",
    1: "CC Quinta Etapa",
    2: "CC Cacique",
    3: "CC Cañaveral",
    4: "Terpel Piedecuesta",
    5: "Éxito Rosita",
    6: "CC La Florida",
    7: "Promotores del Oriente (Girón)"
}
PUNTOS_FIJOS = {
    8:  "UIS Campus Central",
    9:  "UIS Campus Florida",
    10: "UIS Guatiguará (Piedecuesta)",
    11: "UIS Campus Bucarica",
    12: "CENFER",
    13: "UNAB",
    14: "UTS",
    15: "UPB",
    16: "PTAR Río Frío",
    17: "Sede Catay"
}
VEHICULOS = {
    1: {
        "nombre":          "Mini Aceman E",
        "gama":            "Baja/Media Gama",
        "bateria_kwh":     38.5,
        "consumo_wh_km":   167,
        "autonomia_km":    230,
        "factor_calor":    1.10,   # +10% consumo en calor (AC)
        "factor_lluvia":   1.15,   # +15% en lluvia (resistencia, defrost)
        "factor_tormenta": 1.22,
    },
    2: {
        "nombre":          "Porsche Taycan Plus",
        "gama":            "Alta Gama",
        "bateria_kwh":     97.0,
        "consumo_wh_km":   169,
        "autonomia_km":    575,
        "factor_calor":    1.07,
        "factor_lluvia":   1.12,
        "factor_tormenta": 1.18,
    }
}
TRAFICO_POR_HORA = {
     0: 0.78,  1: 0.74,  2: 0.70,  3: 0.68,  4: 0.72,  5: 0.88,
     6: 1.30,  7: 1.65,  8: 1.75,  9: 1.45, 10: 1.18, 11: 1.22,
    12: 1.38, 13: 1.32, 14: 1.16, 15: 1.24, 16: 1.50, 17: 1.80,
    18: 1.72, 19: 1.48, 20: 1.25, 21: 1.10, 22: 0.95, 23: 0.85
}
CLIMAS = {
    0: {"nombre": "Despejado",  "factor_consumo": 1.00, "factor_trafico": 1.00},
    1: {"nombre": "Lluvia",     "factor_consumo": 1.15, "factor_trafico": 1.25},
    2: {"nombre": "Calor",      "factor_consumo": 1.10, "factor_trafico": 1.05},
    3: {"nombre": "Tormenta",   "factor_consumo": 1.25, "factor_trafico": 1.55},
}
INCIDENTES_POSIBLES = [
    {"tipo": "Accidente vial",   "factor": 2.20, "prob": 0.05},
    {"tipo": "Obra en vía",      "factor": 1.65, "prob": 0.08},
    {"tipo": "Manifestación",    "factor": 3.00, "prob": 0.02},
    {"tipo": "Corte de vía",     "factor": 9.99, "prob": 0.01},  
    {"tipo": "Semáforo dañado",  "factor": 1.35, "prob": 0.07},
    {"tipo": "Vía inundada",     "factor": 5.00, "prob": 0.03},  
]
ARISTAS_DIRIGIDAS = [
    # ── Entre electrolineras ──────────────────────────────────
    (0, 1, 3.2), (1, 0, 3.9),   # Homecenter ↔ CC Quinta Etapa
    (0, 2, 4.5), (2, 0, 5.2),   # Homecenter ↔ CC Cacique
    (0, 7, 9.1), (7, 0, 9.8),   # Homecenter ↔ Promotores Oriente
    (1, 2, 2.8), (2, 1, 3.1),   # CC Quinta Etapa ↔ CC Cacique
    (1, 3, 4.0), (3, 1, 4.7),   # CC Quinta Etapa ↔ CC Cañaveral
    (2, 3, 5.1), (3, 2, 5.6),   # CC Cacique ↔ CC Cañaveral
    (2, 6, 5.5), (6, 2, 6.1),   # CC Cacique ↔ CC La Florida
    (3, 4, 8.7), (4, 3, 9.3),   # CC Cañaveral ↔ Terpel Piedecuesta
    (3, 5, 4.2), (5, 3, 4.9),   # CC Cañaveral ↔ Éxito Rosita
    (5, 6, 6.3), (6, 5, 7.0),   # Éxito Rosita ↔ CC La Florida
    (6, 7, 7.8), (7, 6, 8.4),   # CC La Florida ↔ Promotores Oriente
    (4, 5, 6.5), (5, 4, 7.0),   # Terpel ↔ Éxito Rosita
    # ── Entre puntos fijos ────────────────────────────────────
    (8,  9,  3.5), (9,  8,  4.2),   # UIS Central ↔ UIS Florida
    (8,  11, 2.1), (11, 8,  2.9),   # UIS Central ↔ UIS Bucarica
    (8,  12, 1.8), (12, 8,  2.3),   # UIS Central ↔ CENFER
    (8,  13, 2.4), (13, 8,  3.0),   # UIS Central ↔ UNAB
    (9,  16, 4.2), (16, 9,  4.9),   # UIS Florida ↔ PTAR Río Frío
    (11, 12, 1.5), (12, 11, 2.0),   # UIS Bucarica ↔ CENFER
    (12, 13, 2.2), (13, 12, 2.7),   # CENFER ↔ UNAB
    (13, 14, 3.0), (14, 13, 3.5),   # UNAB ↔ UTS
    (14, 15, 2.7), (15, 14, 3.2),   # UTS ↔ UPB
    (15, 8,  3.3), (8,  15, 3.9),   # UPB ↔ UIS Central
    (16, 17, 5.0), (17, 16, 5.6),   # PTAR Río Frío ↔ Sede Catay
    (17, 9,  4.1), (9,  17, 4.8),   # Sede Catay ↔ UIS Florida
    (10, 16, 4.5), (16, 10, 5.0),   # UIS Guatiguará ↔ PTAR Río Frío
    (8,  14, 4.5), (14, 8,  5.1),   # UIS Central ↔ UTS
    (11, 13, 2.8), (13, 11, 3.3),   # UIS Bucarica ↔ UNAB
    # ── Puntos fijos ↔ Electrolineras ────────────────────────
    (8,  0,  2.5), (0,  8,  3.2),   # UIS Central ↔ Homecenter
    (8,  2,  3.8), (2,  8,  4.4),   # UIS Central ↔ CC Cacique
    (9,  6,  2.9), (6,  9,  3.5),   # UIS Florida ↔ CC La Florida
    (10, 4,  3.1), (4,  10, 3.7),   # UIS Guatiguará ↔ Terpel
    (10, 3,  9.5), (3,  10, 10.3),  # UIS Guatiguará ↔ CC Cañaveral
    (11, 1,  3.4), (1,  11, 4.0),   # UIS Bucarica ↔ CC Quinta Etapa
    (11, 0,  3.0), (0,  11, 3.6),   # UIS Bucarica ↔ Homecenter
    (12, 0,  4.1), (0,  12, 4.8),   # CENFER ↔ Homecenter
    (12, 2,  3.3), (2,  12, 3.9),   # CENFER ↔ CC Cacique
    (13, 2,  3.6), (2,  13, 4.2),   # UNAB ↔ CC Cacique
    (13, 1,  4.2), (1,  13, 4.8),   # UNAB ↔ CC Quinta Etapa
    (14, 3,  4.8), (3,  14, 5.4),   # UTS ↔ CC Cañaveral
    (15, 2,  4.0), (2,  15, 4.6),   # UPB ↔ CC Cacique
    (15, 3,  5.5), (3,  15, 6.1),   # UPB ↔ CC Cañaveral
    (16, 5,  5.2), (5,  16, 5.9),   # PTAR Río Frío ↔ Éxito Rosita
    (16, 7,  6.8), (7,  16, 7.4),   # PTAR Río Frío ↔ Promotores
    (17, 6,  3.7), (6,  17, 4.3),   # Sede Catay ↔ CC La Florida
    (17, 7,  4.5), (7,  17, 5.1),   # Sede Catay ↔ Promotores
]
grafo              = nx.DiGraph()
grafo_configurado  = False
historial_recargas: list[dict[str, Any]] = []  
modelo_rf_clf:  Any = None       
modelo_rf_reg:  Any = None       
modelo_entrenado   = False
def validar_entero_positivo(mensaje: str) -> int:
    while True:
        entrada = input(mensaje).strip()
        if not entrada:
            print("   [!] No ingresó ningún valor. Intente de nuevo.")
            continue
        try:
            valor = int(entrada)
            if valor > 0:
                return valor
            print("   [!] Error: el valor debe ser mayor a cero (no negativo, no cero).")
        except ValueError:
            print("   [!] Error: ingrese únicamente un número entero. Sin letras ni símbolos.")
def validar_rango_entero(mensaje: str, minimo: int, maximo: int) -> int:
    while True:
        entrada = input(mensaje).strip()
        if not entrada:
            print("   [!] No ingresó ningún valor.")
            continue
        try:
            valor = int(entrada)
            if minimo <= valor <= maximo:
                return valor
            print(f"   [!] Error: ingrese un valor entre {minimo} y {maximo}.")
        except ValueError:
            print("   [!] Error: se esperaba un número entero, no letras ni símbolos.")
def validar_flotante_rango(mensaje: str, minimo: float, maximo: float) -> float:
    while True:
        entrada = input(mensaje).strip()
        if not entrada:
            print("   [!] No ingresó ningún valor.")
            continue
        try:
            valor = float(entrada)
            if minimo <= valor <= maximo:
                return valor
            print(f"   [!] Error: ingrese un valor entre {minimo} y {maximo}.")
        except ValueError:
            print("   [!] Error: ingrese un número válido.")
def obtener_factor_trafico(hora: int, clima_id: int) -> float:
    f_hora   = TRAFICO_POR_HORA.get(hora % 24, 1.0)
    f_clima  = CLIMAS[clima_id]["factor_trafico"]
    return round(f_hora * f_clima, 3)
def simular_incidente_en_arista(u: int, v: int, clima_id: int) -> dict | None:
    candidatos = list(INCIDENTES_POSIBLES)
    if clima_id in (1, 3):
        candidatos = [
            {**inc, "prob": inc["prob"] * 2.0} if inc["tipo"] == "Vía inundada"
            else inc
            for inc in candidatos
        ]
    for incidente in candidatos:
        if random.random() < incidente["prob"]:
            return incidente
    return None

def calcular_peso_efectivo(peso_base: float, hora: int,
                           clima_id: int, incidente=None) -> float:
    f_trafico  = obtener_factor_trafico(hora, clima_id)
    f_incidente = incidente["factor"] if incidente else 1.0
    return round(peso_base * f_trafico * f_incidente, 4)
def actualizar_pesos_grafo(hora: int, clima_id: int,
                           con_incidentes: bool = True) -> dict:
    incidentes_detectados = {}
    for u, v, datos in grafo.edges(data=True):
        base      = datos["peso_base"]
        incidente = simular_incidente_en_arista(u, v, clima_id) if con_incidentes else None
        efectivo  = calcular_peso_efectivo(base, hora, clima_id, incidente)
        grafo[u][v]["weight"] = efectivo
        if incidente:
            clave = f"{u}→{v}"
            incidentes_detectados[clave] = incidente["tipo"]
    return incidentes_detectados
def configurar_grafo() -> None:
    global grafo, grafo_configurado
    grafo = nx.DiGraph()
    for nid, nombre in ELECTROLINERAS.items():
        grafo.add_node(nid, nombre=nombre, tipo="electrolinera")
    for nid, nombre in PUNTOS_FIJOS.items():
        grafo.add_node(nid, nombre=nombre, tipo="punto_fijo")
    for u, v, km in ARISTAS_DIRIGIDAS:
        grafo.add_edge(u, v, peso_base=km, weight=km)
    grafo_configurado = True
    es_conexo = nx.is_weakly_connected(grafo)
    print("\n  ✔ Dígrafo G = (V, E) configurado exitosamente.")
    print(f"     • Tipo              : Grafo DIRIGIDO (DiGraph)")
    print(f"     • Nodos totales     : {grafo.number_of_nodes()}")
    print(f"     • Aristas dirigidas : {grafo.number_of_edges()}")
    print(f"     • Electrolineras    : {len(ELECTROLINERAS)}")
    print(f"     • Puntos fijos      : {len(PUNTOS_FIJOS)}")
    print(f"     • Conexo (débil)    : {'Sí' if es_conexo else 'No — revisar aristas'}")
    print(f"\n  ℹ  A→B ≠ B→A: los pesos son asimétricos para reflejar")
    print(f"     calles de un solo sentido y rutas alternativas reales.")

def dijkstra_electrolinera_cercana(nodo_origen: int) -> tuple:
    mejor_dist = float('inf')
    mejor_elec = None
    mejor_ruta = []
    for destino in ELECTROLINERAS:
        if destino == nodo_origen:
            continue
        try:
            dist = nx.dijkstra_path_length(grafo, nodo_origen,
                                           destino, weight='weight')
            ruta = nx.dijkstra_path(grafo, nodo_origen,
                                    destino, weight='weight')
            if dist < mejor_dist:
                mejor_dist = dist
                mejor_elec = destino
                mejor_ruta = ruta
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return mejor_elec, mejor_dist, mejor_ruta
def distancias_a_todas_electrolineras(nodo_origen: int) -> list:
    distancias = []
    for eid in range(len(ELECTROLINERAS)):
        try:
            d = nx.dijkstra_path_length(grafo, nodo_origen,
                                        eid, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            d = 999.0
        distancias.append(round(d, 3))
    return distancias
def info_vehiculos() -> None:
    print("\n" + "="*65)
    print("      VEHÍCULOS ELÉCTRICOS – FICHA TÉCNICA")
    print("      Fuente: ev-database.org | Modelos 2024-2026")
    print("="*65)
    for vid, v in VEHICULOS.items():
        kwh_km = v['consumo_wh_km'] / 1000
        print(f"\n  [{vid}]  {v['nombre']}  ({v['gama']})")
        print(f"       Batería usable     : {v['bateria_kwh']} kWh")
        print(f"       Consumo medio      : {v['consumo_wh_km']} Wh/km "
              f"({kwh_km:.4f} kWh/km)")
        print(f"       Autonomía real     : {v['autonomia_km']} km")
        print(f"       Umbral crítico     : "
              f"{v['bateria_kwh']*0.10:.2f} – "
              f"{v['bateria_kwh']*0.20:.2f} kWh  (10%–20%)")
        print(f"       Factor clima calor : ×{v['factor_calor']}")
        print(f"       Factor clima lluvia: ×{v['factor_lluvia']}")
        print(f"       Factor tormenta    : ×{v['factor_tormenta']}")
def simular_recorridos() -> None:
    global historial_recargas
    if not grafo_configurado:
        print("\n  [!] Configure primero el grafo (Opción 1).")
        return
    n = validar_entero_positivo("\n  Número de recorridos a simular: ")
    print(f"\n{'='*65}")
    print(f"  SIMULACIÓN EN CURSO — {n} recorridos")
    print(f"{'='*65}")
    conteo_elec = {eid: 0 for eid in ELECTROLINERAS}  
    conteo_veh  = {1: 0, 2: 0}
    conteo_clima = {cid: 0 for cid in CLIMAS}
    total_incidentes = 0
    historial_recorrido_actual = []
    ids_puntos = list(PUNTOS_FIJOS.keys())
    ids_todos  = list(grafo.nodes())
    for i in range(1, n + 1):
        origen    = random.choice(ids_puntos)
        destino   = random.choice([nd for nd in ids_todos if nd != origen])
        id_veh    = random.choice([1, 2])
        vehiculo  = VEHICULOS[id_veh]
        hora      = random.randint(0, 23)
        clima_id  = random.choices(
            [0, 1, 2, 3], weights=[0.55, 0.25, 0.15, 0.05]
        )[0]
        clima     = CLIMAS[clima_id]
        bat_ini_pct = random.uniform(12.0, 60.0)
        bat_kwh     = (bat_ini_pct / 100.0) * vehiculo['bateria_kwh']
        kwh_km_base = vehiculo['consumo_wh_km'] / 1000.0
        kwh_km      = kwh_km_base * clima['factor_consumo']
        conteo_clima[clima_id] += 1
        incidentes_detectados = actualizar_pesos_grafo(hora, clima_id,
                                                       con_incidentes=True)
        n_inc = len(incidentes_detectados)
        total_incidentes += n_inc
        try:
            ruta_ppal  = nx.dijkstra_path(grafo, origen, destino,
                                          weight='weight')
            dist_total = nx.dijkstra_path_length(grafo, origen, destino,
                                                 weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue  
        bat_actual       = bat_kwh
        recarga          = False
        nodo_critico     = origen
        bat_critica_pct  = 0.0
        elec_usada_id    = None
        dist_a_carga     = 0.0
        ruta_a_carga     = []
        incidente_enruta = None
        for k in range(len(ruta_ppal) - 1):
            u, v = ruta_ppal[k], ruta_ppal[k + 1]
            clave_seg = f"{u}→{v}"
            tipo_inc = incidentes_detectados.get(clave_seg, None)
            if tipo_inc and not incidente_enruta:
                incidente_enruta = tipo_inc
            dist_seg   = grafo[u][v].get('peso_base', grafo[u][v]['weight'])
            consumo    = kwh_km * dist_seg
            bat_actual -= consumo
            pct_actual  = (bat_actual / vehiculo['bateria_kwh']) * 100.0
            if pct_actual <= 20.0:
                nodo_critico    = v
                bat_critica_pct = max(pct_actual, 0.0)
                recarga         = True
                break
        if recarga:
            elec_id, dist_elec, ruta_elec = dijkstra_electrolinera_cercana(
                nodo_critico
            )
            if elec_id is not None:
                conteo_elec[elec_id] += 1
                conteo_veh[id_veh]   += 1
                elec_usada_id         = elec_id
                dist_a_carga          = dist_elec
                ruta_a_carga          = ruta_elec
                nombres_ruta = " → ".join(
                    grafo.nodes[nd]['nombre'] for nd in ruta_elec
                )
                print(f"\n  ── #{i:03d} | {grafo.nodes[origen]['nombre']}"
                      f" → {grafo.nodes[destino]['nombre']}")
                print(f"   Vehículo : {vehiculo['nombre']} | "
                      f"Hora: {hora:02d}:00 | Clima: {clima['nombre']}")
                if incidente_enruta:
                    print(f"   ⚠  Incidente detectado: {incidente_enruta}")
                print(f"   ⚡ BATERÍA CRÍTICA ({bat_critica_pct:.1f}%) — "
                      f"Electrolinera: {ELECTROLINERAS[elec_id]}")
                print(f"   → {dist_elec:.2f} km | Ruta: {nombres_ruta}")
        else:
            bat_fin = (bat_actual / vehiculo['bateria_kwh']) * 100.0
            if i <= 15 or i % 20 == 0: 
                print(f"  ── #{i:03d} | ✔ OK | "
                      f"{grafo.nodes[origen]['nombre'][:20]} | "
                      f"Bat. final: {bat_fin:.1f}% | "
                      f"{clima['nombre']}")
        dists_elec = distancias_a_todas_electrolineras(origen)
        factor_traf = obtener_factor_trafico(hora, clima_id)
        registro = {
            "recorrido_id":          i,
            "fecha_hora":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vehiculo":              vehiculo['nombre'],
            "tipo_vehiculo_id":      id_veh,
            "gama":                  vehiculo['gama'],
            "origen":                grafo.nodes[origen]['nombre'],
            "origen_id":             origen,
            "destino":               grafo.nodes[destino]['nombre'],
            "destino_id":            destino,
            "hora":                  hora,
            "es_hora_pico":          int(hora in range(6, 10) or hora in range(16, 20)),
            "clima":                 clima['nombre'],
            "clima_id":              clima_id,
            "factor_trafico":        factor_traf,
            "incidente_en_ruta":     incidente_enruta or "Ninguno",
            "distancia_total_km":    round(dist_total, 2),
            "bateria_inicio_pct":    round(bat_ini_pct, 1),
            "bateria_critica_pct":   round(bat_critica_pct, 1),
            "recarga_realizada":     recarga,
            "electrolinera_usada":   ELECTROLINERAS.get(elec_usada_id, "N/A")
                                     if elec_usada_id is not None else "N/A",
            "electrolinera_id":      elec_usada_id if elec_usada_id is not None else -1,
            "distancia_a_carga_km":  round(dist_a_carga, 2),
            "dist_elec_0": dists_elec[0], "dist_elec_1": dists_elec[1],
            "dist_elec_2": dists_elec[2], "dist_elec_3": dists_elec[3],
            "dist_elec_4": dists_elec[4], "dist_elec_5": dists_elec[5],
            "dist_elec_6": dists_elec[6], "dist_elec_7": dists_elec[7],
        }
        historial_recargas.append(registro)
        historial_recorrido_actual.append(registro)
    total_recargas = sum(conteo_veh.values())
    print(f"\n{'='*65}")
    print("  RESUMEN DE SIMULACIÓN")
    print(f"{'='*65}")
    print(f"  Recorridos totales       : {n}")
    print(f"  Total recargas           : {total_recargas}")
    print(f"  Incidentes viales        : {total_incidentes}")
    print(f"\n  Recargas por vehículo:")
    for vid, cnt in conteo_veh.items():
        print(f"   • {VEHICULOS[vid]['nombre']:<30}: {cnt}")
    print(f"\n  Condiciones climáticas:")
    for cid, cnt in conteo_clima.items():
        print(f"   • {CLIMAS[cid]['nombre']:<15}: {cnt} recorridos")
    print(f"\n  Ranking de electrolineras:")
    ranking = sorted(conteo_elec.items(), key=lambda x: x[1], reverse=True)
    for pos, (eid, cnt) in enumerate(ranking[:5], 1):
        barra = "█" * min(cnt, 25)
        print(f"   {pos}. {ELECTROLINERAS[eid]:<38} {cnt:>3}  {barra}")
def calcular_ruta_manual() -> None:
    if not grafo_configurado:
        print("\n  [!] Configure primero el grafo (Opción 1).")
        return
    print("\n" + "="*65)
    print("  PUNTOS FIJOS DISPONIBLES (ORIGEN):")
    for pid, nombre in PUNTOS_FIJOS.items():
        print(f"   [{pid}] {nombre}")
    origen = validar_rango_entero("\n  Origen [8-17]: ", 8, 17)
    print("\n  HORA DEL DÍA (afecta el tráfico):")
    hora = validar_rango_entero("  Ingrese la hora [0-23]: ", 0, 23)
    print("\n  CONDICIÓN CLIMÁTICA:")
    for cid, c in CLIMAS.items():
        print(f"   [{cid}] {c['nombre']}  (tráfico ×{c['factor_trafico']},"
              f" consumo ×{c['factor_consumo']})")
    clima_id = validar_rango_entero("  Seleccione clima [0-3]: ", 0, 3)
    print("\n  VEHÍCULO:")
    for vid, v in VEHICULOS.items():
        print(f"   [{vid}] {v['nombre']:<22} ({v['bateria_kwh']} kWh | {v['consumo_wh_km']} Wh/km)")
    id_veh  = validar_rango_entero("  Vehículo [1-2]: ", 1, 2)
    vehiculo = VEHICULOS[id_veh]
    print("\n  NIVEL DE BATERÍA ACTUAL (%):")
    bat_pct = validar_flotante_rango("  Batería actual [1.0-100.0]: ", 1.0, 100.0)
    incidentes = actualizar_pesos_grafo(hora, clima_id, con_incidentes=True)
    f_trafico = obtener_factor_trafico(hora, clima_id)
    elec_id, distancia, ruta = dijkstra_electrolinera_cercana(origen)
    if elec_id is None:
        print("   [!] No se encontró ruta a ninguna electrolinera.")
        return
    nombres_ruta = " → ".join(grafo.nodes[nd]['nombre'] for nd in ruta)
    print(f"\n{'='*65}")
    print(f"  RESULTADO")
    print(f"{'='*65}")
    print(f"  Desde         : {PUNTOS_FIJOS[origen]}")
    print(f"  Hora          : {hora:02d}:00 | Tráfico ×{f_trafico}")
    print(f"  Clima         : {CLIMAS[clima_id]['nombre']}")
    print(f"  Batería       : {bat_pct:.1f}%")
    if incidentes:
        print(f"  ⚠ Incidentes detectados: {len(incidentes)}")
        for k, v in list(incidentes.items())[:3]:
            print(f"     • Arista {k}: {v}")
    kwh_km = vehiculo['consumo_wh_km'] / 1000 * CLIMAS[clima_id]['factor_consumo']
    consumo_est   = round(kwh_km * distancia, 3)
    bat_restante  = bat_pct - (consumo_est / vehiculo['bateria_kwh'] * 100)
    print(f"\n  ✔ Electrolinera más cercana : {ELECTROLINERAS[elec_id]}")
    print(f"    Distancia (efectiva)       : {distancia:.2f} km")
    print(f"    Ruta óptima (Dijkstra)     : {nombres_ruta}")
    print(f"")
    print(f"  Estimación de consumo ({vehiculo['nombre']}):")
    print(f"    Consumo estimado           : {consumo_est:.3f} kWh")
    sufic = " ⚠ BATERÍA INSUFICIENTE" if bat_restante < 0 else ""
    print(f"    Batería al llegar          : {max(bat_restante, 0):.1f}%{sufic}")
    print(f"\n  Ranking de todas las electrolineras desde tu posición:")
    dists_todas = []
    for eid in ELECTROLINERAS:
        try:
            d = nx.dijkstra_path_length(grafo, origen, eid, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            d = float('inf')
        dists_todas.append((eid, d))
    dists_todas.sort(key=lambda x: x[1])
    for pos, (eid, d) in enumerate(dists_todas, 1):
        marca = " ← más cercana" if eid == elec_id else ""
        d_str = f"{d:.2f} km" if d != float('inf') else "sin ruta disponible"
        print(f"   {pos}. {ELECTROLINERAS[eid]:<40} {d_str}{marca}")
def entrenar_modelo_ia() -> None:
    global modelo_rf_clf, modelo_rf_reg, modelo_entrenado
    if not ML_DISPONIBLE:
        print("\n  [!] scikit-learn no está instalado.")
        print("      Ejecuta: pip install scikit-learn numpy")
        return
    datos_recarga = [r for r in historial_recargas if r["recarga_realizada"]]
    if len(datos_recarga) < 20:
        print(f"\n  [!] Datos insuficientes para entrenar.")
        print(f"      Tienes {len(datos_recarga)} recarga(s). "
              f"Necesitas al menos 20.")
        print(f"      Ejecuta más simulaciones (Opción 3).")
        return
    print(f"\n  Entrenando con {len(datos_recarga)} eventos de recarga...")
    FEATURES = [
        "origen_id", "tipo_vehiculo_id", "bateria_inicio_pct",
        "hora", "es_hora_pico", "clima_id", "factor_trafico",
        "dist_elec_0", "dist_elec_1", "dist_elec_2", "dist_elec_3",
        "dist_elec_4", "dist_elec_5", "dist_elec_6", "dist_elec_7",
    ]
    X = np.array([[r[f] for f in FEATURES] for r in datos_recarga])
    y_clf = np.array([r["electrolinera_id"] for r in datos_recarga])
    y_reg = np.array([r["distancia_a_carga_km"] for r in datos_recarga])
    X_train, X_test, y_clf_tr, y_clf_te = train_test_split(  # type: ignore[misc]
        X, y_clf, test_size=0.20, random_state=42, stratify=y_clf
        if len(set(y_clf)) > 1 else None
    )
    _, _, y_reg_tr, y_reg_te = train_test_split(  # type: ignore[misc]
        X, y_reg, test_size=0.20, random_state=42
    )
    modelo_rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    modelo_rf_clf.fit(X_train, y_clf_tr)
    pred_clf = modelo_rf_clf.predict(X_test)
    acc = accuracy_score(y_clf_te, pred_clf)
    modelo_rf_reg = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42
    )
    modelo_rf_reg.fit(X_train, y_reg_tr)
    pred_reg = modelo_rf_reg.predict(X_test)
    mae = mean_absolute_error(y_reg_te, pred_reg)
    r2  = r2_score(y_reg_te, pred_reg)
    modelo_entrenado = True
    print(f"\n{'='*65}")
    print("  RESULTADOS DEL ENTRENAMIENTO")
    print(f"{'='*65}")
    print(f"  Muestras totales     : {len(datos_recarga)}")
    print(f"  Train / Test split   : 80% / 20%")
    print(f"\n  [Clasificador — electrolinera predicha]")
    print(f"   Accuracy            : {acc*100:.1f}%")

    print(f"\n  [Regresor — distancia a electrolinera]")
    print(f"   MAE                 : {mae:.3f} km")
    print(f"   R² Score            : {r2:.3f}")
    importancias = sorted(
        zip(FEATURES, modelo_rf_clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  Top 5 variables más importantes (clasificador):")
    for feat, imp in importancias[:5]:
        barra = "█" * int(imp * 50)
        print(f"   {feat:<30} {imp:.4f}  {barra}")
def predecir_electrolinera() -> None:
    global modelo_rf_clf, modelo_rf_reg
    if not ML_DISPONIBLE:
        print("\n  [!] scikit-learn no disponible.")
        return
    if not modelo_entrenado:
        print("\n  [!] El modelo no ha sido entrenado todavía.")
        print("      Entrénalo con la Opción 5.")
        return
    if not grafo_configurado:
        print("\n  [!] Configure primero el grafo (Opción 1).")
        return
    print("\n  Ingresa el estado del vehículo para la predicción:")
    print("\n  PUNTOS FIJOS (ORIGEN):")
    for pid, nombre in PUNTOS_FIJOS.items():
        print(f"   [{pid}] {nombre}")
    origen_id = validar_rango_entero("  Origen [8-17]: ", 8, 17)
    print("\n  VEHÍCULO: [1] Mini Aceman E   [2] Porsche Taycan Plus")
    tipo_veh = validar_rango_entero("  Vehículo [1-2]: ", 1, 2)
    bat_pct = validar_flotante_rango(
        "  Nivel de batería actual [1-100]: ", 1.0, 100.0
    )
    hora    = validar_rango_entero("  Hora del día [0-23]: ", 0, 23)
    clima_id = validar_rango_entero(
        "  Clima (0=Despejado 1=Lluvia 2=Calor 3=Tormenta) [0-3]: ", 0, 3
    )
    actualizar_pesos_grafo(hora, clima_id, con_incidentes=False)
    dists_elec  = distancias_a_todas_electrolineras(origen_id)
    f_trafico   = obtener_factor_trafico(hora, clima_id)
    es_hora_pico = int(hora in range(6, 10) or hora in range(16, 20))

    FEATURES = [
        "origen_id", "tipo_vehiculo_id", "bateria_inicio_pct",
        "hora", "es_hora_pico", "clima_id", "factor_trafico",
        "dist_elec_0", "dist_elec_1", "dist_elec_2", "dist_elec_3",
        "dist_elec_4", "dist_elec_5", "dist_elec_6", "dist_elec_7",
    ]
    valores = [
        origen_id, tipo_veh, bat_pct,
        hora, es_hora_pico, clima_id, f_trafico,
        *dists_elec
    ]
    X_pred = np.array([valores])
    pred_elec_id = modelo_rf_clf.predict(X_pred)[0]
    pred_dist    = modelo_rf_reg.predict(X_pred)[0]
    proba        = modelo_rf_clf.predict_proba(X_pred)[0]
    clases       = modelo_rf_clf.classes_
    print(f"\n{'='*65}")
    print(f"  PREDICCIÓN DEL MODELO (Random Forest)")
    print(f"{'='*65}")
    elec_pred_key = int(pred_elec_id) if pred_elec_id is not None else -1
    print(f"  Electrolinera predicha  : {ELECTROLINERAS.get(elec_pred_key, 'N/A')}")
    print(f"  Distancia estimada      : {pred_dist:.2f} km")
    print(f"\n  Probabilidades por electrolinera:")
    sorted_proba = sorted(zip(clases, proba), key=lambda x: x[1], reverse=True)
    for eid, p in sorted_proba[:5]:
        barra = "█" * int(p * 40)
        print(f"   {ELECTROLINERAS.get(eid, '?'):<40} {p*100:.1f}%  {barra}")
def exportar_datos() -> None:
    if not historial_recargas:
        print("\n  [!] No hay datos. Ejecute primero la simulación (Opción 3).")
        return
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_csv = f"electrolineras_{ts}.csv"
    archivo_json = f"electrolineras_{ts}.json"
    columnas = list(historial_recargas[0].keys())
    with open(archivo_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(historial_recargas)
    with open(archivo_json, mode='w', encoding='utf-8') as f:
        json.dump(historial_recargas, f, ensure_ascii=False,
                  indent=4, default=str)
    print(f"\n  ✔ Archivos exportados exitosamente:")
    print(f"   • CSV  → {archivo_csv}  ({len(historial_recargas)} registros)")
    print(f"   • JSON → {archivo_json}  ({len(historial_recargas)} registros)")
    print(f"\n  Estos archivos sirven como dataset para entrenar")
    print(f"  el modelo supervisado (Opción 5).")
def leer_archivos() -> None:
    archivos = sorted([
        f for f in os.listdir('.')
        if f.startswith('electrolineras_')
        and (f.endswith('.csv') or f.endswith('.json'))
    ])
    if not archivos:
        print("\n  [!] No se encontraron archivos generados previamente.")
        return
    print("\n  Archivos disponibles:")
    for idx, nombre in enumerate(archivos, 1):
        tam = os.path.getsize(nombre)
        print(f"   [{idx}] {nombre}  ({tam:,} bytes)")
    sel     = validar_rango_entero(
        f"\n  Seleccione archivo [1-{len(archivos)}]: ", 1, len(archivos)
    )
    archivo = archivos[sel - 1]
    if archivo.endswith('.csv'):
        with open(archivo, 'r', encoding='utf-8') as f:
            registros = list(csv.DictReader(f))
        print(f"\n  ── {archivo}  ({len(registros)} registros) ──")
        print(f"  {'#':<5} {'Vehículo':<22} {'Hora':<6} "
              f"{'Clima':<12} {'Recarga':<8} {'Electrolinera'}")
        print("  " + "-"*75)
        for r in registros[:20]:
            print(f"  {r['recorrido_id']:<5} {r['vehiculo']:<22} "
                  f"{r['hora']:<6} {r['clima']:<12} "
                  f"{r['recarga_realizada']:<8} {r['electrolinera_usada']}")
        if len(registros) > 20:
            print(f"  … y {len(registros)-20} registros más.")
    elif archivo.endswith('.json'):
        with open(archivo, 'r', encoding='utf-8') as f:
            registros = json.load(f)
        print(f"\n  ── {archivo}  ({len(registros)} registros) ──")
        for r in registros[:15]:
            estado = "⚡ RECARGA" if r['recarga_realizada'] else "✔ OK"
            print(f"  #{r['recorrido_id']:03d}  {r['vehiculo']:<22}"
                  f"  {estado:<12}  → {r['electrolinera_usada']}")
        if len(registros) > 15:
            print(f"  … y {len(registros)-15} registros más.")
def ver_estadisticas() -> None:
    if not historial_recargas:
        print("\n  [!] Sin datos. Ejecute la simulación primero.")
        return
    total = len(historial_recargas)
    recargas = [r for r in historial_recargas if r['recarga_realizada']]
    n_rec = len(recargas)
    print(f"\n{'='*65}")
    print("  ESTADÍSTICAS GLOBALES DEL SISTEMA")
    print(f"{'='*65}")
    print(f"  Recorridos totales     : {total}")
    print(f"  Recargas efectuadas    : {n_rec}  ({n_rec/total*100:.1f}%)")
    print(f"  Sin recarga            : {total-n_rec}  ({(total-n_rec)/total*100:.1f}%)")
    veh_cnt = defaultdict(int)
    for r in recargas:
        veh_cnt[r['vehiculo']] += 1
    print(f"\n  Recargas por vehículo:")
    for veh, cnt in sorted(veh_cnt.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {veh:<30}: {cnt}")
    clima_cnt = defaultdict(int)
    for r in recargas:
        clima_cnt[r['clima']] += 1
    print(f"\n  Recargas por clima:")
    for clim, cnt in sorted(clima_cnt.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {clim:<15}: {cnt}")
    pico_cnt    = sum(1 for r in recargas if r.get('es_hora_pico', 0))
    no_pico_cnt = n_rec - pico_cnt
    print(f"\n  Recargas en hora pico  : {pico_cnt}")
    print(f"  Recargas fuera de pico : {no_pico_cnt}")
    elec_cnt = defaultdict(int)
    for r in recargas:
        elec_cnt[r['electrolinera_usada']] += 1
    print(f"\n  Ranking de electrolineras más utilizadas:")
    ranking = sorted(elec_cnt.items(), key=lambda x: x[1], reverse=True)
    for pos, (nombre, cnt) in enumerate(ranking, 1):
        barra = "█" * min(cnt, 30)
        print(f"   {pos}. {nombre:<40} {cnt:>3}  {barra}")
    inc_cnt = sum(
        1 for r in historial_recargas
        if r.get('incidente_en_ruta', 'Ninguno') != 'Ninguno'
    )
    print(f"\n  Incidentes viales registrados: {inc_cnt}")
    if recargas:
        dist_prom = sum(r['distancia_a_carga_km'] for r in recargas) / n_rec
        print(f"  Distancia promedio a electrolinera: {dist_prom:.2f} km")
def entrenar_desde_csv() -> None:
    global modelo_rf_clf, modelo_rf_reg, modelo_entrenado
    if not ML_DISPONIBLE:
        print("\n  [!] scikit-learn no disponible.")
        return
    archivos = sorted([
        f for f in os.listdir('.')
        if f.startswith('electrolineras_') and f.endswith('.csv')
    ])
    if not archivos:
        print("\n  [!] No hay archivos CSV. Exporta datos primero (Opción 7).")
        return
    print("\n  Archivos CSV disponibles:")
    for idx, a in enumerate(archivos, 1):
        tam = os.path.getsize(a)
        print(f"   [{idx}] {a}  ({tam:,} bytes)")
    sel = validar_rango_entero(f"  Seleccione [1-{len(archivos)}]: ", 1, len(archivos))
    with open(archivos[sel - 1], 'r', encoding='utf-8') as f:
        todos = list(csv.DictReader(f))
    datos = []
    for r in todos:
        if r['recarga_realizada'] == 'True' and int(r['electrolinera_id']) >= 0:
            try:
                datos.append({
                    "origen_id":            int(r['origen_id']),
                    "tipo_vehiculo_id":     int(r['tipo_vehiculo_id']),
                    "bateria_inicio_pct":   float(r['bateria_inicio_pct']),
                    "hora":                 int(r['hora']),
                    "es_hora_pico":         int(r['es_hora_pico']),
                    "clima_id":             int(r['clima_id']),
                    "factor_trafico":       float(r['factor_trafico']),
                    "dist_elec_0":          float(r['dist_elec_0']),
                    "dist_elec_1":          float(r['dist_elec_1']),
                    "dist_elec_2":          float(r['dist_elec_2']),
                    "dist_elec_3":          float(r['dist_elec_3']),
                    "dist_elec_4":          float(r['dist_elec_4']),
                    "dist_elec_5":          float(r['dist_elec_5']),
                    "dist_elec_6":          float(r['dist_elec_6']),
                    "dist_elec_7":          float(r['dist_elec_7']),
                    "electrolinera_id":     int(r['electrolinera_id']),
                    "distancia_a_carga_km": float(r['distancia_a_carga_km']),
                })
            except (ValueError, KeyError):
                continue
    if len(datos) < 30:
        print(f"\n  [!] Solo {len(datos)} recargas en el archivo.")
        print(f"      Necesitas al menos 30. Ejecuta más simulaciones.")
        return
    print(f"\n  Entrenando con {len(datos)} registros del archivo CSV...")
    FEATURES = [
        "origen_id", "tipo_vehiculo_id", "bateria_inicio_pct",
        "hora", "es_hora_pico", "clima_id", "factor_trafico",
        "dist_elec_0", "dist_elec_1", "dist_elec_2", "dist_elec_3",
        "dist_elec_4", "dist_elec_5", "dist_elec_6", "dist_elec_7",
    ]
    X     = np.array([[d[f] for f in FEATURES] for d in datos])
    y_clf = np.array([d["electrolinera_id"] for d in datos])
    y_reg = np.array([d["distancia_a_carga_km"] for d in datos])
    clases_unicas = list(set(y_clf))
    estratificar  = len(clases_unicas) > 1 and min(
        sum(y_clf == c) for c in clases_unicas
    ) >= 2
    X_train, X_test, y_clf_tr, y_clf_te = train_test_split(
        X, y_clf, test_size=0.20, random_state=42,
        stratify=y_clf if estratificar else None
    )
    _, _, y_reg_tr, y_reg_te = train_test_split(
        X, y_reg, test_size=0.20, random_state=42
    )
    modelo_rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=1,
        max_features='sqrt', random_state=42, class_weight='balanced'
    )
    modelo_rf_clf.fit(X_train, y_clf_tr)
    modelo_rf_reg = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=1, random_state=42
    )
    modelo_rf_reg.fit(X_train, y_reg_tr)
    acc = accuracy_score(y_clf_te, modelo_rf_clf.predict(X_test))
    pred_reg = modelo_rf_reg.predict(X_test)
    mae = mean_absolute_error(y_reg_te, pred_reg)
    r2  = r2_score(y_reg_te, pred_reg)
    modelo_entrenado = True
    print(f"\n{'='*65}")
    print("  ENTRENAMIENTO DESDE CSV")
    print(f"{'='*65}")
    print(f"  Registros usados     : {len(datos)}")
    print(f"  Train / Test         : 80% / 20%")
    print(f"\n  [Clasificador]  Accuracy : {acc*100:.1f}%")
    print(f"  [Regresor]      MAE      : {mae:.3f} km  |  R² : {r2:.3f}")
    importancias = sorted(
        zip(FEATURES, modelo_rf_clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  Top 5 features más importantes:")
    for feat, imp in importancias[:5]:
        barra = "█" * int(imp * 60)
        print(f"   {feat:<30} {imp:.4f}  {barra}")
def analizar_centralidad_red() -> None:
    if not grafo_configurado:
        print("\n  [!] Configure primero el grafo (Opción 1).")
        return
    print(f"\n{'='*65}")
    print("  ANÁLISIS DE CENTRALIDAD DE LA RED")
    print(f"{'='*65}")
    todos = {**ELECTROLINERAS, **PUNTOS_FIJOS}
    bt = nx.betweenness_centrality(grafo, weight="weight", normalized=True)
    ranking_bt = sorted(bt.items(), key=lambda x: x[1], reverse=True)
    print("\n  ► Betweenness Centrality (normalizada) — cuellos de botella:")
    for pos, (nid, val) in enumerate(ranking_bt[:10], 1):
        icono = "⚡" if nid in ELECTROLINERAS else "📍"
        barra = "█" * int(val * 60)
        print(f"   {pos:2}. {icono} {todos[nid]:<38} {val:.4f}  {barra}")
    nodo_crit = ranking_bt[0][0]
    print(f"\n     → Nodo más crítico de la red: {todos[nodo_crit]}")
    print("\n  ► Closeness Centrality — electrolinera más accesible:")
    try:
        cl = nx.closeness_centrality(grafo, distance="weight")
        ranking_cl = sorted(
            ((nid, val) for nid, val in cl.items() if nid in ELECTROLINERAS),
            key=lambda x: x[1], reverse=True
        )
        for pos, (nid, val) in enumerate(ranking_cl, 1):
            barra = "█" * int(val * 80)
            print(f"   {pos}. {ELECTROLINERAS[nid]:<40} {val:.4f}  {barra}")
        print(f"\n     → Más céntrica: {ELECTROLINERAS[ranking_cl[0][0]]}")
    except Exception as exc:
        print(f"   [!] {exc}")
    print("\n  ► Grado de entrada ponderado (demanda potencial):")
    grados = {
        nid: round(sum(d.get("peso_base", 0) for _, _, d in grafo.in_edges(nid, data=True)), 1)
        for nid in ELECTROLINERAS
    }
    for pos, (nid, g) in enumerate(sorted(grados.items(), key=lambda x: x[1], reverse=True), 1):
        barra = "█" * int(g / 4)
        print(f"   {pos}. {ELECTROLINERAS[nid]:<40} {g:>6.1f} km  {barra}")
def benchmark_dijkstra_vs_floyd() -> None:
    if not grafo_configurado:
        print("\n  [!] Configure primero el grafo (Opción 1).")
        return
    import time
    V = grafo.number_of_nodes()
    E = grafo.number_of_edges()
    print(f"\n{'='*65}")
    print("  BENCHMARK: DIJKSTRA vs FLOYD-WARSHALL")
    print(f"{'='*65}")
    print(f"  V = {V} nodos  |  E = {E} aristas")
    print(f"  Floyd-Warshall   O(V³)         = {V**3:,} operaciones")
    d_ops = int(V * (E + V) * (V ** 0.5))
    print(f"  Dijkstra × V     O(V(E+V)logV) ≈ {d_ops:,} operaciones")
    t0 = time.perf_counter()
    for nodo in grafo.nodes():
        try:
            nx.single_source_dijkstra_path_length(grafo, nodo, weight="weight")
        except Exception:
            pass
    t_dijk = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    try:
        fw_dist, _ = nx.floyd_warshall_predecessor_and_distance(grafo, weight="weight")
        t_fw = (time.perf_counter() - t0) * 1000
        ganador = "Dijkstra (todos los pares)" if t_dijk <= t_fw else "Floyd-Warshall"
        print(f"\n  Tiempo Dijkstra (todos pares) : {t_dijk:.3f} ms")
        print(f"  Tiempo Floyd-Warshall         : {t_fw:.3f} ms")
        print(f"  Algoritmo más rápido          : {ganador}")
        print(f"\n  ℹ  En OSMnx (miles de nodos) Dijkstra es hasta 100× más rápido.")
        print(f"  Por eso el sistema usa Dijkstra para rutas en tiempo real.")
    except Exception as exc:
        print(f"\n  Dijkstra (todos pares) : {t_dijk:.3f} ms")
        print(f"  Floyd-Warshall         : Error — {exc}")
_VIZ_HTML_FILENAME = "visualizador_electrolineras.html"
def visualizar_grafo() -> None:
    import webbrowser
    candidatos = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), _VIZ_HTML_FILENAME),
        os.path.join(os.getcwd(), _VIZ_HTML_FILENAME),
        _VIZ_HTML_FILENAME,
    ]
    archivo_html = None
    for ruta in candidatos:
        if os.path.isfile(ruta):
            archivo_html = ruta
            break
    if archivo_html is None:
        print(f"\n  [!] No se encontró '{_VIZ_HTML_FILENAME}' en el directorio.")
        print(f"  Asegúrese de que ese archivo esté junto al script .py.")
        print(f"  Rutas buscadas:")
        for r in candidatos:
            print(f"     • {r}")
        return
    url = f"file://{os.path.abspath(archivo_html)}"
    print(f"\n  ✔ Abriendo visualizador interactivo en el navegador...")
    print(f"     Archivo: {os.path.abspath(archivo_html)}")
    print(f"\n  Instrucciones de uso:")
    print(f"   • Clic en un nodo VERDE (punto fijo) → calcula ruta Dijkstra animada")
    print(f"   • Desliza el control de HORA → actualiza pesos de tráfico en tiempo real")
    print(f"   • Botón CLIMA → cambia factores de congestión y consumo")
    print(f"   • Botón VORONOI → muestra zonas de cobertura por electrolinera")
    print(f"   • Botón CENTRALIDAD → tamaño de nodo ∝ betweenness centrality")
    print(f"   • Botón SIMULAR VEHÍCULO → anima un EV recorriendo la ruta")
    print(f"   • Hover sobre nodos/aristas → tooltips con distancias y métricas")
    try:
        webbrowser.open(url)
        print(f"\n  (El navegador se abrió automáticamente.)")
    except Exception as exc:
        print(f"  [!] No se pudo abrir automáticamente: {exc}")
        print(f"  Copie y pegue esta URL en su navegador:")
        print(f"     {url}")
def mostrar_menu() -> None:
    print("\n" + "="*65)
    print("  ELECTROLINERAS BUCARAMANGA  |  Proyecto Integrador 2026-1")
    print("  Ingeniería en Ciencia de Datos – UIS")
    print("="*65)
    estado_grafo = "✔ Configurado" if grafo_configurado else "✘ No configurado"
    estado_ml    = "✔ Entrenado"   if modelo_entrenado  else "✘ Sin entrenar"
    n_datos      = len(historial_recargas)
    print(f"  Grafo: {estado_grafo} | Modelo IA: {estado_ml} | "
          f"Datos: {n_datos} registros")
    print("="*65)
    print("  1. Configurar Dígrafo  (red vial dirigida, ponderada)")
    print("  2. Información de Vehículos")
    print("  3. Simular Recorridos  (tráfico + clima + incidentes)")
    print("  4. Calcular Ruta Manual  (Dijkstra con condiciones)")
    print("  5. Entrenar Modelo IA  (Random Forest)")
    print("  5b. Entrenar desde CSV  (dataset acumulado — mejor accuracy)")
    print("  6. Predecir con Modelo IA  (sin ejecutar Dijkstra)")
    print("  7. Exportar Datos  (.csv y .json)")
    print("  8. Archivos / Estadísticas / Visualizador")
    print("  9. Salir")
    print("="*65)
def main() -> None:
    opcion = 0
    print("\n  Bienvenido al Sistema de Electrolineras — Bucaramanga")
    print("  Configure el grafo primero (Opción 1) para comenzar.\n")
    while opcion != 9:   
        mostrar_menu()
        entrada = input("  Seleccione una opción [1-9]: ").strip()
        if entrada.lower() == '5b':
            entrenar_desde_csv()
            opcion = 0
            continue
        try:
            opcion = int(entrada)
        except ValueError:
            print("\n  [!] Error: ingrese solo un número (1-9) o '5b'.")
            opcion = 0
            continue
        if opcion < 1 or opcion > 9:
            print("\n  [!] Opción fuera de rango. Ingrese entre 1 y 9.")
            opcion = 0
            continue
        if   opcion == 1: configurar_grafo()
        elif opcion == 2: info_vehiculos()
        elif opcion == 3: simular_recorridos()
        elif opcion == 4: calcular_ruta_manual()
        elif opcion == 5: entrenar_modelo_ia()
        elif opcion == 6: predecir_electrolinera()
        elif opcion == 7: exportar_datos()
        elif opcion == 8:
            print("\n  [a] Leer archivos      [b] Estadísticas globales")
            print("  [c] Visualizar grafo   [d] Centralidad de la red")
            print("  [e] Benchmark Dijkstra vs Floyd-Warshall")
            sub = input("  Seleccione [a-e]: ").strip().lower()
            if   sub == 'a': leer_archivos()
            elif sub == 'b': ver_estadisticas()
            elif sub == 'c': visualizar_grafo()
            elif sub == 'd': analizar_centralidad_red()
            elif sub == 'e': benchmark_dijkstra_vs_floyd()
            else: print("  Opción no reconocida.")
        elif opcion == 9:
            print("\n  ¡Hasta pronto! Sistema cerrado correctamente.")
            print(f"  Sesión con {len(historial_recargas)} recorridos registrados.\n")
if __name__ == "__main__":
    main()
    