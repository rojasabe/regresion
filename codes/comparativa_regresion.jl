using Gtk
using Plots
using GLM
using DataFrames
plotlyjs()

const X_3 = [1, 2, 3]
const Y_3 = [2, 4, 5.5]
const X_5 = [1, 2, 3, 4, 5]
const Y_5 = [1.5, 3.0, 4.0, 5.5, 7]
const PENDIENTE_VERDADERO = 1.5
const INTERCEPTO_VERDADERO = 0.5

function ruido_determinista(x)
    return 0.001 * (x - 2) * (x - 4) * (x - 8)
end

const X_100 = collect(range(1, 10, length=100))
const Y_100 = [PENDIENTE_VERDADERO * x + INTERCEPTO_VERDADERO + ruido_determinista(x) for x in X_100]
const X_1000 = collect(range(1, 10, length=1000))
const Y_1000 = [PENDIENTE_VERDADERO * x + INTERCEPTO_VERDADERO + ruido_determinista(x) for x in X_1000]

function modelo_lineal(w,b,x)
    return w * x + b
end

function funcion_costo(w,b, x_datos, y_datos)
    m = length(x_datos)
    suma = sum((modelo_lineal(w,b,x_datos[i]) - y_datos[i])^2 for i in 1:m)
    return suma / (2 * m)
end

function dj_dw(w,b,x_datos,y_datos)
    m = length(x_datos)
    suma = sum(modelo_lineal(w,b,x_datos[i]) - y_datos[i] for i in 1:m)
    return suma / m
end

function dj_db(w,b,x_datos,y_datos)
    m = length(x_datos)
    suma = sum(modelo_lineal(w,b,x_datos[i]) - y_datos[i] for i in 1:m)
    return suma / m
end

function gradiente_descendiente(x_datos, y_datos, alfa, numero_iteraciones)
    w = 0
    b = 0
    historial_costo = Float64[]
    historial_w = Float64[]
    historial_b = float64[]
    for _ in 1:numero_iteraciones
        temporal_w = w - alfa * dj_dw(w,b,x_datos,y_datos)
        temporal_b = b - alfa * dj_db(w,b,x_datos,y_datos)
        w = temporal_w
        b = temporal_b
        push!