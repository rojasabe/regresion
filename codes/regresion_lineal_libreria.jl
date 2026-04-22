# regresion_lineal_libreria.jl
# regresion lineal con GLM.jl usando error cuadratico minimo (OLS)
# Autor: Porfirio Rojas

using Gtk
using Plots
using GLM
using DataFrames
plotlyjs()   # backend con zoom y hover en el navegador, igual que en la version a mano

# los mismos datasets que la version a mano, pero solo 3 y 5 puntos
const X_3 = [1.0, 2.0, 3.0]
const Y_3 = [2.0, 4.0, 5.5]

const X_5 = [1.0, 2.0, 3.0, 4.0, 5.0]
const Y_5 = [1.5, 3.0, 4.0, 5.5, 7.0]


# ajuste del modelo con GLM
# lm resuelve internamente las ecuaciones normales y nos entrega
# el w y b que minimizan el error cuadratico, sin iterar nada
function ajustar(x_datos, y_datos)
    df = DataFrame(x = x_datos, y = y_datos)
    modelo = lm(@formula(y ~ x), df)
    coefs = coef(modelo)
    b = coefs[1]   # intercepto
    w = coefs[2]   # pendiente
    return w, b
end


# error cuadratico medio, el mismo J(w,b) que minimizamos a mano
function error_cuadratico(w, b, x_datos, y_datos)
    m = length(x_datos)
    suma = sum((w * x_datos[i] + b - y_datos[i])^2 for i in 1:m)
    return suma / (2.0 * m)
end


# GLM llega al optimo en un solo paso, asi que para ver como seria el
# "camino de aprendizaje" simulamos gradiente descendiente aparte.
# esto nos sirve solo para las graficas de evolucion, parabola y convergencia
function simular_aprendizaje(x_datos, y_datos, alfa, num_iter)
    w = 0.0
    b = 0.0
    m = length(x_datos)
    hist_w = Float64[]
    hist_b = Float64[]
    hist_costo = Float64[]
    for iter in 1:num_iter
        grad_w = sum((w * x_datos[i] + b - y_datos[i]) * x_datos[i] for i in 1:m) / m
        grad_b = sum(w * x_datos[i] + b - y_datos[i] for i in 1:m) / m
        w = w - alfa * grad_w
        b = b - alfa * grad_b
        push!(hist_w, w)
        push!(hist_b, b)
        push!(hist_costo, error_cuadratico(w, b, x_datos, y_datos))
    end
    return hist_w, hist_b, hist_costo
end


# grafica: las mismas 4 vistas que la version a mano
# la recta optima viene de GLM, el camino viene de la simulacion
function graficar(x_datos, y_datos, w_opt, b_opt, hist_w, hist_b, hist_costo, titulo)
    n       = length(hist_w)
    x_rango = collect(range(minimum(x_datos), maximum(x_datos), length=200))
    y_final = w_opt .* x_rango .+ b_opt

    # subplot 1: puntos + recta optima de GLM
    p1 = scatter(x_datos, y_datos,
        label      = "datos ($(length(x_datos)) puntos)",
        xlabel     = "x",
        ylabel     = "y",
        title      = titulo,
        markersize = 7,
        color      = :steelblue,
        legend     = :topleft
    )
    plot!(p1, x_rango, y_final,
        label     = "GLM: f*(x) = $(round(w_opt, digits=4))x + $(round(b_opt, digits=4))",
        color     = :firebrick,
        linewidth = 2.5
    )

    # subplot 2: evolucion de las rectas mientras "aprendia"
    paso_recta = 10
    max_rectas = 8
    candidatos = collect(paso_recta:paso_recta:n)
    isempty(candidatos) && (candidatos = [n])
    if length(candidatos) > max_rectas
        paso2      = ceil(Int, length(candidatos) / max_rectas)
        candidatos = candidatos[1:paso2:end]
    end

    p2 = scatter(x_datos, y_datos,
        label      = "datos ($(length(x_datos)) puntos)",
        xlabel     = "x",
        ylabel     = "y",
        title      = "evolucion de f(x) = wx + b",
        markersize = 7,
        color      = :steelblue,
        legend     = :outertopright
    )
    paleta = cgrad(:blues, length(candidatos) + 2)
    for (k, idx) in enumerate(candidatos)
        wk = hist_w[idx]
        bk = hist_b[idx]
        yk = wk .* x_rango .+ bk
        plot!(p2, x_rango, yk,
            label     = "t=$idx",
            color     = paleta[k / (length(candidatos) + 1)],
            linewidth = 1.5,
            alpha     = 0.85
        )
    end
    plot!(p2, x_rango, y_final,
        label     = "optima (GLM)  w*=$(round(w_opt,digits=3))  b*=$(round(b_opt,digits=3))",
        color     = :firebrick,
        linewidth = 2.5,
        linestyle = :dash
    )

    # subplot 3: parabola J(w) con b fijado en b_opt y el camino naranja
    w_ini  = hist_w[1]
    margen = max(abs(w_opt - w_ini) * 1.5, abs(w_opt) * 0.5, 1.0)
    w_eje  = collect(range(min(w_ini, w_opt) - margen * 0.3,
                           max(w_ini, w_opt) + margen * 0.3, length=400))
    j_eje  = [error_cuadratico(wv, b_opt, x_datos, y_datos) for wv in w_eje]

    p3 = plot(w_eje, j_eje,
        label     = "",
        xlabel    = "w",
        ylabel    = "j(w, b*)",
        title     = "superficie de costo j(w)  --  w* = $(round(w_opt, digits=4))",
        color     = :mediumpurple,
        linewidth = 2
    )
    paso_cam = max(1, n ÷ 60)
    idx_cam  = collect(1:paso_cam:n)
    j_cam    = [error_cuadratico(hist_w[i], hist_b[i], x_datos, y_datos) for i in idx_cam]
    scatter!(p3, hist_w[idx_cam], j_cam,
        label      = "",
        color      = :darkorange,
        markersize = 4,
        alpha      = 0.75
    )
    scatter!(p3, [w_opt], [error_cuadratico(w_opt, b_opt, x_datos, y_datos)],
        label       = "w* = $(round(w_opt, digits=4))",
        color       = :green,
        markersize  = 11,
        markershape = :diamond
    )

    # subplot 4: convergencia del costo sobre las iteraciones simuladas
    j_final = hist_costo[end]
    p4 = plot(1:n, hist_costo,
        label     = "",
        xlabel    = "iteracion",
        ylabel    = "j(w,b)",
        title     = "convergencia del costo  --  j* = $(round(j_final, digits=6))",
        color     = :darkorange,
        linewidth = 2
    )
    scatter!(p4, [n], [j_final],
        label       = "j* = $(round(j_final, digits=6))",
        color       = :green,
        markersize  = 9,
        markershape = :diamond
    )

    figura = plot(p1, p2, p3, p4,
        layout        = (2, 2),
        size          = (1500, 950),
        left_margin   = 8Plots.mm,
        right_margin  = 8Plots.mm,
        top_margin    = 5Plots.mm,
        bottom_margin = 5Plots.mm
    )
    display(figura)
end


function lanzar_aplicacion()

    ventana = GtkWindow("regresion lineal -- GLM (error cuadratico minimo)", 520, 340)
    caja    = GtkBox(:v)
    set_gtk_property!(caja, :spacing,       10)
    set_gtk_property!(caja, :margin_top,    12)
    set_gtk_property!(caja, :margin_bottom, 12)
    set_gtk_property!(caja, :margin_start,  12)
    set_gtk_property!(caja, :margin_end,    12)

    titulo_lbl = GtkLabel(
        "<b>regresion lineal con GLM.jl</b>\n" *
        "metodo: minimos cuadrados ordinarios (OLS)\n" *
        "modelo: f(x) = wx + b"
    )
    set_gtk_property!(titulo_lbl, :use_markup, true)
    set_gtk_property!(titulo_lbl, :justify,    2)

    boton_3 = GtkButton("3 puntos : (1,2.0), (2,4.0), (3,5.5)")
    boton_5 = GtkButton("5 puntos : (1,1.5), (2,3.0), (3,4.0), (4,5.5), (5,7.0)")

    label_res = GtkLabel("presiona un boton para ajustar el modelo...")
    set_gtk_property!(label_res, :margin_top, 10)
    set_gtk_property!(label_res, :xalign,     0.0f0)

    for widget in [titulo_lbl, boton_3, boton_5, label_res]
        push!(caja, widget)
    end
    push!(ventana, caja)

    # todo el flujo junto: GLM da el optimo, la simulacion da el camino
    function ejecutar(X, Y, titulo)
        w_opt, b_opt = ajustar(X, Y)
        hist_w, hist_b, hist_costo = simular_aprendizaje(X, Y, 0.05, 100)
        j = error_cuadratico(w_opt, b_opt, X, Y)
        texto = (
            "ajuste con $(length(X)) puntos completado\n" *
            "  w (pendiente)     = $(round(w_opt, digits=6))\n" *
            "  b (intercepto)    = $(round(b_opt, digits=6))\n" *
            "  error J(w,b)      = $(round(j,     digits=8))\n" *
            "  ecuacion ajustada : f(x) = $(round(w_opt, digits=4))x + $(round(b_opt, digits=4))"
        )
        set_gtk_property!(label_res, :label, texto)
        graficar(X, Y, w_opt, b_opt, hist_w, hist_b, hist_costo, titulo)
    end

    signal_connect(boton_3, "clicked") do _
        ejecutar(X_3, Y_3, "regresion lineal (GLM) -- 3 puntos")
    end

    signal_connect(boton_5, "clicked") do _
        ejecutar(X_5, Y_5, "regresion lineal (GLM) -- 5 puntos")
    end

    showall(ventana)

    if !isinteractive()
        condicion = Condition()
        signal_connect(ventana, "destroy") do _
            notify(condicion)
        end
        @async Gtk.gtk_main()
        wait(condicion)
    end
end

lanzar_aplicacion()
