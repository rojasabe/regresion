# regresion_lineal.jl -- gradiente descendiente a mano, sin librerias de ML
# Autor: Porfirio Rojas

using Gtk
using Plots
plotlyjs()   # backend con zoom, hover y leyenda interactiva en el navegador

const X_3 = [1.0,2.0,3.0]
const Y_3 = [2.0,4.0,5.5]   # 3 puntos a mano, todos rondan y ~ 1.5x + 0.5

const X_5 = [1.0,2.0,3.0,4.0,5.0]
const Y_5 = [1.5,3.0,4.0,5.5,7.0]

const X_6 = [1.0,2.0,3.0,4.0,5.0,6.0]
const Y_6 = [2.0,2.5,4.0,5.0,6.0,7.5]

const X_7 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0]
const Y_7 = [1.5,3.0,3.5,5.0,5.5,7.0,8.5]


function modelo(w, b, x)
    return w * x + b   # la recta: w es la pendiente y b es donde corta el eje y
end


function funcion_costo(w, b, x_datos, y_datos)
    m = length(x_datos)   # cuantos puntos tenemos en el dataset
    suma = sum((modelo(w, b, x_datos[i]) - y_datos[i])^2 for i in 1:m)   # suma de (prediccion - real)^2 para cada punto
    return suma / (2.0 * m)   # el 1/2 es un truco: al derivar el 2 del exponente se cancela y queda mas limpio
end

function dj_dw(w, b, x_datos, y_datos)
    m = length(x_datos)
    suma = sum((modelo(w, b, x_datos[i]) - y_datos[i]) * x_datos[i] for i in 1:m)   # va multiplicado por x porque d/dw de (wx+b) = x, regla de la cadena
    return suma / m
end


function dj_db(w, b, x_datos, y_datos)
    m = length(x_datos)
    suma = sum(modelo(w, b, x_datos[i]) - y_datos[i] for i in 1:m)   # sin x porque d/db de (wx+b) = 1, b es constante respecto a x
    return suma / m
end


function gradiente_descendiente(x_datos, y_datos, alfa, num_iter)
    w = 0.0   # arrancamos en el origen, el algoritmo solito encuentra el minimo
    b = 0.0

    hist_costo = Float64[]   # log del costo en cada iteracion, para graficar la curva de convergencia
    hist_w     = Float64[]   # log de w, para ver el camino que recorre sobre la parabola
    hist_b     = Float64[]

    println("="^55)
    println("  entrenamiento con $(length(x_datos)) puntos")
    println("  alfa = $alfa,  iteraciones = $num_iter")
    println("="^55)

    for iter in 1:num_iter
        grad_w = dj_dw(w, b, x_datos, y_datos)   # calculamos las derivadas con los valores actuales ANTES de tocarlos
        grad_b = dj_db(w, b, x_datos, y_datos)

        temp_w = w - alfa * grad_w   # nuevo w calculado con el w y b viejos
        temp_b = b - alfa * grad_b   # nuevo b calculado con el w y b viejos
        w = temp_w   # ahora si pisamos: actualizacion simultanea, si actualizaramos w primero dj_db usaria la w nueva y estaria mal
        b = temp_b

        costo = funcion_costo(w, b, x_datos, y_datos)
        push!(hist_costo, costo)
        push!(hist_w, w)
        push!(hist_b, b)

        if iter == 1 || iter % max(1, num_iter ÷ 10) == 0 || iter == num_iter   # imprimimos solo en iter 1 y cada 10% del total
            println("  iter $iter: w=$(round(w,digits=5)),  b=$(round(b,digits=5)),  j=$(round(costo,digits=6))")
        end
    end

    println("optimo encontrado: f*(x) = $(round(w,digits=4))x + $(round(b,digits=4))")
    println("="^55 * "\n")

    return w, b, hist_costo, hist_w, hist_b
end


function graficar(x_datos, y_datos, w, b, hist_costo, hist_w, hist_b,
                  titulo, paso_recta, max_rectas)

    n       = length(hist_w)
    x_rango = collect(range(minimum(x_datos), maximum(x_datos), length=200))   # 200 puntos para que la recta se vea suave

    y_final = modelo.(w, b, x_rango)   # recta optima final, la que muestra el resultado

    # subplot 1: puntos reales + recta ajustada final
    y_all = vcat(y_datos, y_final)
    pad1y = (maximum(y_all) - minimum(y_all)) * 0.20
    pad1x = (maximum(x_datos) - minimum(x_datos)) * 0.10
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
        label     = "f*(x) = $(round(w,digits=4))x + $(round(b,digits=4))",
        color     = :firebrick,
        linewidth = 2.5
    )
    xlims!(p1, (minimum(x_datos) - pad1x, maximum(x_datos) + pad1x))
    ylims!(p1, (minimum(y_all)   - pad1y, maximum(y_all)   + pad1y))   # los limites los fijamos al final para que plotlyjs no los pise

    # subplot 2: como fue evolucionando la recta durante el entrenamiento
    candidatos = collect(paso_recta:paso_recta:n)
    isempty(candidatos) && (candidatos = [n])
    if length(candidatos) > max_rectas
        paso2      = ceil(Int, length(candidatos) / max_rectas)
        candidatos = candidatos[1:paso2:end]   # submuestreamos para no meter demasiadas rectas en la leyenda
    end

    p2 = scatter(x_datos, y_datos,
        label      = "datos ($(length(x_datos)) puntos)",
        xlabel     = "x",
        ylabel     = "y",
        title      = "evolucion de f(x) = wx + b",
        markersize = 7,
        color      = :steelblue,
        legend     = :outertopright   # fuera del area para no tapar las rectas
    )
    paleta = cgrad(:blues, length(candidatos) + 2)
    for (k, idx) in enumerate(candidatos)
        wk = hist_w[idx]
        bk = hist_b[idx]
        yk = modelo.(wk, bk, x_rango)
        plot!(p2, x_rango, yk,
            label     = "t=$idx",   # t = numero de iteracion en ese momento
            color     = paleta[k / (length(candidatos) + 1)],
            linewidth = 1.5,
            alpha     = 0.85
        )
    end
    plot!(p2, x_rango, y_final,
        label     = "optima t=$n  w*=$(round(w,digits=3))  b*=$(round(b,digits=3))",
        color     = :firebrick,
        linewidth = 2.5,
        linestyle = :dash
    )

    # subplot 3: la parabola j(w) con b fijado en b* -- muestra la superficie de costo y el camino del GD
    w_ini   = hist_w[1]
    margen  = max(abs(w - w_ini) * 1.5, abs(w) * 0.5, 1.0)
    w_eje   = collect(range(min(w_ini, w) - margen * 0.3,
                            max(w_ini, w) + margen * 0.3, length=400))
    j_eje   = [funcion_costo(wv, b, x_datos, y_datos) for wv in w_eje]   # evaluamos j para cada w posible con b fijo

    p3 = plot(w_eje, j_eje,
        label     = "",
        xlabel    = "w",
        ylabel    = "j(w, b*)",
        title     = "superficie de costo j(w)  --  w* = $(round(w,digits=4))",
        color     = :mediumpurple,
        linewidth = 2
    )
    paso_cam = max(1, n ÷ 60)
    idx_cam  = collect(1:paso_cam:n)
    j_cam    = [funcion_costo(hist_w[i], hist_b[i], x_datos, y_datos) for i in idx_cam]
    scatter!(p3, hist_w[idx_cam], j_cam,
        label      = "",   # sin entrada en leyenda para no saturarla
        color      = :darkorange,
        markersize = 4,
        alpha      = 0.75   # un poco transparente para ver la parabola detras
    )
    scatter!(p3, [w], [funcion_costo(w, b, x_datos, y_datos)],
        label       = "w* = $(round(w,digits=4))",
        color       = :green,
        markersize  = 11,
        markershape = :diamond   # diamante verde para que salte a la vista el minimo
    )

    # subplot 4: convergencia -- como baja j con cada iteracion, deberia verse una curva que decae
    j_final = hist_costo[end]
    p4 = plot(1:length(hist_costo), hist_costo,
        label     = "",
        xlabel    = "iteracion",
        ylabel    = "j(w,b)",
        title     = "convergencia del costo  --  j* = $(round(j_final,digits=6))",
        color     = :darkorange,
        linewidth = 2
    )
    scatter!(p4, [length(hist_costo)], [j_final],
        label       = "j* = $(round(j_final,digits=6))",
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
        bottom_margin = 5Plots.mm,
        link          = :none   # cada subplot maneja sus propios ejes, no los sincronizamos
    )
    display(figura)
end


function lanzar_aplicacion()

    ventana = GtkWindow("regresion lineal -- gradiente descendiente", 620, 740)
    caja    = GtkBox(:v)   # caja vertical: todo apilado de arriba a abajo
    set_gtk_property!(caja, :spacing, 8)
    set_gtk_property!(caja, :margin_top,    12)
    set_gtk_property!(caja, :margin_bottom, 12)
    set_gtk_property!(caja, :margin_start,  12)
    set_gtk_property!(caja, :margin_end,    12)

    titulo_lbl = GtkLabel(
        "<b>regresion lineal con gradiente descendiente</b>\n" *
        "modelo: f(w,b)(x) = wx + b\n" *
        "costo:  j(w,b) = (1/2m) * suma[(wx+b - y)^2]"
    )
    set_gtk_property!(titulo_lbl, :use_markup, true)   # para que el <b> del label funcione como negrita
    set_gtk_property!(titulo_lbl, :justify, 2)

    function fila_campo(etiqueta, valor_default, ancho=10)   # helper: crea una fila con label + campo de texto
        fila = GtkBox(:h)
        set_gtk_property!(fila, :spacing, 6)
        lbl  = GtkLabel(etiqueta)
        set_gtk_property!(lbl, :xalign, 0.0f0)
        ent  = GtkEntry()
        set_gtk_property!(ent, :text, valor_default)
        set_gtk_property!(ent, :width_chars, ancho)
        push!(fila, lbl)
        push!(fila, ent)
        return fila, ent
    end

    marco_params = GtkFrame("  parametros del entrenamiento  ")
    caja_params  = GtkBox(:v)
    set_gtk_property!(caja_params, :spacing, 6)
    set_gtk_property!(caja_params, :margin_top,    6)
    set_gtk_property!(caja_params, :margin_bottom, 6)
    set_gtk_property!(caja_params, :margin_start,  8)
    set_gtk_property!(caja_params, :margin_end,    8)

    fila_a, entry_alfa      = fila_campo("learning rate (alfa):              ", "0.01")
    fila_i, entry_iter      = fila_campo("numero de iteraciones:             ", "100")
    fila_p, entry_paso      = fila_campo("graficar recta cada N iteraciones: ", "10")
    fila_r, entry_maxrectas = fila_campo("max rectas en grafica de evolucion:", "5")

    nota = GtkLabel("  alfa recomendado para estos datasets: entre 0.001 y 0.05")
    set_gtk_property!(nota, :xalign, 0.0f0)

    for w in [fila_a, nota, fila_i, fila_p, fila_r]
        push!(caja_params, w)
    end
    push!(marco_params, caja_params)

    marco_pred = GtkFrame("  conjuntos de datos predefinidos (puntos manuales)  ")
    caja_pred  = GtkBox(:v)
    set_gtk_property!(caja_pred, :spacing, 6)
    set_gtk_property!(caja_pred, :margin_top,    6)
    set_gtk_property!(caja_pred, :margin_bottom, 6)
    set_gtk_property!(caja_pred, :margin_start,  8)
    set_gtk_property!(caja_pred, :margin_end,    8)

    boton_3 = GtkButton("3 puntos  : (1,2.0), (2,4.0), (3,5.5)")
    boton_5 = GtkButton("5 puntos  : (1,1.5), (2,3.0), (3,4.0), (4,5.5), (5,7.0)")
    boton_6 = GtkButton("6 puntos  : (1,2.0), (2,2.5), (3,4.0), (4,5.0), (5,6.0), (6,7.5)")
    boton_7 = GtkButton("7 puntos  : (1,1.5), (2,3.0), (3,3.5), (4,5.0), (5,5.5), (6,7.0), (7,8.5)")

    for btn in [boton_3, boton_5, boton_6, boton_7]
        push!(caja_pred, btn)
    end
    push!(marco_pred, caja_pred)

    marco_manual = GtkFrame("  entrada manual (valores separados por coma)  ")
    caja_manual  = GtkBox(:v)
    set_gtk_property!(caja_manual, :spacing, 6)
    set_gtk_property!(caja_manual, :margin_top,    6)
    set_gtk_property!(caja_manual, :margin_bottom, 6)
    set_gtk_property!(caja_manual, :margin_start,  8)
    set_gtk_property!(caja_manual, :margin_end,    8)

    fila_mx, entry_x = fila_campo("valores x: ", "1, 2, 3", 28)
    fila_my, entry_y = fila_campo("valores y: ", "2, 4, 5.5", 28)
    boton_manual     = GtkButton("entrenar con los puntos ingresados")

    push!(caja_manual, fila_mx)
    push!(caja_manual, fila_my)
    push!(caja_manual, boton_manual)
    push!(marco_manual, caja_manual)

    marco_res = GtkFrame("  resultados del entrenamiento  ")
    label_res = GtkLabel("presiona un boton para entrenar el modelo...")
    set_gtk_property!(label_res, :margin_top,    10)
    set_gtk_property!(label_res, :margin_bottom, 10)
    set_gtk_property!(label_res, :margin_start,  10)
    set_gtk_property!(label_res, :margin_end,    10)
    set_gtk_property!(label_res, :xalign, 0.0f0)
    push!(marco_res, label_res)

    for widget in [titulo_lbl, marco_params, marco_pred, marco_manual, marco_res]   # ensamblamos todo en la ventana principal
        push!(caja, widget)
    end
    push!(ventana, caja)

    function leer_params()   # saca los cuatro valores de los campos de texto y los convierte al tipo correcto
        alfa    = parse(Float64, strip(get_gtk_property(entry_alfa,      :text, String)))
        iters   = parse(Int,     strip(get_gtk_property(entry_iter,      :text, String)))
        paso    = parse(Int,     strip(get_gtk_property(entry_paso,      :text, String)))
        maxrec  = parse(Int,     strip(get_gtk_property(entry_maxrectas, :text, String)))
        return alfa, iters, paso, maxrec
    end

    function mostrar_resultado(w, b, costo, n_pts, alfa, iters)   # actualiza el panel de texto con los resultados del entrenamiento
        texto = (
            "entrenamiento con $n_pts puntos completado\n" *
            "  alfa               = $alfa,   iteraciones = $iters\n" *
            "  w* (peso optimo)   = $(round(w,     digits=6))\n" *
            "  b* (sesgo optimo)  = $(round(b,     digits=6))\n" *
            "  j(w*,b*) final     = $(round(costo, digits=8))\n" *
            "  ecuacion ajustada  : f*(x) = $(round(w,digits=4))x + $(round(b,digits=4))"
        )
        set_gtk_property!(label_res, :label, texto)
    end

    function entrenar_y_graficar(X, Y, titulo_graf)   # todo el flujo junto: lee params, entrena, muestra resultado y grafica
        try
            alfa, iters, paso, maxrec = leer_params()
            w, b, hc, hw, hb = gradiente_descendiente(X, Y, alfa, iters)
            costo = funcion_costo(w, b, X, Y)
            mostrar_resultado(w, b, costo, length(X), alfa, iters)
            graficar(X, Y, w, b, hc, hw, hb, titulo_graf, paso, maxrec)
        catch err
            set_gtk_property!(label_res, :label, "error: $(string(err))")   # si algo falla lo mostramos en el panel
        end
    end

    signal_connect(boton_3, "clicked") do _
        entrenar_y_graficar(X_3, Y_3, "regresion lineal -- 3 puntos")
    end

    signal_connect(boton_5, "clicked") do _
        entrenar_y_graficar(X_5, Y_5, "regresion lineal -- 5 puntos")
    end

    signal_connect(boton_6, "clicked") do _
        entrenar_y_graficar(X_6, Y_6, "regresion lineal -- 6 puntos")
    end

    signal_connect(boton_7, "clicked") do _
        entrenar_y_graficar(X_7, Y_7, "regresion lineal -- 7 puntos")
    end

    signal_connect(boton_manual, "clicked") do _
        try
            alfa, iters, paso, maxrec = leer_params()
            xs = parse.(Float64, strip.(split(get_gtk_property(entry_x, :text, String), ',')))   # parseamos la lista de x separada por comas
            ys = parse.(Float64, strip.(split(get_gtk_property(entry_y, :text, String), ',')))
            length(xs) == length(ys) || error("la cantidad de valores x e y debe ser igual")   # validacion basica antes de entrenar
            w, b, hc, hw, hb = gradiente_descendiente(xs, ys, alfa, iters)
            costo = funcion_costo(w, b, xs, ys)
            mostrar_resultado(w, b, costo, length(xs), alfa, iters)
            graficar(xs, ys, w, b, hc, hw, hb, "regresion lineal -- puntos manuales", paso, maxrec)
        catch err
            set_gtk_property!(label_res, :label, "error: $(string(err))")
        end
    end

    showall(ventana)   # mostramos la ventana y entramos al loop de eventos

    if !isinteractive()   # si corremos desde terminal (no REPL) bloqueamos el hilo principal para que la ventana no se cierre sola
        condicion = Condition()
        signal_connect(ventana, "destroy") do _w
            notify(condicion)
        end
        @async Gtk.gtk_main()
        wait(condicion)
    end
end

lanzar_aplicacion()
