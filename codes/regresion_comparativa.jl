# regresion_comparativa.jl -- comparativa entre gradiente descendiente (a mano) y OLS (GLM.jl)
# corre los dos metodos sobre el mismo dataset y muestra que tan cerca llega el GD
# del minimo exacto que entrega OLS. la idea: misma data, mismas formulas, mismos
# botones -- la unica fuente de diferencia es el metodo de ajuste.
# para los datasets sinteticos ademas conocemos los parametros verdaderos (a, b) que
# generaron los puntos, asi que podemos comparar cada metodo contra LA VERDAD y declarar
# ganador en cada coeficiente, no solo medir cuanto se diferencian GD y OLS entre si.
# Autor: Porfirio Rojas

using Gtk
using Plots
using GLM
using DataFrames
plotlyjs()   # backend con zoom y hover en el navegador

# datasets a mano (coinciden con lineal_regresion_guinativa.jl y regresion_glm.jl):
const X_3 = [1.0, 2.0, 3.0]
const Y_3 = [2.0, 4.0, 5.5]

const X_5 = [1.0, 2.0, 3.0, 4.0, 5.0]
const Y_5 = [1.5, 3.0, 4.0, 5.5, 7.0]

# ============================================================
# datasets sinteticos deterministas. los parametros VERDADEROS estan definidos
# como const para usarlos en (a) la generacion de y, y (b) la comparacion del
# panel "errores vs la verdad". asi sabemos exactamente que numero deberia salir
# y quien (GD u OLS) se acerca mas a ese objetivo predefinido.
# ============================================================

const A_VERDADERO = 1.5    # pendiente real
const B_VERDADERO = 0.5    # intercepto real

# ruido determinista usando UNICAMENTE  +, -, *  (sin mod, sin sin, sin cos, sin nada).
# es un polinomio cubico con raices en x = 2, 4, 8:
#   ruido(x) = 0.001 * (x - 2) * (x - 4) * (x - 8)
# para x en [1, 10] el maximo absoluto se da en x = 10 (valor 96), asi:
#   |ruido(x)| < 0.001 * 96 = 0.096 < 0.1  -> sin outliers por construccion.
# NO usa rand() ni semilla aleatoria, es 100% reproducible.
# como el polinomio NO es antisimetrico (la raiz central no esta en el medio de
# [1,10]), tiene media y covarianza con x distintas de cero. eso sesga LIGERAMENTE
# la estimacion de OLS: w_OLS no es exactamente 1.5 y b_OLS no es exactamente 0.5.
# el GD converge hacia ese sesgo. con iters moderadas (~1000) el GD no termina de
# llegar a OLS y, por casualidad numerica, puede quedar MAS cerca de la verdad
# que OLS en algun coeficiente -- de ahi salen las comparaciones interesantes
# donde GD gana en w y OLS gana en b (o vicversa, depende del signo del sesgo).
function ruido_det(x)
    return 0.001 * (x - 2.0) * (x - 4.0) * (x - 8.0)
end

# el patron de ruido es el mismo para 100, 1000 y 10000 puntos (la misma funcion
# continua muestreada mas densamente), asi la comparacion entre tamaños de dataset
# es justa: la unica variable es n. permite ver el escalado de tiempo GD vs OLS limpio.
const X_100   = collect(range(1.0, 10.0, length =   100))
const Y_100   = [A_VERDADERO * x + B_VERDADERO + ruido_det(x) for x in X_100]

const X_1000  = collect(range(1.0, 10.0, length =  1000))
const Y_1000  = [A_VERDADERO * x + B_VERDADERO + ruido_det(x) for x in X_1000]

const X_10000 = collect(range(1.0, 10.0, length = 10000))
const Y_10000 = [A_VERDADERO * x + B_VERDADERO + ruido_det(x) for x in X_10000]


# ============================================================
# bloque 1: gradiente descendiente "a mano"
# (IDENTICO matematicamente al de lineal_regresion_guinativa.jl: misma firma,
# mismas formulas, mismo loop, mismo punto de arranque (0, 0). la unica razon
# por la que existe aca duplicado en vez de importarse desde el otro archivo
# es para mantener el estilo monolitico del proyecto y poder leer un archivo
# de una sola pasada.)
# ============================================================

function modelo(w, b, x)
    return w * x + b
end

function funcion_costo(w, b, x_datos, y_datos)
    m = length(x_datos)
    suma = sum((modelo(w, b, x_datos[i]) - y_datos[i])^2 for i in 1:m)
    return suma / (2.0 * m)   # OLS minimiza la misma suma pero sin el /(2m). el optimo coincide
end

function dj_dw(w, b, x_datos, y_datos)
    m = length(x_datos)
    suma = sum((modelo(w, b, x_datos[i]) - y_datos[i]) * x_datos[i] for i in 1:m)
    return suma / m
end

function dj_db(w, b, x_datos, y_datos)
    m = length(x_datos)
    suma = sum(modelo(w, b, x_datos[i]) - y_datos[i] for i in 1:m)
    return suma / m
end

function gradiente_descendiente(x_datos, y_datos, alfa, num_iter)
    w = 0.0
    b = 0.0
    hist_costo = Float64[]
    hist_w     = Float64[]
    hist_b     = Float64[]

    for _ in 1:num_iter
        # actualizacion simultanea: temp_w y temp_b se calculan con los w,b viejos
        # antes de pisarlos. si actualizaramos w primero, dj_db usaria la w nueva y estaria mal.
        temp_w = w - alfa * dj_dw(w, b, x_datos, y_datos)
        temp_b = b - alfa * dj_db(w, b, x_datos, y_datos)
        w = temp_w
        b = temp_b
        push!(hist_costo, funcion_costo(w, b, x_datos, y_datos))
        push!(hist_w, w)
        push!(hist_b, b)
    end

    return w, b, hist_costo, hist_w, hist_b
end


# ============================================================
# bloque 1.5: estandarizacion de features para el GD
# OLS no la necesita (es invariante a transformaciones lineales de x). el GD si,
# y mucho: con x en distinto rango el GD necesita muchisimo mas iteraciones para
# converger -- a veces incluso con num_iter razonables no llega nunca. todas las
# referencias (Ng/CS229, Geron capitulo 4, Raschka FAQ, blog de Akshay sobre
# Julia) lo enfatizan como la diferencia practica mas importante entre los dos
# metodos. desde la GUI se toggle con un checkbox para verlo en vivo.
# ============================================================

function estandarizar_x(x_datos)
    # z-score: despues de la transformacion, mean(x_std) = 0 y std(x_std) = 1.
    # devolvemos mu y sigma para poder revertir los coeficientes al final.
    m     = length(x_datos)
    mu    = sum(x_datos) / m
    var   = sum((xi - mu)^2 for xi in x_datos) / m
    sigma = sqrt(var)
    x_std = [(xi - mu) / sigma for xi in x_datos]
    return x_std, mu, sigma
end

function desestandarizar_coef(w_std, b_std, mu, sigma)
    # si entrenamos en escala estandarizada con  x_std = (x - mu) / sigma
    # y obtenemos  y = w_std * x_std + b_std, podemos volver a la escala original:
    #   y = w_std * (x - mu) / sigma + b_std
    #   y = (w_std / sigma) * x + (b_std - w_std * mu / sigma)
    # de ahi las dos formulas:
    w_orig = w_std / sigma
    b_orig = b_std - w_std * mu / sigma
    return w_orig, b_orig
end


# ============================================================
# bloque 2: OLS via GLM.jl  (sin cambios -- es siempre un solo paso)
# ============================================================

function ajustar_ols(x_datos, y_datos)
    df = DataFrame(x = x_datos, y = y_datos)
    return lm(@formula(y ~ x), df)
end

function rss_de(w_val, b_val, x_datos, y_datos)
    # RSS sin /(2m). lo usamos para dibujar la parabola del panel 3 en la misma
    # escala que el panel de OLS, de manera que el camino del GD y el minimo OLS
    # vivan en el mismo eje vertical.
    return sum((w_val .* x_datos .+ b_val .- y_datos) .^ 2)
end


# ============================================================
# bloque 3: graficar la comparativa -- 4 paneles en 2x2
# ============================================================

function graficar_comparativa(x_datos, y_datos,
                              w_gd, b_gd, hist_costo, hist_w,
                              w_ols, b_ols, rss_ols,
                              titulo, alfa, iters)

    n       = length(hist_w)
    x_rango = collect(range(minimum(x_datos), maximum(x_datos), length = 200))
    y_gd    = modelo.(w_gd,  b_gd,  x_rango)
    y_ols   = modelo.(w_ols, b_ols, x_rango)

    # ---- p1: datos + las dos rectas superpuestas ----
    # alpha bajo y markersize chico porque con 10k puntos se apilan demasiado.
    y_all = vcat(y_datos, y_gd, y_ols)
    pad1y = (maximum(y_all) - minimum(y_all)) * 0.20
    pad1x = (maximum(x_datos) - minimum(x_datos)) * 0.10
    p1 = scatter(x_datos, y_datos,
        label      = "datos ($(length(x_datos)) puntos)",
        xlabel     = "x",
        ylabel     = "y",
        title      = titulo,
        markersize = 4,
        color      = :steelblue,
        legend     = :topleft,
        alpha      = 0.5
    )
    plot!(p1, x_rango, y_gd,
        label     = "GD : y = $(round(w_gd,  digits = 4))x + $(round(b_gd,  digits = 4))",
        color     = :firebrick,
        linewidth = 2.5
    )
    plot!(p1, x_rango, y_ols,
        label     = "OLS: y = $(round(w_ols, digits = 4))x + $(round(b_ols, digits = 4))",
        color     = :darkgreen,
        linewidth = 2.5,
        linestyle = :dash
    )
    xlims!(p1, (minimum(x_datos) - pad1x, maximum(x_datos) + pad1x))
    ylims!(p1, (minimum(y_all)   - pad1y, maximum(y_all)   + pad1y))

    # ---- p2: diferencia |y_GD(x) - y_OLS(x)| a lo largo del rango ----
    dif = abs.(y_gd .- y_ols)
    p2 = plot(x_rango, dif,
        label     = "|y_GD(x) - y_OLS(x)|",
        xlabel    = "x",
        ylabel    = "diferencia absoluta entre rectas",
        title     = "separacion entre las dos rectas ajustadas",
        color     = :darkorange,
        linewidth = 2,
        legend    = :topleft
    )

    # ---- p3: parabola RSS(w) con b fijado en b_OLS + camino del GD + minimo OLS ----
    margen = max(abs(w_ols) * 0.8, abs(w_ols - hist_w[1]) * 1.2, 1.0)
    w_eje  = collect(range(w_ols - margen, w_ols + margen, length = 400))
    j_eje  = [rss_de(wv, b_ols, x_datos, y_datos) for wv in w_eje]
    p3 = plot(w_eje, j_eje,
        label     = "",
        xlabel    = "w",
        ylabel    = "RSS(w, b_OLS)",
        title     = "superficie de costo: camino del GD vs minimo OLS",
        color     = :mediumpurple,
        linewidth = 2,
        legend    = :topright
    )
    paso_cam  = max(1, n ÷ 60)
    idx_cam   = collect(1:paso_cam:n)
    j_gd_path = [rss_de(hist_w[i], b_ols, x_datos, y_datos) for i in idx_cam]
    scatter!(p3, hist_w[idx_cam], j_gd_path,
        label      = "camino del GD",
        color      = :darkorange,
        markersize = 4,
        alpha      = 0.75
    )
    scatter!(p3, [w_ols], [rss_ols],
        label       = "OLS exacto  w* = $(round(w_ols, digits = 4))",
        color       = :darkgreen,
        markersize  = 11,
        markershape = :diamond
    )
    scatter!(p3, [w_gd], [rss_de(w_gd, b_ols, x_datos, y_datos)],
        label       = "GD final  w = $(round(w_gd, digits = 4))",
        color       = :firebrick,
        markersize  = 9,
        markershape = :star5
    )

    # ---- p4: convergencia del J del GD con linea de referencia del costo de OLS ----
    j_ols = rss_ols / (2.0 * length(x_datos))
    p4 = plot(1:length(hist_costo), hist_costo,
        label     = "J del GD",
        xlabel    = "iteracion",
        ylabel    = "J(w,b) = RSS / (2m)",
        title     = "convergencia del GD vs OLS  --  alfa = $alfa, iters = $iters",
        color     = :darkorange,
        linewidth = 2,
        legend    = :topright
    )
    hline!(p4, [j_ols],
        label     = "J*  (referencia OLS) = $(round(j_ols, digits = 6))",
        color     = :darkgreen,
        linewidth = 2,
        linestyle = :dash
    )
    scatter!(p4, [length(hist_costo)], [hist_costo[end]],
        label       = "J final GD = $(round(hist_costo[end], digits = 6))",
        color       = :firebrick,
        markersize  = 8,
        markershape = :star5
    )

    figura = plot(p1, p2, p3, p4,
        layout        = (2, 2),
        size          = (1500, 950),
        left_margin   = 8Plots.mm,
        right_margin  = 8Plots.mm,
        top_margin    = 5Plots.mm,
        bottom_margin = 5Plots.mm,
        link          = :none
    )
    display(figura)
end


# ============================================================
# bloque 4: la GUI
# ============================================================

function lanzar_aplicacion()

    # warmup JIT: corremos cada metodo una vez con datos triviales ANTES de mostrar
    # la GUI. asi cuando el usuario aprieta el primer boton ya esta todo compilado
    # y los tiempos son representativos desde la primera medicion (sin overhead de
    # compilacion JIT de Julia). esto resuelve el clasico "el primer click siempre
    # tarda mas" en Julia y hace que el cronometro sea defendible desde la corrida 1.
    let X_warm = [1.0, 2.0, 3.0], Y_warm = [1.0, 2.0, 3.0]
        gradiente_descendiente(X_warm, Y_warm, 0.01, 10)
        ajustar_ols(X_warm, Y_warm)
        estandarizar_x(X_warm)
    end

    ventana = GtkWindow("comparativa  GD vs OLS  --  gradiente descendiente vs GLM.jl", 820, 880)
    caja    = GtkBox(:v)
    set_gtk_property!(caja, :spacing,       8)
    set_gtk_property!(caja, :margin_top,    12)
    set_gtk_property!(caja, :margin_bottom, 12)
    set_gtk_property!(caja, :margin_start,  12)
    set_gtk_property!(caja, :margin_end,    12)

    titulo_lbl = GtkLabel(
        "<b>comparativa  GD (a mano)  vs  OLS (GLM.jl)</b>\n" *
        "modelo : y = wx + b   --   misma data, dos metodos de ajuste\n" *
        "GD : iterativo, depende de alfa e iteraciones   --   OLS : cerrado, un solo paso\n" *
        "datasets sinteticos generados con  a = $A_VERDADERO,  b = $B_VERDADERO  (puntaje a igualar)"
    )
    set_gtk_property!(titulo_lbl, :use_markup, true)
    set_gtk_property!(titulo_lbl, :justify,    2)

    function fila_campo(etiqueta, valor_default, ancho = 12)
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

    marco_params = GtkFrame("  hiperparametros del GD  (OLS no los usa, su ajuste es siempre el mismo)  ")
    caja_params  = GtkBox(:v)
    set_gtk_property!(caja_params, :spacing,       6)
    set_gtk_property!(caja_params, :margin_top,    6)
    set_gtk_property!(caja_params, :margin_bottom, 6)
    set_gtk_property!(caja_params, :margin_start,  8)
    set_gtk_property!(caja_params, :margin_end,    8)

    # default iters = 1000 -- elegido para que el GD AUN no termine de converger a OLS
    # con estandarizacion ON, asi las diferencias entre los dos metodos son visibles
    # en el panel de resultados. con iters > 2000 el GD converge tan bien que w_GD y
    # w_OLS coinciden hasta la precision de Float64 y los numeros aparecen identicos.
    fila_a, entry_alfa = fila_campo("learning rate (alfa):  ", "0.01")
    fila_i, entry_iter = fila_campo("numero de iteraciones: ", "1000")

    # checkbox para estandarizar x antes del GD. con datasets grandes (1000+) sin
    # estandarizar el GD necesita muchisimo mas iteraciones para converger. OLS es
    # invariante a la escala, asi que NO se le aplica nunca. default ON porque la
    # literatura lo recomienda; desmarcar para VER el problema de convergencia en vivo.
    check_std = GtkCheckButton("estandarizar x para el GD  (z-score: mu=0, sigma=1)  -- OLS es invariante a la escala, no lo necesita")
    set_gtk_property!(check_std, :active, true)

    nota = GtkLabel("  alfa: 0.001 a 0.05   --   iters bajos (~1000) muestran diferencias visibles GD vs OLS;   iters altos (>2000) hacen converger ambos al mismo valor")
    set_gtk_property!(nota, :xalign, 0.0f0)

    for widget in [fila_a, fila_i, check_std, nota]
        push!(caja_params, widget)
    end
    push!(marco_params, caja_params)

    marco_pred = GtkFrame("  conjuntos de datos predefinidos  (los 2 primeros son a mano; los 3 ultimos son sinteticos con verdad conocida)  ")
    caja_pred  = GtkBox(:v)
    set_gtk_property!(caja_pred, :spacing,       6)
    set_gtk_property!(caja_pred, :margin_top,    6)
    set_gtk_property!(caja_pred, :margin_bottom, 6)
    set_gtk_property!(caja_pred, :margin_start,  8)
    set_gtk_property!(caja_pred, :margin_end,    8)

    boton_3     = GtkButton("3 puntos       a mano      : (1,2.0), (2,4.0), (3,5.5)   -- sin verdad definida")
    boton_5     = GtkButton("5 puntos       a mano      : (1,1.5), (2,3.0), (3,4.0), (4,5.5), (5,7.0)   -- sin verdad definida")
    boton_100   = GtkButton("100 puntos     sintetico   : y = $(A_VERDADERO)x + $(B_VERDADERO) + ruido suave determinista (|<0.1|, sin outliers)")
    boton_1000  = GtkButton("1000 puntos    sintetico   : misma formula, mas denso -- aqui empieza a notarse el escalado")
    boton_10000 = GtkButton("10000 puntos   sintetico   : misma formula, denso -- aqui se ve la ventaja real de OLS en tiempo")

    for btn in [boton_3, boton_5, boton_100, boton_1000, boton_10000]
        push!(caja_pred, btn)
    end
    push!(marco_pred, caja_pred)

    marco_res = GtkFrame("  comparativa  (w*, b*, vs verdad cuando aplica, tiempos)  ")
    label_res = GtkLabel("presiona un boton para ejecutar ambos metodos sobre la misma data...")
    set_gtk_property!(label_res, :margin_top,    10)
    set_gtk_property!(label_res, :margin_bottom, 10)
    set_gtk_property!(label_res, :margin_start,  10)
    set_gtk_property!(label_res, :margin_end,    10)
    set_gtk_property!(label_res, :xalign,        0.0f0)
    push!(marco_res, label_res)

    for widget in [titulo_lbl, marco_params, marco_pred, marco_res]
        push!(caja, widget)
    end
    push!(ventana, caja)

    function leer_params()
        alfa   = parse(Float64, strip(get_gtk_property(entry_alfa, :text, String)))
        iters  = parse(Int,     strip(get_gtk_property(entry_iter, :text, String)))
        std_on = get_gtk_property(check_std, :active, Bool)
        return alfa, iters, std_on
    end

    function formatear_tiempo(t_seg)
        if t_seg < 1.0e-3
            return "$(round(t_seg * 1.0e6, digits = 2)) us"
        elseif t_seg < 1.0
            return "$(round(t_seg * 1.0e3, digits = 3)) ms"
        else
            return "$(round(t_seg, digits = 4)) s"
        end
    end

    # mostramos: w_GD, b_GD, w_OLS, b_OLS, diferencias entre los dos, tiempos, y
    # SI el dataset es sintetico (a_true, b_true no son NaN), tambien comparamos
    # cada metodo contra los parametros VERDADEROS y declaramos un ganador en w
    # y en b por separado.
    # precision de impresion: 8 digitos para w y b (los valores), 10 para las
    # diferencias (suficiente para que sean visibles aun cuando sean del orden 1e-5).
    function mostrar_comparativa(n_pts, w_gd, b_gd, w_ols, b_ols,
                                 alfa, iters, std_on, t_gd, t_ols,
                                 a_true, b_true)
        dif_w     = abs(w_gd - w_ols)
        dif_b     = abs(b_gd - b_ols)
        ratio_tex = t_ols > 0 ? "$(round(t_gd / t_ols, digits = 2))x" : "n/a"
        std_tex   = std_on ? "SI (z-score)" : "NO"

        # seccion "vs verdad" -- solo aparece si el dataset es sintetico (a_true no es NaN)
        seccion_verdad = ""
        if !isnan(a_true)
            err_w_gd  = abs(w_gd  - a_true)
            err_w_ols = abs(w_ols - a_true)
            err_b_gd  = abs(b_gd  - b_true)
            err_b_ols = abs(b_ols - b_true)
            ganador_w = err_w_gd < err_w_ols ? "GD"  : (err_w_gd > err_w_ols ? "OLS" : "empate")
            ganador_b = err_b_gd < err_b_ols ? "GD"  : (err_b_gd > err_b_ols ? "OLS" : "empate")

            seccion_verdad = (
                "\n" *
                "parametros VERDADEROS del dataset sintetico: a = $a_true,  b = $b_true\n" *
                "errores absolutos respecto a la verdad:\n" *
                "  GD :  |w_GD  - $a_true| = $(round(err_w_gd,  digits = 10))     |b_GD  - $b_true| = $(round(err_b_gd,  digits = 10))\n" *
                "  OLS:  |w_OLS - $a_true| = $(round(err_w_ols, digits = 10))     |b_OLS - $b_true| = $(round(err_b_ols, digits = 10))\n" *
                "  ganador en w: $ganador_w     --     ganador en b: $ganador_b\n"
            )
        end
        texto = (
            "comparativa sobre $n_pts puntos  --  alfa = $alfa, iters = $iters, estandarizar = $std_tex\n" *
            "\n" *
            "metodo a mano (gradiente descendiente):\n" *
            "  w_GD  = $(round(w_gd, digits = 8))\n" *
            "  b_GD  = $(round(b_gd, digits = 8))\n" *
            "  ecuacion : y = $(round(w_gd, digits = 4))x + $(round(b_gd, digits = 4))\n" *
            "  tiempo   : $(formatear_tiempo(t_gd))   ($iters iteraciones)\n" *
            "\n" *
            "metodo nativo (OLS via GLM.jl, referencia exacta):\n" *
            "  w_OLS = $(round(w_ols, digits = 8))\n" *
            "  b_OLS = $(round(b_ols, digits = 8))\n" *
            "  ecuacion : y = $(round(w_ols, digits = 4))x + $(round(b_ols, digits = 4))\n" *
            "  tiempo   : $(formatear_tiempo(t_ols))   (1 paso, ecuaciones normales)\n" *
            "\n" *
            "diferencias absolutas  |GD - OLS|:\n" *
            "  |w_GD - w_OLS| = $(round(dif_w, digits = 10))\n" *
            "  |b_GD - b_OLS| = $(round(dif_b, digits = 10))\n" *
            "  GD tardo $ratio_tex respecto de OLS" *
            seccion_verdad
        )
        set_gtk_property!(label_res, :label, texto)
    end

    # todo el flujo junto: lee params, estandariza si corresponde, corre GD, corre OLS,
    # desestandariza coeficientes para que la comparacion sea en escala original,
    # muestra resultado (con o sin seccion vs verdad segun el dataset) y grafica.
    # a_true y b_true son los parametros con los que se genero el dataset; para los
    # datasets a mano se pasan como NaN y la comparacion vs verdad se omite.
    function comparar_y_graficar(X, Y, a_true, b_true, titulo_graf)
        try
            alfa, iters, std_on = leer_params()

            # estandarizamos x SOLO para el GD (OLS no lo necesita). guardamos mu y sigma
            # para revertir los coeficientes al final y mostrarlos en la escala original.
            if std_on
                x_para_gd, mu_x, sigma_x = estandarizar_x(X)
            else
                x_para_gd = X
                mu_x      = 0.0
                sigma_x   = 1.0
            end

            # GD identico al de lineal_regresion_guinativa.jl (misma firma, misma matematica).
            # cronometramos solo el ajuste.
            t0_gd = time_ns()
            w_gd_raw, b_gd_raw, hist_costo, hist_w_raw, _ =
                gradiente_descendiente(x_para_gd, Y, alfa, iters)
            t_gd = (time_ns() - t0_gd) / 1.0e9

            # si estandarizamos, convertimos coeficientes y trayectoria a la escala
            # original. hist_w se proyecta en escala original via w_orig = w_std / sigma;
            # eso es lo que se grafica sobre el corte b = b_OLS en el panel 3.
            if std_on
                w_gd, b_gd = desestandarizar_coef(w_gd_raw, b_gd_raw, mu_x, sigma_x)
                hist_w     = [w_std_t / sigma_x for w_std_t in hist_w_raw]
            else
                w_gd   = w_gd_raw
                b_gd   = b_gd_raw
                hist_w = hist_w_raw
            end

            # OLS sobre X, Y originales (es invariante a la escala lineal, no necesita
            # estandarizacion). cronometramos solo el ajuste.
            t0_ols     = time_ns()
            modelo_ols = ajustar_ols(X, Y)
            t_ols      = (time_ns() - t0_ols) / 1.0e9
            coefs      = coef(modelo_ols)
            b_ols      = coefs[1]
            w_ols      = coefs[2]
            rss_ols    = deviance(modelo_ols)

            mostrar_comparativa(length(X), w_gd, b_gd, w_ols, b_ols,
                                alfa, iters, std_on, t_gd, t_ols,
                                a_true, b_true)
            graficar_comparativa(X, Y,
                                 w_gd, b_gd, hist_costo, hist_w,
                                 w_ols, b_ols, rss_ols,
                                 titulo_graf, alfa, iters)
        catch err
            set_gtk_property!(label_res, :label, "error: $(string(err))")
        end
    end

    # datasets a mano: NaN en a_true y b_true porque no hay verdad definida.
    # la seccion "vs verdad" del panel se omite automaticamente cuando es NaN.
    signal_connect(boton_3, "clicked") do _
        comparar_y_graficar(X_3, Y_3, NaN, NaN, "comparativa GD vs OLS -- 3 puntos a mano")
    end

    signal_connect(boton_5, "clicked") do _
        comparar_y_graficar(X_5, Y_5, NaN, NaN, "comparativa GD vs OLS -- 5 puntos a mano")
    end

    # datasets sinteticos: pasamos A_VERDADERO y B_VERDADERO para activar la
    # comparacion contra los parametros con los que se genero la data.
    signal_connect(boton_100, "clicked") do _
        comparar_y_graficar(X_100, Y_100, A_VERDADERO, B_VERDADERO, "comparativa GD vs OLS -- 100 puntos sinteticos")
    end

    signal_connect(boton_1000, "clicked") do _
        comparar_y_graficar(X_1000, Y_1000, A_VERDADERO, B_VERDADERO, "comparativa GD vs OLS -- 1000 puntos sinteticos")
    end

    signal_connect(boton_10000, "clicked") do _
        comparar_y_graficar(X_10000, Y_10000, A_VERDADERO, B_VERDADERO, "comparativa GD vs OLS -- 10000 puntos sinteticos")
    end

    showall(ventana)

    if !isinteractive()   # si corremos desde terminal bloqueamos el hilo principal asi la ventana no se cierra sola
        condicion = Condition()
        signal_connect(ventana, "destroy") do _
            notify(condicion)
        end
        @async Gtk.gtk_main()
        wait(condicion)
    end
end


lanzar_aplicacion()
