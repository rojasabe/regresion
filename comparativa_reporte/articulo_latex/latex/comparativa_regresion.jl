using Gtk
using Plots
using GLM
using DataFrames
plotlyjs()
const X_3 = [1.0, 2.0, 3.0]
const Y_3 = [2.0, 4.0, 5.5]

const X_5 = [1.0, 2.0, 3.0, 4.0, 5.0]
const Y_5 = [1.5, 3.0, 4.0, 5.5, 7.0]

const PENDIENTE_VERDADERO = 1.5
const INTERCEPTO_VERDADERO = 0.5

function ruido_determinista(x)
    return 0.001 * (x - 2.0) * (x - 4.0) * (x - 8.0)
end

const X_100 = collect(range(1.0, 10.0, length = 100))
const Y_100 = [PENDIENTE_VERDADERO * x + INTERCEPTO_VERDADERO + ruido_determinista(x) for x in X_100]

const X_1000 = collect(range(1.0, 10.0, length = 1000))
const Y_1000 = [PENDIENTE_VERDADERO * x + INTERCEPTO_VERDADERO + ruido_determinista(x) for x in X_1000]

const X_10000 = collect(range(1.0, 10.0, length = 10000))
const Y_10000 = [PENDIENTE_VERDADERO * x + INTERCEPTO_VERDADERO + ruido_determinista(x) for x in X_10000]


function modelo_lineal(pendiente, intercepto, x)
    return pendiente * x + intercepto
end

function funcion_costo(pendiente, intercepto, x_datos, y_datos)
    cantidad_muestras = length(x_datos)
    suma = sum((modelo_lineal(pendiente, intercepto, x_datos[i]) - y_datos[i])^2 for i in 1:cantidad_muestras)
    return suma / (2.0 * cantidad_muestras)
end

function derivada_pendiente(pendiente, intercepto, x_datos, y_datos)
    cantidad_muestras = length(x_datos)
    suma = sum((modelo_lineal(pendiente, intercepto, x_datos[i]) - y_datos[i]) * x_datos[i] for i in 1:cantidad_muestras)
    return suma / cantidad_muestras
end

function derivada_intercepto(pendiente, intercepto, x_datos, y_datos)
    cantidad_muestras = length(x_datos)
    suma = sum(modelo_lineal(pendiente, intercepto, x_datos[i]) - y_datos[i] for i in 1:cantidad_muestras)
    return suma / cantidad_muestras
end

function gradiente_descendiente(x_datos, y_datos, alfa, numero_iteraciones)
    pendiente = 0.0
    intercepto = 0.0
    historial_costo = Float64[]
    historial_pendiente = Float64[]
    historial_intercepto = Float64[]

    for _ in 1:numero_iteraciones
        w = pendiente - alfa * derivada_pendiente(pendiente, intercepto, x_datos, y_datos)
        b = intercepto - alfa * derivada_intercepto(pendiente, intercepto, x_datos, y_datos)
        pendiente = w
        intercepto = b
        push!(historial_costo, funcion_costo(pendiente, intercepto, x_datos, y_datos))
        push!(historial_pendiente, pendiente)
        push!(historial_intercepto, intercepto)
    end

    return pendiente, intercepto, historial_costo, historial_pendiente, historial_intercepto
end


function estandarizar_x(x_datos)
    cantidad_muestras = length(x_datos)
    media = sum(x_datos) / cantidad_muestras
    varianza = sum((valor - media)^2 for valor in x_datos) / cantidad_muestras
    desviacion_estandar = sqrt(varianza)
    x_estandarizado = [(valor - media) / desviacion_estandar for valor in x_datos]
    return x_estandarizado, media, desviacion_estandar
end

function desestandarizar_coeficientes(pendiente_estandarizada, intercepto_estandarizado, media, desviacion_estandar)
    pendiente_original = pendiente_estandarizada / desviacion_estandar
    intercepto_original = intercepto_estandarizado - pendiente_estandarizada * media / desviacion_estandar
    return pendiente_original, intercepto_original
end


function ajustar_minimos_cuadrados(x_datos, y_datos)
    tabla_datos = DataFrame(x = x_datos, y = y_datos)
    return lm(@formula(y ~ x), tabla_datos)
end

function calcular_rss(valor_pendiente, valor_intercepto, x_datos, y_datos)
    return sum((valor_pendiente .* x_datos .+ valor_intercepto .- y_datos) .^ 2)
end


function graficar_comparativa(x_datos, y_datos,
                              pendiente_gradiente, intercepto_gradiente, historial_costo, historial_pendiente,
                              pendiente_minimos_cuadrados, intercepto_minimos_cuadrados, rss_minimos_cuadrados,
                              titulo, alfa, iteraciones)

    cantidad_iteraciones = length(historial_pendiente)
    rango_x = collect(range(minimum(x_datos), maximum(x_datos), length = 200))
    y_gradiente = modelo_lineal.(pendiente_gradiente, intercepto_gradiente, rango_x)
    y_minimos_cuadrados = modelo_lineal.(pendiente_minimos_cuadrados, intercepto_minimos_cuadrados, rango_x)

    y_completo = vcat(y_datos, y_gradiente, y_minimos_cuadrados)
    margen_vertical = (maximum(y_completo) - minimum(y_completo)) * 0.20
    margen_horizontal = (maximum(x_datos) - minimum(x_datos)) * 0.10
    panel_uno = scatter(x_datos, y_datos,
        label = "datos ($(length(x_datos)) puntos)",
        xlabel = "x",
        ylabel = "y",
        title = titulo,
        markersize = 4,
        color = :blue,
        legend = :topleft,
        alpha = 0.5
    )
    plot!(panel_uno, rango_x, y_gradiente,
        label = "GD : y = $(round(Float64(pendiente_gradiente), digits = 4))x + $(round(Float64(intercepto_gradiente), digits = 4))",
        color = :red,
        linewidth = 2.5
    )
    plot!(panel_uno, rango_x, y_minimos_cuadrados,
        label = "OLS: y = $(round(Float64(pendiente_minimos_cuadrados), digits = 4))x + $(round(Float64(intercepto_minimos_cuadrados), digits = 4))",
        color = :green,
        linewidth = 2.5,
        linestyle = :dash
    )
    xlims!(panel_uno, (minimum(x_datos) - margen_horizontal, maximum(x_datos) + margen_horizontal))
    ylims!(panel_uno, (minimum(y_completo) - margen_vertical, maximum(y_completo) + margen_vertical))

    y_predicho_gradiente = modelo_lineal.(pendiente_gradiente, intercepto_gradiente, x_datos)
    y_predicho_minimos_cuadrados = modelo_lineal.(pendiente_minimos_cuadrados, intercepto_minimos_cuadrados, x_datos)
    residuo_gradiente = y_datos .- y_predicho_gradiente
    residuo_minimos_cuadrados = y_datos .- y_predicho_minimos_cuadrados

    panel_dos = scatter(y_predicho_gradiente, residuo_gradiente,
        label = "GD",
        xlabel = "predicho y_hat",
        ylabel = "residuo (y - y_hat)",
        title = "residuos vs predichos",
        color = :red,
        markersize = 4,
        alpha = 0.75,
        legend = :topright
    )
    scatter!(panel_dos, y_predicho_minimos_cuadrados, residuo_minimos_cuadrados,
        label = "OLS",
        color = :green,
        markersize = 4,
        alpha = 0.75
    )
    hline!(panel_dos, [0.0],
        label = "cero",
        color = :blue,
        linewidth = 1.5,
        linestyle = :dash
    )

    margen = max(abs(pendiente_minimos_cuadrados) * 0.8, abs(pendiente_minimos_cuadrados - historial_pendiente[1]) * 1.2, 1.0)
    eje_pendiente = collect(range(pendiente_minimos_cuadrados - margen, pendiente_minimos_cuadrados + margen, length = 400))
    eje_costo = [calcular_rss(valor, intercepto_minimos_cuadrados, x_datos, y_datos) for valor in eje_pendiente]
    panel_tres = plot(eje_pendiente, eje_costo,
        label = "",
        xlabel = "w",
        ylabel = "RSS(w, b_OLS)",
        title = "superficie de costo: camino del GD vs minimo OLS",
        color = :purple,
        linewidth = 2,
        legend = :topright
    )
    paso_camino = max(1, cantidad_iteraciones ÷ 60)
    indices_camino = collect(1:paso_camino:cantidad_iteraciones)
    costos_camino_gradiente = [calcular_rss(historial_pendiente[i], intercepto_minimos_cuadrados, x_datos, y_datos) for i in indices_camino]
    scatter!(panel_tres, historial_pendiente[indices_camino], costos_camino_gradiente,
        label = "camino del GD",
        color = :orange,
        markersize = 4,
        alpha = 0.75
    )
    scatter!(panel_tres, [pendiente_minimos_cuadrados], [rss_minimos_cuadrados],
        label = "OLS exacto  w* = $(round(pendiente_minimos_cuadrados, digits = 4))",
        color = :green,
        markersize = 11,
        markershape = :diamond
    )
    scatter!(panel_tres, [pendiente_gradiente], [calcular_rss(pendiente_gradiente, intercepto_minimos_cuadrados, x_datos, y_datos)],
        label = "GD final  w = $(round(pendiente_gradiente, digits = 4))",
        color = :red,
        markersize = 9,
        markershape = :diamond
    )

    costo_minimos_cuadrados = rss_minimos_cuadrados / (2.0 * length(x_datos))
    panel_cuatro = plot(1:length(historial_costo), historial_costo,
        label = "J del GD",
        xlabel = "iteracion",
        ylabel = "J(w,b) = RSS / (2m)",
        title = "convergencia del GD vs OLS  --  alfa = $alfa, iters = $iteraciones",
        color = :orange,
        linewidth = 2,
        legend = :topright
    )
    hline!(panel_cuatro, [costo_minimos_cuadrados],
        label = "J*  (referencia OLS) = $(round(costo_minimos_cuadrados, digits = 6))",
        color = :green,
        linewidth = 2,
        linestyle = :dash
    )
    scatter!(panel_cuatro, [length(historial_costo)], [historial_costo[end]],
        label = "J final GD = $(round(historial_costo[end], digits = 6))",
        color = :red,
        markersize = 8,
        markershape = :diamond
    )

    figura = plot(panel_uno, panel_dos, panel_tres, panel_cuatro,
        layout = (2, 2),
        size = (1500, 950),
        left_margin = 8Plots.mm,
        right_margin = 8Plots.mm,
        top_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm,
        link = :none
    )
    display(figura)
end


function lanzar_aplicacion()

    let x_calentamiento = [1.0, 2.0, 3.0], y_calentamiento = [1.0, 2.0, 3.0]
        gradiente_descendiente(x_calentamiento, y_calentamiento, 0.01, 10)
        ajustar_minimos_cuadrados(x_calentamiento, y_calentamiento)
        estandarizar_x(x_calentamiento)
    end

    ventana = GtkWindow("comparativa  GD vs OLS  --  gradiente descendiente vs GLM.jl", 820, 880)
    caja = GtkBox(:v)
    set_gtk_property!(caja, :spacing, 8)
    set_gtk_property!(caja, :margin_top, 12)
    set_gtk_property!(caja, :margin_bottom, 12)
    set_gtk_property!(caja, :margin_start, 12)
    set_gtk_property!(caja, :margin_end, 12)

    etiqueta_titulo = GtkLabel(
        "<b>comparativa  GD (a mano)  vs  OLS (GLM.jl)</b>\n" *
        "modelo : y = wx + b   --   misma data, dos metodos de ajuste\n" *
        "GD : iterativo, depende de alfa e iteraciones   --   OLS : cerrado, un solo paso\n" *
        "datasets sinteticos generados con  a = $PENDIENTE_VERDADERO,  b = $INTERCEPTO_VERDADERO  (puntaje a igualar)"
    )
    set_gtk_property!(etiqueta_titulo, :use_markup, true)
    set_gtk_property!(etiqueta_titulo, :justify, 2)

    function fila_campo(texto_etiqueta, valor_predeterminado, ancho_caracteres = 12)
        fila = GtkBox(:h)
        set_gtk_property!(fila, :spacing, 6)
        etiqueta = GtkLabel(texto_etiqueta)
        set_gtk_property!(etiqueta, :xalign, 0.0f0)
        entrada = GtkEntry()
        set_gtk_property!(entrada, :text, valor_predeterminado)
        set_gtk_property!(entrada, :width_chars, ancho_caracteres)
        push!(fila, etiqueta)
        push!(fila, entrada)
        return fila, entrada
    end

    marco_parametros = GtkFrame("  hiperparametros del GD  (OLS no los usa, su ajuste es siempre el mismo)  ")
    caja_parametros = GtkBox(:v)
    set_gtk_property!(caja_parametros, :spacing, 6)
    set_gtk_property!(caja_parametros, :margin_top, 6)
    set_gtk_property!(caja_parametros, :margin_bottom, 6)
    set_gtk_property!(caja_parametros, :margin_start, 8)
    set_gtk_property!(caja_parametros, :margin_end, 8)

    fila_alfa, entrada_alfa = fila_campo("learning rate (alfa): ", "0.01")
    fila_iteraciones, entrada_iteraciones = fila_campo("numero de iteraciones: ", "1000")

    casilla_estandarizar = GtkCheckButton("estandarizar x para el GD (z-score: mu=0, sigma=1) -- OLS es invariante a la escala, no lo necesita")
    set_gtk_property!(casilla_estandarizar, :active, true)

    etiqueta_nota = GtkLabel("  alfa: 0.001 a 0.05  --  iters bajos (~1000) muestran diferencias visibles GD vs OLS; iters altos (>2000) hacen converger ambos al mismo valor")
    set_gtk_property!(etiqueta_nota, :xalign, 0.0f0)

    for widget in [fila_alfa, fila_iteraciones, casilla_estandarizar, etiqueta_nota]
        push!(caja_parametros, widget)
    end
    push!(marco_parametros, caja_parametros)

    marco_predefinidos = GtkFrame("  conjuntos de datos predefinidos  (los 2 primeros son a mano; los 3 ultimos son sinteticos con verdad conocida)  ")
    caja_predefinidos = GtkBox(:v)
    set_gtk_property!(caja_predefinidos, :spacing, 6)
    set_gtk_property!(caja_predefinidos, :margin_top, 6)
    set_gtk_property!(caja_predefinidos, :margin_bottom, 6)
    set_gtk_property!(caja_predefinidos, :margin_start, 8)
    set_gtk_property!(caja_predefinidos, :margin_end, 8)

    boton_3 = GtkButton("3 puntos a mano: (1, 2.0), (2, 4.0), (3, 5.5) -- sin verdad definida")
    boton_5 = GtkButton("5 puntos a mano: (1, 1.5), (2, 3.0), (3, 4.0), (4, 5.5), (5, 7.0) -- sin verdad definida")
    boton_100 = GtkButton("100 puntos sintetico: y = $(PENDIENTE_VERDADERO)x + $(INTERCEPTO_VERDADERO) + ruido suave determinista (|<0.1|, sin outliers)")
    boton_1000 = GtkButton("1000 puntos sintetico: misma formula, mas denso -- aqui empieza a notarse el escalado")
    boton_10000 = GtkButton("10000 puntos sintetico: misma formula, denso -- aqui se ve la ventaja real de OLS en tiempo")

    for boton in [boton_3, boton_5, boton_100, boton_1000, boton_10000]
        push!(caja_predefinidos, boton)
    end
    push!(marco_predefinidos, caja_predefinidos)

    marco_resultados = GtkFrame("  comparativa  (w*, b*, vs verdad cuando aplica, tiempos)  ")
    etiqueta_resultados = GtkLabel("presiona un boton para ejecutar ambos metodos sobre la misma data...")
    set_gtk_property!(etiqueta_resultados, :margin_top, 10)
    set_gtk_property!(etiqueta_resultados, :margin_bottom, 10)
    set_gtk_property!(etiqueta_resultados, :margin_start, 10)
    set_gtk_property!(etiqueta_resultados, :margin_end, 10)
    set_gtk_property!(etiqueta_resultados, :xalign, 0.0f0)
    push!(marco_resultados, etiqueta_resultados)

    for widget in [etiqueta_titulo, marco_parametros, marco_predefinidos, marco_resultados]
        push!(caja, widget)
    end
    push!(ventana, caja)

    function leer_parametros()
        alfa = parse(Float64, strip(get_gtk_property(entrada_alfa, :text, String)))
        iteraciones = parse(Int, strip(get_gtk_property(entrada_iteraciones, :text, String)))
        estandarizar_activado = get_gtk_property(casilla_estandarizar, :active, Bool)
        return alfa, iteraciones, estandarizar_activado
    end

    function formatear_tiempo(tiempo_segundos)
        if tiempo_segundos < 1.0e-3
            return "$(round(tiempo_segundos * 1.0e6, digits = 2)) us"
        elseif tiempo_segundos < 1.0
            return "$(round(tiempo_segundos * 1.0e3, digits = 3)) ms"
        else
            return "$(round(tiempo_segundos, digits = 4)) s"
        end
    end

    function mostrar_comparativa(cantidad_puntos, pendiente_gradiente, intercepto_gradiente, pendiente_minimos_cuadrados, intercepto_minimos_cuadrados,
                                 alfa, iteraciones, estandarizar_activado, tiempo_gradiente, tiempo_minimos_cuadrados,
                                 pendiente_verdadera, intercepto_verdadero, rss_gradiente, rss_minimos_cuadrados)
        function porcentaje_mejora(valor_mejor, valor_peor)
            return valor_peor > 0 ? (1.0 - valor_mejor / valor_peor) * 100.0 : NaN
        end

        diferencia_pendiente = abs(pendiente_gradiente - pendiente_minimos_cuadrados)
        diferencia_intercepto = abs(intercepto_gradiente - intercepto_minimos_cuadrados)
        texto_estandarizacion = estandarizar_activado ? "SI (z-score)" : "NO"

        ganador_tiempo = "empate"
        porcentaje_tiempo = 0.0
        if abs(tiempo_gradiente - tiempo_minimos_cuadrados) > 1.0e-9
            ganador_tiempo = tiempo_gradiente < tiempo_minimos_cuadrados ? "GD" : "OLS"
            tiempo_menor = min(tiempo_gradiente, tiempo_minimos_cuadrados)
            tiempo_mayor = max(tiempo_gradiente, tiempo_minimos_cuadrados)
            porcentaje_tiempo = porcentaje_mejora(tiempo_menor, tiempo_mayor)
        end
        texto_tiempo = isfinite(porcentaje_tiempo) ? "$(ganador_tiempo) es $(round(porcentaje_tiempo, digits = 2))% mas rapido" : "n/a"
        ganador_precision = "empate"
        porcentaje_precision = 0.0
        if abs(rss_gradiente - rss_minimos_cuadrados) > 1.0e-12
            ganador_precision = rss_gradiente < rss_minimos_cuadrados ? "GD" : "OLS"
            rss_menor = min(rss_gradiente, rss_minimos_cuadrados)
            rss_mayor = max(rss_gradiente, rss_minimos_cuadrados)
            porcentaje_precision = porcentaje_mejora(rss_menor, rss_mayor)
        end
        texto_precision = isfinite(porcentaje_precision) ? "$(ganador_precision) es $(round(porcentaje_precision, digits = 2))% mejor" : "n/a"

        brecha_optimizacion = rss_minimos_cuadrados > 0 ? (rss_gradiente - rss_minimos_cuadrados) / rss_minimos_cuadrados * 100.0 : NaN
        texto_brecha = isfinite(brecha_optimizacion) ? "$(round(brecha_optimizacion, digits = 6))% por encima del minimo OLS" : "n/a"

        seccion_verdad = ""
        if !isnan(pendiente_verdadera)
            error_pendiente_gradiente = abs(pendiente_gradiente - pendiente_verdadera)
            error_pendiente_minimos_cuadrados = abs(pendiente_minimos_cuadrados - pendiente_verdadera)
            error_intercepto_gradiente = abs(intercepto_gradiente - intercepto_verdadero)
            error_intercepto_minimos_cuadrados = abs(intercepto_minimos_cuadrados - intercepto_verdadero)
            ganador_pendiente = error_pendiente_gradiente < error_pendiente_minimos_cuadrados ? "GD" : (error_pendiente_gradiente > error_pendiente_minimos_cuadrados ? "OLS" : "empate")
            ganador_intercepto = error_intercepto_gradiente < error_intercepto_minimos_cuadrados ? "GD" : (error_intercepto_gradiente > error_intercepto_minimos_cuadrados ? "OLS" : "empate")

            error_porcentual_pendiente_gradiente = pendiente_verdadera != 0 ? error_pendiente_gradiente / abs(pendiente_verdadera) * 100.0 : NaN
            error_porcentual_pendiente_minimos_cuadrados = pendiente_verdadera != 0 ? error_pendiente_minimos_cuadrados / abs(pendiente_verdadera) * 100.0 : NaN
            error_porcentual_intercepto_gradiente = intercepto_verdadero != 0 ? error_intercepto_gradiente / abs(intercepto_verdadero) * 100.0 : NaN
            error_porcentual_intercepto_minimos_cuadrados = intercepto_verdadero != 0 ? error_intercepto_minimos_cuadrados / abs(intercepto_verdadero) * 100.0 : NaN

            seccion_verdad = (
                "\n" *
                "parametros VERDADEROS del dataset sintetico: a = $pendiente_verdadera, b = $intercepto_verdadero\n" *
                "errores absolutos respecto a la verdad:\n" *
                "  GD :  |w_GD  - $pendiente_verdadera| = $(round(error_pendiente_gradiente, digits = 10))     |b_GD  - $intercepto_verdadero| = $(round(error_intercepto_gradiente, digits = 10))\n" *
                "  OLS:  |w_OLS - $pendiente_verdadera| = $(round(error_pendiente_minimos_cuadrados, digits = 10))     |b_OLS - $intercepto_verdadero| = $(round(error_intercepto_minimos_cuadrados, digits = 10))\n" *
                "errores porcentuales relativos a la verdad:\n" *
                "  GD :  w = $(round(error_porcentual_pendiente_gradiente, digits = 6))%     b = $(round(error_porcentual_intercepto_gradiente, digits = 6))%\n" *
                "  OLS:  w = $(round(error_porcentual_pendiente_minimos_cuadrados, digits = 6))%     b = $(round(error_porcentual_intercepto_minimos_cuadrados, digits = 6))%\n" *
                "  ganador en w: $ganador_pendiente     --     ganador en b: $ganador_intercepto\n"
            )
        end
        texto_resultado = (
            "comparativa sobre $cantidad_puntos puntos  --  alfa = $alfa, iters = $iteraciones, estandarizar = $texto_estandarizacion\n" *
            "\n" *
            "metodo a mano (gradiente descendiente):\n" *
            "  w_GD  = $(round(pendiente_gradiente, digits = 8))\n" *
            "  b_GD  = $(round(intercepto_gradiente, digits = 8))\n" *
            "  ecuacion : y = $(round(pendiente_gradiente, digits = 4))x + $(round(intercepto_gradiente, digits = 4))\n" *
            "  tiempo   : $(formatear_tiempo(tiempo_gradiente))   ($iteraciones iteraciones)\n" *
            "\n" *
            "metodo nativo (OLS via GLM.jl, referencia exacta):\n" *
            "  w_OLS = $(round(pendiente_minimos_cuadrados, digits = 8))\n" *
            "  b_OLS = $(round(intercepto_minimos_cuadrados, digits = 8))\n" *
            "  ecuacion : y = $(round(pendiente_minimos_cuadrados, digits = 4))x + $(round(intercepto_minimos_cuadrados, digits = 4))\n" *
            "  tiempo   : $(formatear_tiempo(tiempo_minimos_cuadrados))   (1 paso, ecuaciones normales)\n" *
            "\n" *
            "diferencias absolutas  |GD - OLS|:\n" *
            "  |w_GD - w_OLS| = $(round(diferencia_pendiente, digits = 10))\n" *
            "  |b_GD - b_OLS| = $(round(diferencia_intercepto, digits = 10))\n" *
            "\n" *
            "comparativa porcentual (menor es mejor):\n" *
            "  tiempo    : $texto_tiempo\n" *
            "  precision : $texto_precision\n" *
            "  brecha    : GD esta $texto_brecha (gap de optimizacion respecto al RSS optimo)\n" *
            seccion_verdad
        )
        set_gtk_property!(etiqueta_resultados, :label, texto_resultado)
    end

    function comparar_y_graficar(x_datos, y_datos, pendiente_verdadera, intercepto_verdadero, titulo_grafica)
        try
            alfa, iteraciones, estandarizar_activado = leer_parametros()

            if estandarizar_activado
                x_para_gradiente, media_x, desviacion_x = estandarizar_x(x_datos)
            else
                x_para_gradiente = x_datos
                media_x = 0.0
                desviacion_x = 1.0
            end

            tiempo_inicio_gradiente = time_ns()
            pendiente_gradiente_cruda, intercepto_gradiente_crudo, historial_costo, historial_pendiente_crudo, _ =
                gradiente_descendiente(x_para_gradiente, y_datos, alfa, iteraciones)
            tiempo_gradiente = (time_ns() - tiempo_inicio_gradiente) / 1.0e9

            if estandarizar_activado
                pendiente_gradiente, intercepto_gradiente = desestandarizar_coeficientes(pendiente_gradiente_cruda, intercepto_gradiente_crudo, media_x, desviacion_x)
                historial_pendiente = [valor / desviacion_x for valor in historial_pendiente_crudo]
            else
                pendiente_gradiente = pendiente_gradiente_cruda
                intercepto_gradiente = intercepto_gradiente_crudo
                historial_pendiente = historial_pendiente_crudo
            end

            tiempo_inicio_minimos_cuadrados = time_ns()
            modelo_minimos_cuadrados = ajustar_minimos_cuadrados(x_datos, y_datos)
            tiempo_minimos_cuadrados = (time_ns() - tiempo_inicio_minimos_cuadrados) / 1.0e9
            coeficientes = coef(modelo_minimos_cuadrados)
            intercepto_minimos_cuadrados = coeficientes[1]
            pendiente_minimos_cuadrados = coeficientes[2]
            rss_minimos_cuadrados = deviance(modelo_minimos_cuadrados)
            rss_gradiente = calcular_rss(pendiente_gradiente, intercepto_gradiente, x_datos, y_datos)

            mostrar_comparativa(length(x_datos), pendiente_gradiente, intercepto_gradiente, pendiente_minimos_cuadrados, intercepto_minimos_cuadrados,
                                alfa, iteraciones, estandarizar_activado, tiempo_gradiente, tiempo_minimos_cuadrados,
                                pendiente_verdadera, intercepto_verdadero, rss_gradiente, rss_minimos_cuadrados)
            graficar_comparativa(x_datos, y_datos,
                                 pendiente_gradiente, intercepto_gradiente, historial_costo, historial_pendiente,
                                 pendiente_minimos_cuadrados, intercepto_minimos_cuadrados, rss_minimos_cuadrados,
                                 titulo_grafica, alfa, iteraciones)
        catch error_capturado
            set_gtk_property!(etiqueta_resultados, :label, "error: $(string(error_capturado))")
        end
    end

    signal_connect(boton_3, "clicked") do _
        comparar_y_graficar(X_3, Y_3, NaN, NaN, "comparativa GD vs OLS -- 3 puntos a mano")
    end

    signal_connect(boton_5, "clicked") do _
        comparar_y_graficar(X_5, Y_5, NaN, NaN, "comparativa GD vs OLS -- 5 puntos a mano")
    end

    signal_connect(boton_100, "clicked") do _
        comparar_y_graficar(X_100, Y_100, PENDIENTE_VERDADERO, INTERCEPTO_VERDADERO, "comparativa GD vs OLS -- 100 puntos sinteticos")
    end

    signal_connect(boton_1000, "clicked") do _
        comparar_y_graficar(X_1000, Y_1000, PENDIENTE_VERDADERO, INTERCEPTO_VERDADERO, "comparativa GD vs OLS -- 1000 puntos sinteticos")
    end

    signal_connect(boton_10000, "clicked") do _
        comparar_y_graficar(X_10000, Y_10000, PENDIENTE_VERDADERO, INTERCEPTO_VERDADERO, "comparativa GD vs OLS -- 10000 puntos sinteticos")
    end

    showall(ventana)

    if !isinteractive()
        condicion_cierre = Condition()
        signal_connect(ventana, "destroy") do _
            notify(condicion_cierre)
        end
        @async Gtk.gtk_main()
        wait(condicion_cierre)
    end
end


lanzar_aplicacion()