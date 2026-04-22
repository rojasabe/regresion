# regresion_lineal_libreria.jl
# ============================================================
# regresión lineal usando GLM.jl — solución analítica (OLS)
# mismos datos y estructura de GUI que regresion_lineal.jl,
# pero sin implementar nada desde cero.
#
# modelo:   Y ~ X  →  ŷ = b + w·x
# método:   OLS — ecuaciones normales: θ = (XᵀX)⁻¹Xᵀy
# ventaja:  solución exacta, sin hiper-parámetros (α, iters)
#
# librerías clave:
#   GLM        — ajuste del modelo lineal
#   DataFrames — tabla de datos requerida por GLM
#   Statistics — media/std para el histograma
#   Plots      — visualización (backend GR)
#   Gtk        — interfaz gráfica (mismo framework)
# ============================================================

using Gtk
using Plots
using GLM
using DataFrames
using Statistics
gr()

# ============================================================
# constantes: mismos datasets que regresion_lineal.jl
# y ≈ 2x + ruido gaussiano
# ============================================================
const X_20  = sort(rand(20)  * 20.0)
const Y_20  = 2.0 .* X_20  .+ randn(20)  * 2.0

const X_50  = sort(rand(50)  * 50.0)
const Y_50  = 2.0 .* X_50  .+ randn(50)  * 4.0

const X_100 = sort(rand(100) * 100.0)
const Y_100 = 2.0 .* X_100 .+ randn(100) * 6.0


# ============================================================
# ajuste del modelo con GLM
#
# lm(@formula(Y ~ X), df) internamente resuelve:
#   θ = (XᵀX)⁻¹ Xᵀy
# que es exactamente la misma solución a la que converge el
# gradiente descendiente de regresion_lineal.jl después de
# miles de iteraciones — pero en una sola operación matricial.
# ============================================================
function ajustar_modelo(x_datos, y_datos)
    df = DataFrame(X = x_datos, Y = y_datos)
    return lm(@formula(Y ~ X), df)
end


# ============================================================
# extrae parámetros y métricas del modelo ajustado
#   coef(m)[1] → b  (intercepto / sesgo)
#   coef(m)[2] → w  (pendiente  / peso)
#   r2(m)      → R² (coeficiente de determinación, 0–1)
# ============================================================
function extraer_parametros(m)
    coefs  = coef(m)
    b      = coefs[1]
    w      = coefs[2]
    r2_val = r2(m)
    return w, b, r2_val
end


# ============================================================
# imprime resultados en consola — mismo estilo que la versión
# manual, más la tabla estadística completa que entrega GLM
# (coeficientes, error estándar, estadístico t, p-valor)
# ============================================================
function imprimir_resultados(m, x_datos, y_datos, titulo)
    w, b, r2_val = extraer_parametros(m)
    n     = length(x_datos)
    y_hat = predict(m)
    costo = sum((y_hat .- y_datos).^2) / (2 * n)   # j(w,b) igual que la versión manual

    println("\n" * "="^62)
    println("  GLM — $titulo")
    println("="^62)
    println("  datos               : $n puntos")
    println("  método              : OLS (ecuaciones normales)")
    println("="^62)
    println("  w  (pendiente)      = $(round(w,      digits=6))")
    println("  b  (intercepto)     = $(round(b,      digits=6))")
    println("  R²                  = $(round(r2_val, digits=6))")
    println("  j(w,b) = (1/2n)Σe² = $(round(costo,  digits=8))")
    println("  ecuación            : f(x) = $(round(w, digits=4))x + $(round(b, digits=4))")
    println("─"^62)
    println("  tabla estadística completa (GLM):")
    println(m)
    println("="^62 * "\n")

    return costo
end


# ============================================================
# graficación: 3 subplots análogos a los de la versión manual
#
#   versión manual          →  versión librería
#   ─────────────────────────────────────────────────────────
#   datos + abanico GD     →  datos + línea + banda IC 95%
#   convergencia j(w,b)    →  residuos vs valores ajustados
#   parábola j(w) + camino →  histograma de residuos
#
# la banda IC 95% y el análisis de residuos son diagnósticos
# estándar que GLM nos entrega gratis.
# ============================================================
function graficar(x_datos, y_datos, m, titulo_base)
    w, b, r2_val = extraer_parametros(m)
    y_hat    = predict(m)
    residuos = residuals(m)

    # línea ajustada + intervalo de confianza 95% en 200 puntos
    x_min   = minimum(x_datos)
    x_max   = maximum(x_datos)
    x_linea = collect(range(x_min, x_max, length=200))
    df_pred = DataFrame(X = x_linea)
    pred_ci = predict(m, df_pred; interval=:confidence, level=0.95)
    y_linea  = pred_ci.prediction
    banda_lo = y_linea .- pred_ci.lower    # ancho hacia abajo
    banda_hi = pred_ci.upper .- y_linea   # ancho hacia arriba

    # ── subplot 1: dispersión + línea ajustada + banda IC 95% ──
    # equivale al subplot de regresión de la versión manual,
    # pero la incertidumbre se muestra como banda estadística
    # en lugar de un abanico de iteraciones intermedias
    g1 = scatter(
        x_datos, y_datos,
        label          = "datos",
        xlabel         = "x",
        ylabel         = "y",
        title          = "$titulo_base\nR² = $(round(r2_val, digits=4))",
        titlefontsize  = 10,
        color          = :steelblue,
        markersize     = 5,
        legend         = :topleft,
        legendfontsize = 8,
        top_margin     = 8Plots.mm,
        bottom_margin  = 6Plots.mm,
        left_margin    = 6Plots.mm
    )
    plot!(g1, x_linea, y_linea,
        ribbon    = (banda_lo, banda_hi),
        fillalpha = 0.2,
        fillcolor = :red,
        color     = :red,
        linewidth = 2.5,
        label     = "f(x)=$(round(w,digits=4))x+$(round(b,digits=4))  [IC 95%]"
    )

    # ── subplot 2: residuos vs valores ajustados ──
    # equivale a la gráfica de convergencia: permite ver si
    # el modelo captura bien la estructura o quedan patrones
    g2 = scatter(
        y_hat, residuos,
        xlabel         = "valores ajustados (ŷ)",
        ylabel         = "residuos  (y − ŷ)",
        title          = "residuos vs ajustados",
        titlefontsize  = 10,
        color          = :darkorange,
        markersize     = 5,
        label          = "residuos",
        legendfontsize = 8,
        top_margin     = 8Plots.mm,
        bottom_margin  = 6Plots.mm,
        left_margin    = 6Plots.mm
    )
    hline!(g2, [0.0],
        color     = :black,
        linewidth = 1.5,
        linestyle = :dash,
        label     = "e = 0"
    )

    # ── subplot 3: histograma de residuos ──
    # equivale a la parábola de costo: muestra la distribución
    # de errores; idealmente centrada en 0 y simétrica
    media_res = mean(residuos)
    g3 = histogram(
        residuos,
        xlabel         = "residuo  (y − ŷ)",
        ylabel         = "frecuencia",
        title          = "distribución de residuos",
        titlefontsize  = 10,
        color          = :purple,
        alpha          = 0.7,
        label          = "residuos",
        legendfontsize = 8,
        top_margin     = 8Plots.mm,
        bottom_margin  = 6Plots.mm,
        left_margin    = 6Plots.mm,
        right_margin   = 4Plots.mm
    )
    vline!(g3, [media_res],
        color     = :red,
        linewidth = 2,
        linestyle = :dash,
        label     = "media=$(round(media_res, digits=3))"
    )

    figura = plot(g1, g2, g3, layout=(1, 3), size=(1300, 460))
    display(figura)
end


# ============================================================
# interfaz gráfica — misma estructura que regresion_lineal.jl
# se eliminan los campos α e iteraciones porque OLS no los
# necesita: la solución es analítica y siempre exacta.
# ============================================================
function lanzar_aplicacion()

    ventana   = GtkWindow("regresión lineal — GLM (OLS)", 540, 480)
    caja_raiz = GtkBox(:v)
    set_gtk_property!(caja_raiz, :spacing,       8)
    set_gtk_property!(caja_raiz, :margin_top,    12)
    set_gtk_property!(caja_raiz, :margin_bottom, 12)
    set_gtk_property!(caja_raiz, :margin_start,  12)
    set_gtk_property!(caja_raiz, :margin_end,    12)

    # --- título ---
    etiqueta_titulo = GtkLabel(
        "<b>regresión lineal con GLM.jl — OLS</b>\n" *
        "modelo: Y ~ X  →  ŷ = b + w·x\n" *
        "método: ecuaciones normales  θ = (XᵀX)⁻¹Xᵀy\n" *
        "<i>no requiere learning rate ni iteraciones</i>"
    )
    set_gtk_property!(etiqueta_titulo, :use_markup, true)
    set_gtk_property!(etiqueta_titulo, :justify, 2)

    # --------------------------------------------------------
    # sección: datos predefinidos (mismos 3 datasets)
    # --------------------------------------------------------
    marco_pred = GtkFrame("  datos predefinidos — solo presiona el botón  ")
    caja_pred  = GtkBox(:v)
    set_gtk_property!(caja_pred, :spacing,       6)
    set_gtk_property!(caja_pred, :margin_top,    6)
    set_gtk_property!(caja_pred, :margin_bottom, 6)
    set_gtk_property!(caja_pred, :margin_start,  8)
    set_gtk_property!(caja_pred, :margin_end,    8)

    boton_20  = GtkButton("ajustar con  20 puntos  (X_20, Y_20 — constantes)")
    boton_50  = GtkButton("ajustar con  50 puntos  (X_50, Y_50 — constantes)")
    boton_100 = GtkButton("ajustar con 100 puntos  (X_100, Y_100 — constantes)")
    push!(caja_pred, boton_20)
    push!(caja_pred, boton_50)
    push!(caja_pred, boton_100)
    push!(marco_pred, caja_pred)

    # --------------------------------------------------------
    # sección: entrada manual de 3 puntos (igual que el manual)
    # --------------------------------------------------------
    marco_manual = GtkFrame("  entrada manual — 3 puntos  ")
    caja_manual  = GtkBox(:v)
    set_gtk_property!(caja_manual, :spacing,       6)
    set_gtk_property!(caja_manual, :margin_top,    6)
    set_gtk_property!(caja_manual, :margin_bottom, 6)
    set_gtk_property!(caja_manual, :margin_start,  8)
    set_gtk_property!(caja_manual, :margin_end,    8)

    instruccion = GtkLabel("ingresa cada punto como   x , y")
    set_gtk_property!(instruccion, :xalign, 0.0f0)

    fila_p1 = GtkBox(:h); label_p1 = GtkLabel("punto 1  (x, y) : "); entry_p1 = GtkEntry()
    fila_p2 = GtkBox(:h); label_p2 = GtkLabel("punto 2  (x, y) : "); entry_p2 = GtkEntry()
    fila_p3 = GtkBox(:h); label_p3 = GtkLabel("punto 3  (x, y) : "); entry_p3 = GtkEntry()

    for (fila, lbl, entry, default) in [
            (fila_p1, label_p1, entry_p1, "1, 2"),
            (fila_p2, label_p2, entry_p2, "2, 4"),
            (fila_p3, label_p3, entry_p3, "3, 5.8")]
        set_gtk_property!(entry, :text, default)
        set_gtk_property!(entry, :width_chars, 12)
        push!(fila, lbl)
        push!(fila, entry)
    end

    boton_manual = GtkButton("ajustar con los 3 puntos manuales")

    push!(caja_manual, instruccion)
    push!(caja_manual, fila_p1)
    push!(caja_manual, fila_p2)
    push!(caja_manual, fila_p3)
    push!(caja_manual, boton_manual)
    push!(marco_manual, caja_manual)

    # --------------------------------------------------------
    # sección: panel de resultados
    # --------------------------------------------------------
    marco_resultado = GtkFrame("  resultados del ajuste  ")
    label_resultado = GtkLabel("presiona un botón para ajustar el modelo...")
    set_gtk_property!(label_resultado, :margin_top,    10)
    set_gtk_property!(label_resultado, :margin_bottom, 10)
    set_gtk_property!(label_resultado, :margin_start,  10)
    set_gtk_property!(label_resultado, :margin_end,    10)
    set_gtk_property!(label_resultado, :xalign, 0.0f0)
    push!(marco_resultado, label_resultado)

    # --------------------------------------------------------
    # ensamblamos la ventana
    # --------------------------------------------------------
    push!(caja_raiz, etiqueta_titulo)
    push!(caja_raiz, marco_pred)
    push!(caja_raiz, marco_manual)
    push!(caja_raiz, marco_resultado)
    push!(ventana, caja_raiz)

    # ============================================================
    # funciones auxiliares de la GUI
    # ============================================================

    function parsear_punto(entry)
        texto  = get_gtk_property(entry, :text, String)
        partes = split(texto, ',')
        x = parse(Float64, strip(partes[1]))
        y = parse(Float64, strip(partes[2]))
        return x, y
    end

    function mostrar_resultado(w, b, r2_val, costo, n)
        texto = (
            "ajuste completado con $n puntos\n" *
            "  método               :  OLS (GLM.jl — ecuaciones normales)\n" *
            "─────────────────────────────────────\n" *
            "  w  (pendiente)       =  $(round(w,      digits=6))\n" *
            "  b  (intercepto)      =  $(round(b,      digits=6))\n" *
            "  R²                   =  $(round(r2_val, digits=6))\n" *
            "  j(w,b) = (1/2n)Σe²  =  $(round(costo,  digits=8))\n" *
            "─────────────────────────────────────\n" *
            "  ecuación ajustada    :  f(x) = $(round(w,digits=4))x + $(round(b,digits=4))"
        )
        set_gtk_property!(label_resultado, :label, texto)
    end

    # ============================================================
    # callbacks de los botones — cada uno llama a ajustar_modelo
    # en lugar de gradiente_descendiente
    # ============================================================

    signal_connect(boton_20, "clicked") do _
        try
            m     = ajustar_modelo(X_20, Y_20)
            costo = imprimir_resultados(m, X_20, Y_20, "20 puntos")
            w, b, r2_val = extraer_parametros(m)
            mostrar_resultado(w, b, r2_val, costo, 20)
            graficar(X_20, Y_20, m, "regresión lineal (GLM) — 20 puntos")
        catch err
            set_gtk_property!(label_resultado, :label, "error: $(string(err))")
        end
    end

    signal_connect(boton_50, "clicked") do _
        try
            m     = ajustar_modelo(X_50, Y_50)
            costo = imprimir_resultados(m, X_50, Y_50, "50 puntos")
            w, b, r2_val = extraer_parametros(m)
            mostrar_resultado(w, b, r2_val, costo, 50)
            graficar(X_50, Y_50, m, "regresión lineal (GLM) — 50 puntos")
        catch err
            set_gtk_property!(label_resultado, :label, "error: $(string(err))")
        end
    end

    signal_connect(boton_100, "clicked") do _
        try
            m     = ajustar_modelo(X_100, Y_100)
            costo = imprimir_resultados(m, X_100, Y_100, "100 puntos")
            w, b, r2_val = extraer_parametros(m)
            mostrar_resultado(w, b, r2_val, costo, 100)
            graficar(X_100, Y_100, m, "regresión lineal (GLM) — 100 puntos")
        catch err
            set_gtk_property!(label_resultado, :label, "error: $(string(err))")
        end
    end

    signal_connect(boton_manual, "clicked") do _
        try
            x1, y1 = parsear_punto(entry_p1)
            x2, y2 = parsear_punto(entry_p2)
            x3, y3 = parsear_punto(entry_p3)
            x_man = [x1, x2, x3]
            y_man = [y1, y2, y3]

            m     = ajustar_modelo(x_man, y_man)
            costo = imprimir_resultados(m, x_man, y_man, "3 puntos manuales")
            w, b, r2_val = extraer_parametros(m)
            mostrar_resultado(w, b, r2_val, costo, 3)
            graficar(x_man, y_man, m, "regresión lineal (GLM) — 3 puntos manuales")
        catch err
            set_gtk_property!(label_resultado, :label, "error al parsear puntos: $(string(err))")
        end
    end

    # ============================================================
    # mostrar ventana y entrar al loop de eventos
    # ============================================================
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


# ============================================================
# punto de entrada
# ============================================================
lanzar_aplicacion()
