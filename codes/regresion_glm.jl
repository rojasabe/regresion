# regresion_glm.jl
# regresion lineal con GLM.jl -- minimos cuadrados ordinarios (OLS) nativo
# GUI al estilo de la version a mano (guinativa) pero ajuste en un solo paso
# Autor: Porfirio Rojas

using Gtk
using Plots
using GLM
using DataFrames
plotlyjs()   # backend con zoom y hover en el navegador

# los mismos 4 datasets que la version a mano, para poder comparar resultados
const X_3 = [1.0, 2.0, 3.0]
const Y_3 = [2.0, 4.0, 5.5]

const X_5 = [1.0, 2.0, 3.0, 4.0, 5.0]
const Y_5 = [1.5, 3.0, 4.0, 5.5, 7.0]


# OLS resuelve las ecuaciones normales (X'X) beta = X'y en un solo paso.
# lm() devuelve el modelo y de ahi sacamos todo nativamente.
function ajustar(x_datos, y_datos)
    df = DataFrame(x = x_datos, y = y_datos)
    return lm(@formula(y ~ x), df)
end


# RSS(w, b) = sum (y - (wx + b))^2  -- lo que OLS minimiza, sin 1/2m.
# auxiliar para dibujar la parabola de costo en funcion de w.
function rss_de(w_val, b_val, x_datos, y_datos)
    return sum((w_val .* x_datos .+ b_val .- y_datos) .^ 2)
end


# 4 paneles al estilo guinativa, pero todos con contenido OLS nativo.
function graficar(x_datos, y_datos, modelo, titulo)
    coefs  = coef(modelo)
    b, w   = coefs[1], coefs[2]            # b = intercepto, w = pendiente
    rss    = deviance(modelo)              # suma de cuadrados de residuos (lo que OLS minimiza)
    r2_v   = r2(modelo)
    n      = Int(nobs(modelo))             # GLM.nobs devuelve Float64, lo casteamos para poder indexar arrays con 1:n
    dof    = dof_residual(modelo)
    rse    = sqrt(rss / dof)               # residual standard error (sigma_hat)
    resid  = residuals(modelo)
    yhat   = predict(modelo)

    # grilla densa para la recta y para la banda de confianza
    x_eje  = collect(range(minimum(x_datos), maximum(x_datos), length = 200))
    df_new = DataFrame(x = x_eje)
    pred   = predict(modelo, df_new; interval = :confidence, level = 0.95)
    y_fit  = pred.prediction
    y_lo   = pred.lower
    y_hi   = pred.upper

    # 1) datos + recta OLS + banda de confianza al 95%
    y_all = vcat(y_datos, y_fit)
    pad1y = (maximum(y_all) - minimum(y_all)) * 0.20
    pad1x = (maximum(x_datos) - minimum(x_datos)) * 0.10

    p1 = scatter(x_datos, y_datos,
        label      = "datos ($(n) puntos)",
        xlabel     = "x",
        ylabel     = "y",
        title      = titulo,
        markersize = 7,
        color      = :steelblue,
        legend     = :topleft
    )
    plot!(p1, x_eje, y_fit,
        ribbon    = (y_fit .- y_lo, y_hi .- y_fit),
        fillalpha = 0.2,
        label     = "OLS:  y = $(round(w, digits = 4))x + $(round(b, digits = 4))",
        color     = :firebrick,
        linewidth = 2.5
    )
    xlims!(p1, (minimum(x_datos) - pad1x, maximum(x_datos) + pad1x))
    ylims!(p1, (minimum(y_all)   - pad1y, maximum(y_all)   + pad1y))

    # 2) residuos como segmentos verticales: cada naranja es un (y_i - y_hat_i)
    # que entra al RSS. visualizacion directa de lo que OLS minimiza.
    p2 = scatter(x_datos, y_datos,
        label      = "observado",
        xlabel     = "x",
        ylabel     = "y",
        title      = "residuos OLS  --  RSS = $(round(rss, digits = 6))",
        markersize = 7,
        color      = :steelblue,
        legend     = :topleft
    )
    plot!(p2, x_eje, w .* x_eje .+ b,
        label     = "recta ajustada",
        color     = :firebrick,
        linewidth = 2
    )
    for i in 1:n
        plot!(p2, [x_datos[i], x_datos[i]], [y_datos[i], yhat[i]],
            label     = (i == 1 ? "residuo y - ŷ" : ""),
            color     = :darkorange,
            linewidth = 1.8,
            alpha     = 0.75
        )
    end
    scatter!(p2, x_datos, yhat,
        label       = "predicho ŷ",
        color       = :darkorange,
        markersize  = 5,
        markershape = :diamond
    )

    # 3) parabola RSS(w) con b fijado en b*  (analogo al panel 3 de guinativa).
    # OLS no itera: solo marcamos el minimo w* en verde, no hay trayectoria.
    margen = max(abs(w) * 0.8, 1.0)
    w_eje  = collect(range(w - margen, w + margen, length = 400))
    j_eje  = [rss_de(wv, b, x_datos, y_datos) for wv in w_eje]

    p3 = plot(w_eje, j_eje,
        label     = "",
        xlabel    = "w",
        ylabel    = "RSS(w, b*)",
        title     = "superficie de costo RSS(w)  --  w* = $(round(w, digits = 4))",
        color     = :mediumpurple,
        linewidth = 2
    )
    scatter!(p3, [w], [rss],
        label       = "w* = $(round(w, digits = 4))   RSS = $(round(rss, digits = 4))",
        color       = :green,
        markersize  = 11,
        markershape = :diamond
    )

    # 4) residuos vs predichos: diagnostico OLS clasico.
    # si los puntos no muestran patron alrededor de cero, el supuesto se sostiene.
    p4 = scatter(yhat, resid,
        xlabel     = "valor predicho  ŷ",
        ylabel     = "residuo  y - ŷ",
        title      = "residuos vs predichos  --  R² = $(round(r2_v, digits = 4))   RSE = $(round(rse, digits = 4))",
        markersize = 7,
        color      = :steelblue,
        label      = ""
    )
    hline!(p4, [0],
        label     = "cero",
        color     = :firebrick,
        linewidth = 1.5,
        linestyle = :dash
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


function lanzar_aplicacion()

    ventana = GtkWindow("regresion lineal -- GLM (OLS nativo)", 620, 620)
    caja    = GtkBox(:v)
    set_gtk_property!(caja, :spacing,       8)
    set_gtk_property!(caja, :margin_top,    12)
    set_gtk_property!(caja, :margin_bottom, 12)
    set_gtk_property!(caja, :margin_start,  12)
    set_gtk_property!(caja, :margin_end,    12)

    titulo_lbl = GtkLabel(
        "<b>regresion lineal con GLM.jl</b>\n" *
        "metodo: minimos cuadrados ordinarios (OLS)\n" *
        "modelo: y = wx + b   --   ajuste por ecuaciones normales en un solo paso"
    )
    set_gtk_property!(titulo_lbl, :use_markup, true)
    set_gtk_property!(titulo_lbl, :justify,    2)

    function fila_campo(etiqueta, valor_default, ancho = 10)
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

    # marco con los datasets predefinidos (mismos que la version a mano)
    marco_pred = GtkFrame("  conjuntos de datos predefinidos  ")
    caja_pred  = GtkBox(:v)
    set_gtk_property!(caja_pred, :spacing,       6)
    set_gtk_property!(caja_pred, :margin_top,    6)
    set_gtk_property!(caja_pred, :margin_bottom, 6)
    set_gtk_property!(caja_pred, :margin_start,  8)
    set_gtk_property!(caja_pred, :margin_end,    8)

    boton_3 = GtkButton("3 puntos  : (1,2.0), (2,4.0), (3,5.5)")
    boton_5 = GtkButton("5 puntos  : (1,1.5), (2,3.0), (3,4.0), (4,5.5), (5,7.0)")

    for btn in [boton_3, boton_5]
        push!(caja_pred, btn)
    end
    push!(marco_pred, caja_pred)

    # marco para entrada manual
    marco_manual = GtkFrame("  entrada manual (valores separados por coma)  ")
    caja_manual  = GtkBox(:v)
    set_gtk_property!(caja_manual, :spacing,       6)
    set_gtk_property!(caja_manual, :margin_top,    6)
    set_gtk_property!(caja_manual, :margin_bottom, 6)
    set_gtk_property!(caja_manual, :margin_start,  8)
    set_gtk_property!(caja_manual, :margin_end,    8)

    fila_mx, entry_x = fila_campo("valores x: ", "1, 2, 3", 28)
    fila_my, entry_y = fila_campo("valores y: ", "2, 4, 5.5", 28)
    boton_manual     = GtkButton("ajustar OLS con los puntos ingresados")

    push!(caja_manual, fila_mx)
    push!(caja_manual, fila_my)
    push!(caja_manual, boton_manual)
    push!(marco_manual, caja_manual)

    # marco para resultados
    marco_res = GtkFrame("  resultados del ajuste OLS  ")
    label_res = GtkLabel("presiona un boton para ajustar el modelo...")
    set_gtk_property!(label_res, :margin_top,    10)
    set_gtk_property!(label_res, :margin_bottom, 10)
    set_gtk_property!(label_res, :margin_start,  10)
    set_gtk_property!(label_res, :margin_end,    10)
    set_gtk_property!(label_res, :xalign,        0.0f0)
    push!(marco_res, label_res)

    for widget in [titulo_lbl, marco_pred, marco_manual, marco_res]
        push!(caja, widget)
    end
    push!(ventana, caja)

    # actualiza el panel de texto con los estadisticos OLS nativos:
    # coeficientes con SE/t/p, RSS, RSE, R^2 y R^2 ajustado.
    function mostrar_resultado(modelo)
        ct     = coeftable(modelo)
        coefs  = coef(modelo)
        b, w   = coefs[1], coefs[2]
        se     = ct.cols[2]
        tval   = ct.cols[3]
        pval   = ct.cols[4]
        rss    = deviance(modelo)
        r2_v   = r2(modelo)
        adj_v  = adjr2(modelo)
        n      = Int(nobs(modelo))
        dof    = Int(dof_residual(modelo))
        rse    = sqrt(rss / dof)

        texto = (
            "ajuste OLS con $(n) puntos completado\n" *
            "  w  = $(round(w, digits = 6))   SE = $(round(se[2], digits = 6))   t = $(round(tval[2], digits = 3))   p = $(round(pval[2], digits = 4))\n" *
            "  b  = $(round(b, digits = 6))   SE = $(round(se[1], digits = 6))   t = $(round(tval[1], digits = 3))   p = $(round(pval[1], digits = 4))\n" *
            "  RSS = $(round(rss, digits = 6))    RSE = $(round(rse, digits = 6))    df = $(dof)\n" *
            "  R² = $(round(r2_v, digits = 6))    R² ajustado = $(round(adj_v, digits = 6))\n" *
            "  ecuacion : y = $(round(w, digits = 4))x + $(round(b, digits = 4))"
        )
        set_gtk_property!(label_res, :label, texto)
    end

    function ajustar_y_graficar(X, Y, titulo_graf)
        try
            modelo = ajustar(X, Y)
            mostrar_resultado(modelo)
            graficar(X, Y, modelo, titulo_graf)
        catch err
            set_gtk_property!(label_res, :label, "error: $(string(err))")
        end
    end

    signal_connect(boton_3, "clicked") do _
        ajustar_y_graficar(X_3, Y_3, "regresion OLS (GLM) -- 3 puntos")
    end

    signal_connect(boton_5, "clicked") do _
        ajustar_y_graficar(X_5, Y_5, "regresion OLS (GLM) -- 5 puntos")
    end

    signal_connect(boton_manual, "clicked") do _
        try
            xs = parse.(Float64, strip.(split(get_gtk_property(entry_x, :text, String), ',')))
            ys = parse.(Float64, strip.(split(get_gtk_property(entry_y, :text, String), ',')))
            length(xs) == length(ys) || error("la cantidad de valores x e y debe ser igual")
            length(xs) >= 3            || error("se requieren al menos 3 puntos para que el modelo tenga grados de libertad")
            modelo = ajustar(xs, ys)
            mostrar_resultado(modelo)
            graficar(xs, ys, modelo, "regresion OLS (GLM) -- puntos manuales")
        catch err
            set_gtk_property!(label_res, :label, "error: $(string(err))")
        end
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
