#regresion lineal con GLM
#modelo: y(predictiva)= w*x + b
#método: 


using GLM, DataFrames, Plots, Gtk, Statistics

#generar datos de ejemplo que sean estáticos 5x y 5y
x = [1,3,6,9,12]
y = [2,4,5,7,8]
#generar otros datos de ejemplo pero 3x y 3y
x2 = [2,4,7]
y2 = [3,5,6]

#crear un dataframe con los datos
df = DataFrame(x=x, y=y)
#despues ajustamos el modelo de rgresion con GLM
modelo = lm(@formula(y ~ x), df)
#mostrar los coeficientes del modelo
coeficientes = coef(modelo)
println("coeficientes del modelo: ", coeficientes)
#ahora hacer las predicciones con el modelo
predicciones = predict(modelo)
println("predicciones del modelo: ", predicciones)

#ahora graficamos los datos y la línea regresion
regresion_figura = scatter(x,y, label="datos", title="regresion con GLM", xlabel="x", ylabel="y")
plot!(regresion_figura, x, predicciones, label="modelo", color=:red)

#guardamos la figura y la mostramos en gtk
savefig(regresion_figura, "regresion_glm.png")
imagen = Gtk.GtkImage("regresion_glm.png")
ventana_gtk = Gtk.GtkWindow("Regresion GLM", 400, 300)

push!(ventana_gtk, imagen)
show(ventana_gtk)

#para mantener viva la ventana
if !isinteractive()
    condicion = Condition()
    signal_connect(ventana_gtk, "destroy") do widget
        notify(condicion)
    end
    @async Gtk.gtk_main()
    wait(condicion)
end



