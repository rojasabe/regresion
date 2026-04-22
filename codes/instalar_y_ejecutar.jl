#archivo para instalar librerías nuevas o actualizarlas

using Pkg

println("instalando dependencias...")
Pkg.add("Gtk")
Pkg.add("Plots")
Pkg.add("PlotlyJS")#backend interactivo de julia
Pkg.add("GR")
Pkg.add("Pluto")
Pkg.add("PlutoUI")
Pkg.add(["GLM", "DataFrames"])
println("dependencias instaladas exitosamente")
println()
println("ejecutando la aplicacion en Pluto...")
println("el navegador se abrira automaticamente con el notebook")
println("para cerrar: Ctrl+C en esta terminal")
println()
using Pluto
Pluto.run(notebook = joinpath(@__DIR__, "regresion_pluto.jl"))
