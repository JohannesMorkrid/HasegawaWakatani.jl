using HasegawaWakatani
using HDF5
using FFMPEG
using Plots

file = h5open("heat_equation.h5", "r")
sol = last(collect(read(file)))[2]

# Create a GIF
gif_filename = "heat_evolution.gif"
@gif for t in 1:size(sol["fields"], 3)
    heatmap(sol["fields"][:, :, t], title="Time: $t", xlabel="X", ylabel="Y", c=:viridis)
end every 10

print("LOL")