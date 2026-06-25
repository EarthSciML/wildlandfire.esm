# Camp Fire Spread Simulation
# Rothermel + LevelSet + LANDFIRE fuel + USGS3DEP terrain + ERA5 wind + ArrayOp vectorized code generation.
using WildlandFire
using EarthSciMLBase
using EarthSciData: LANDFIRE, USGS3DEP, ERA5
using ModelingToolkit
using ModelingToolkit: t, D
using MethodOfLines
using OrdinaryDiffEqSSPRK
using CairoMakie
using Tyler, Tyler.TileProviders
using Proj
using DomainSets
using DynamicQuantities
using Dates
using Symbolics, SymbolicUtils
using SciMLBase
using DiffEqCallbacks: PeriodicCallback, CallbackSet
using Downloads, TiffImages
using LinearAlgebra
using Statistics: mean
using ProgressLogging


# ============================================================================
# Configuration — change resolution here
# ============================================================================
const DX = 2000.0            # Grid spacing (meters). Try 2000, 1000, 500, 250.
const DT = DX / 400.0      # Solver timestep (seconds), scales with CFL condition
const IGNITION_RADIUS = max(DX * 1.5, 500.0)  # Must be > dx for initial fire
const SAVEAT = 600.0         # Save interval (seconds)
const SIM_HOURS = 16.0       # Simulation duration (hours)
const DOMAIN_X_HALF = 18000.0  # Half-width of domain (meters)
const DOMAIN_Y_HALF = 20000.0  # Half-height of domain (meters)

# ============================================================================
# 1. Domain setup — Camp Fire area in Lambert Conformal projection
# ============================================================================
proj_string = "+proj=lcc +lat_1=30.0 +lat_2=60.0 +lat_0=39.0 +lon_0=-97.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

lonlat_to_lcc = Proj.Transformation("EPSG:4326", proj_string; always_xy=true)
x_ign, y_ign = lonlat_to_lcc(-121.44, 39.81)   # Pulga (ignition)
x_par, y_par = lonlat_to_lcc(-121.62, 39.76)    # Paradise
x_center = (x_ign + x_par) / 2
y_center = (y_ign + y_par) / 2

x_half = DX * round(DOMAIN_X_HALF / DX)
y_half = DX * round(DOMAIN_Y_HALF / DX)
xrange = (x_center - x_half):DX:(x_center + x_half)
yrange = (y_center - y_half):DX:(y_center + y_half)

println("Resolution: dx=$(DX)m, dt=$(DT)s")
println("Ignition (Pulga): x=$(round(Int,x_ign)) y=$(round(Int,y_ign))")
println("Paradise:         x=$(round(Int,x_par)) y=$(round(Int,y_par))")
println("Domain: $(length(xrange))×$(length(yrange)) grid")
flush(stdout)

domain = DomainInfo(
    DateTime(2018, 11, 8, 14, 30),
    DateTime(2018, 11, 8, 14, 30) + Second(round(Int, SIM_HOURS * 3600));
    xrange = xrange,
    yrange = yrange,
    spatial_ref = proj_string,
)

# ERA5 requires a 3D domain (with pressure levels)
domain_3d = DomainInfo(
    DateTime(2018, 11, 8, 14, 30),
    DateTime(2018, 11, 8, 14, 30) + Second(round(Int, SIM_HOURS * 3600));
    xrange = xrange,
    yrange = yrange,
    levrange = 1:5,  # MOL needs ≥3 points for stencil; 5 is a safe default
    spatial_ref = proj_string,
)

# ============================================================================
# 2. Create and couple fire model components with real terrain and fuel data
# ============================================================================
println("Creating fire model with LANDFIRE + USGS3DEP + ERA5 + fuel moisture + directional spread + fuel consumption...")

r = RothermelFireSpread()
ls = LevelSetFireSpread(domain;
    initial_condition = (xx, yy) -> sqrt((xx - x_ign)^2 + (yy - y_ign)^2) - IGNITION_RADIUS,
)

# Data-driven components
lf = LANDFIRE(domain)           # Fuel model codes from LANDFIRE
dep = USGS3DEP(domain)          # Elevation, dzdx, dzdy from USGS 3DEP
era = ERA5(domain_3d; variables=["u_component_of_wind", "v_component_of_wind",
    "temperature", "relative_humidity"])

# Fire behavior components
fm = FuelModelLookup()          # Maps fuel codes → Rothermel parameters (σ, w0, δ, Mx, h)
ts = TerrainSlope()             # Computes tanϕ from dzdx, dzdy
mw = MidflameWind()             # Converts 10m wind → midflame height wind speed
fc = FuelConsumption()          # Fuel consumption feedback (burned fuel → R=0)

# Fuel moisture from ERA5 weather
emc = EquilibriumMoistureContent()  # EMC from temperature + relative humidity
fm1 = OneHourFuelMoisture()         # Fine dead fuel moisture → Rothermel Mf

# Couple everything:
cs = couple(r, ls, fm, ts, mw, fc, emc, fm1, lf, dep, era, domain)
pde = convert(PDESystem, cs)

println("Num equations: ", length(equations(pde)))
println("Num DVs: ", length(pde.dvs))
for (i, eq) in enumerate(equations(pde))
    println("  $i: $eq")
end
println("\nDVs:")
for dv in pde.dvs
    println("  ", dv)
end

println("PDESystem: $(length(equations(pde))) eqs, $(length(pde.dvs)) dvs")
flush(stdout)


println("Discretizing...")
flush(stdout)
t_disc = @elapsed begin
    disc = MOLFiniteDifference(
        [pde.ivs[i] =>
             (Symbolics.tosymbol(pde.ivs[i], escape = false) == :lev ? 1.0 : DX)
         for i in 2:length(pde.ivs)],
        pde.ivs[1];
        discretization_strategy = MethodOfLines.ArrayDiscretization(),
    )
    prob = MethodOfLines.discretize(pde, disc; checks = false, simplify = false)
end
println("Discretized: $(length(prob.u0)) unknowns in $(round(t_disc, digits=1))s")
flush(stdout)

simpsys = prob.f.sys
# Check if β_ratio is in unknowns or observed
obs_names = [string(Symbolics.tosymbol(eq.lhs, escape=false)) for eq in ModelingToolkit.observed(simpsys)]
filter(n -> occursin("β_ratio", n), obs_names) |> println
unk_names = [string(Symbolics.tosymbol(u, escape=false)) for u in ModelingToolkit.unknowns(simpsys)]
filter(n -> occursin("β_ratio", n), unk_names) |> println

println("Solving $(SIM_HOURS)h with dt=$(DT)s...")
flush(stdout)

t_solve = @elapsed begin
    sol = solve(prob, SSPRK33(); dt = DT, saveat = SAVEAT,
        progress=true, progress_steps=1)
end
println("Solution: $(sol.retcode), $(length(sol.t)) frames in $(round(t_solve, digits=1))s")
flush(stdout)

# ============================================================================
# 5. Extract ψ field using symbolic variable mapping
# ============================================================================
dvs = unknowns(pde)
psi_var = nothing
for (k, v) in enumerate(dvs)
    s = string(v)
    if occursin("ψ", s)
        psi_var = v
    end
end
println("Found ψ variable: $psi_var")

xs = nothing
ys = nothing
for (k, v) in enumerate(pde.ivs)
    s = string(v)
    if occursin("x", s)
        xs = v
    elseif occursin("y", s)
        ys = v
    end
end

# Extract ψ, x, y grids from symbolic solution
psi_data = sol[psi_var]    # Array indexed by (x_i, y_j, time_k)
x_grid = sol[xs]           # x coordinate values
y_grid = sol[ys]           # y coordinate values
times = sol.t

nx = length(x_grid)
ny = length(y_grid)

# psi_data is (t, x, y) — use directly
psi = psi_data
println("ψ field: $(size(psi)) (t×x×y)")

ign_ix = argmin(abs.(x_grid .- x_ign))
ign_iy = argmin(abs.(y_grid .- y_ign))
par_ix = argmin(abs.(x_grid .- x_par))
par_iy = argmin(abs.(y_grid .- y_par))
println("ψ grid: $(nx)×$(ny), Initial ψ@ignition=$(round(psi[1, ign_ix, ign_iy], digits=0)), ψ@Paradise=$(round(psi[1, par_ix, par_iy], digits=0))")
flush(stdout)

# ============================================================================
# 6. Download elevation and satellite imagery for visualization
# ============================================================================
println("Downloading elevation and imagery...")
flush(stdout)
lcc_to_lonlat = Proj.Transformation(proj_string, "EPSG:4326"; always_xy=true)
corners = [lcc_to_lonlat(x_grid[i], y_grid[j]) for i in [1, nx] for j in [1, ny]]
lons = [c[1] for c in corners]; lats = [c[2] for c in corners]

# Higher-res imagery grid (4x the simulation grid for visual quality)
img_nx, img_ny = min(nx * 4, 1024), min(ny * 4, 1024)

# Download elevation
elevation = zeros(Float64, img_nx, img_ny)
try
    elev_url = string(
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?",
        "bbox=$(minimum(lons)),$(minimum(lats)),$(maximum(lons)),$(maximum(lats))",
        "&bboxSR=4326&imageSR=4326&size=$(img_nx),$(img_ny)&format=tiff&pixelType=F32&f=image",
    )
    p = Downloads.download(elev_url, joinpath(@__DIR__, "elevation.tif"))
    img = TiffImages.load(p)
    h, w = size(img)
    for col in 1:min(w, img_nx), row in 1:min(h, img_ny)
        elevation[col, h - row + 1] = Float64(img[row, col])
    end
    println("Elevation: $(round(minimum(elevation[elevation.>0])))–$(round(maximum(elevation))) m")
catch e
    @warn "Elevation download failed: $e"
end

# Download ESRI World Imagery satellite basemap as PNG
satellite = Matrix{RGBAf}(undef, img_nx, img_ny)
fill!(satellite, RGBAf(0.3, 0.4, 0.2, 1.0))  # fallback green
try
    sat_url = string(
        "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export?",
        "bbox=$(minimum(lons)),$(minimum(lats)),$(maximum(lons)),$(maximum(lats))",
        "&bboxSR=4326&imageSR=4326&size=$(img_nx),$(img_ny)&format=png&f=image",
    )
    sat_path = Downloads.download(sat_url, joinpath(@__DIR__, "satellite.png"))
    sat_img = Makie.FileIO.load(sat_path)
    sh, sw = size(sat_img)
    for col in 1:min(sw, img_nx), row in 1:min(sh, img_ny)
        satellite[col, sh - row + 1] = RGBAf(sat_img[row, col])
    end
    println("Satellite imagery loaded: $(sw)×$(sh)")
catch e
    @warn "Satellite download failed, using fallback: $e"
end
flush(stdout)

# Compute hillshade from elevation
function compute_hillshade(elev; azimuth_deg=315.0, altitude_deg=45.0)
    _nx, _ny = size(elev)
    shade = ones(Float64, _nx, _ny)
    az = deg2rad(azimuth_deg)
    alt = deg2rad(altitude_deg)
    for j in 2:_ny-1, i in 2:_nx-1
        dzdx = (elev[i+1, j] - elev[i-1, j]) / 2
        dzdy = (elev[i, j+1] - elev[i, j-1]) / 2
        slope = atan(sqrt(dzdx^2 + dzdy^2))
        aspect = atan(dzdy, -dzdx)
        shade[i, j] = clamp(
            sin(alt) * cos(slope) + cos(alt) * sin(slope) * cos(az - aspect),
            0.15, 1.0)
    end
    shade
end
hillshade = compute_hillshade(elevation)

# Blend satellite imagery with hillshade and fire overlay
function make_surface_colors(sat, shade, burning)
    _nx, _ny = size(sat)
    colors = Matrix{RGBAf}(undef, _nx, _ny)
    for j in 1:_ny, i in 1:_nx
        s = Float32(0.4 * shade[i, j] + 0.6)
        if burning[i, j]
            colors[i, j] = RGBAf(clamp(1.0f0 * s, 0f0, 1f0), clamp(0.2f0 * s, 0f0, 1f0), 0.0f0, 1.0f0)
        else
            c = sat[i, j]
            colors[i, j] = RGBAf(clamp(c.r * s, 0f0, 1f0), clamp(c.g * s, 0f0, 1f0), clamp(c.b * s, 0f0, 1f0), 1.0f0)
        end
    end
    return colors
end

# Interpolate ψ from simulation grid to imagery grid for visualization
function interp_psi_to_img(psi_frame, x_grid, y_grid, img_nx, img_ny, x_half, y_half, x_center, y_center)
    img_x = range(x_center - x_half, x_center + x_half, length=img_nx)
    img_y = range(y_center - y_half, y_center + y_half, length=img_ny)
    result = zeros(img_nx, img_ny)
    for jj in 1:img_ny, ii in 1:img_nx
        # Nearest-neighbor interpolation from simulation grid
        xi = clamp(round(Int, (img_x[ii] - x_grid[1]) / (x_grid[2] - x_grid[1])) + 1, 1, length(x_grid))
        yi = clamp(round(Int, (img_y[jj] - y_grid[1]) / (y_grid[2] - y_grid[1])) + 1, 1, length(y_grid))
        result[ii, jj] = psi_frame[xi, yi]
    end
    result
end

x_km = range(x_center - x_half, x_center + x_half, length=img_nx) ./ 1000.0
y_km = range(y_center - y_half, y_center + y_half, length=img_ny) ./ 1000.0
z_exag = 1.5
elev_scaled = elevation .* z_exag ./ 1000.0

# ============================================================================
# 7. Static fire front plot using Tyler.jl satellite basemap
# ============================================================================
println("Creating fire front plot...")
flush(stdout)

# Convert simulation grid to full 2D lon/lat for accurate contour plotting
lon_grid = zeros(nx, ny)
lat_grid = zeros(nx, ny)
for i in 1:nx, j in 1:ny
    lon_grid[i, j], lat_grid[i, j] = lcc_to_lonlat(x_grid[i], y_grid[j])
end

# Tyler.Map with tight extent matching simulation grid
extent = Tyler.Extent(X = extrema(lon_grid), Y = extrema(lat_grid))
fig2 = Figure(size = (1000, 900))
m = Tyler.Map(extent; provider=TileProviders.Esri(:WorldImagery), figure=fig2,
    crs=Tyler.wgs84, axis=(; aspect=DataAspect()))
ax2 = m.axis
ax2.title = "Camp Fire — Fire Front Progression (dx=$(DX)m)"
ax2.xlabel = "Longitude"
ax2.ylabel = "Latitude"

# contour! needs a rectilinear grid. Use the ignition row/col as reference
# so the contours align with the ignition marker.
ign_j = argmin(abs.(y_grid .- y_ign))
ign_i = argmin(abs.(x_grid .- x_ign))
x_lon_sim = lon_grid[:, ign_j]
y_lat_sim = lat_grid[ign_i, :]
xlims!(ax2, extrema(x_lon_sim))
ylims!(ax2, extrema(y_lat_sim))
front_hours = 0:2:Int(SIM_HOURS)
front_colors_cg = cgrad(:inferno, length(front_hours); categorical=true)
legend_entries = []
legend_labels = String[]
for (idx, hr) in enumerate(front_hours)
    target_t = hr * 3600.0
    fi = argmin(abs.(times .- target_t))
    actual_hr = times[fi] / 3600.0
    contour!(ax2, x_lon_sim, y_lat_sim, psi[fi, :, :];
        levels = [0.0], color = front_colors_cg[idx], linewidth = 2.5)
    push!(legend_entries, LineElement(color = front_colors_cg[idx], linewidth = 2.5))
    push!(legend_labels, "t = $(round(actual_hr, digits=1))h")
end

# Markers in lon/lat
ign_lon, ign_lat = lcc_to_lonlat(x_ign, y_ign)
par_lon, par_lat = lcc_to_lonlat(x_par, y_par)
scatter!(ax2, [ign_lon], [ign_lat]; color = :red, markersize = 15, marker = :star5)
scatter!(ax2, [par_lon], [par_lat]; color = :white, markersize = 12,
    marker = :diamond, strokewidth = 2, strokecolor = :black)
push!(legend_entries, MarkerElement(color = :red, marker = :star5, markersize = 15))
push!(legend_labels, "Ignition (Pulga)")
push!(legend_entries, MarkerElement(color = :white, marker = :diamond, markersize = 12,
    strokewidth = 2, strokecolor = :black))
push!(legend_labels, "Paradise")

Legend(fig2[1, 2], legend_entries, legend_labels)

wait(m)  # Wait for tiles to load
# Re-apply limits after tiles load (Tyler may have expanded them)
xlims!(ax2, extrema(x_lon_sim))
ylims!(ax2, extrema(y_lat_sim))
static_path = joinpath(@__DIR__, "camp_fire_fronts.png")
save(static_path, fig2; px_per_unit = 2)
println("Saved: $static_path")
flush(stdout)

# ============================================================================
# 8. 3D Surface Animation (more overhead view)
# ============================================================================
println("Creating animation...")
flush(stdout)

psi_img_0 = interp_psi_to_img(psi[1, :, :], x_grid, y_grid, img_nx, img_ny, x_half, y_half, x_center, y_center)
color_obs = Observable(make_surface_colors(satellite, hillshade, psi_img_0 .< 0))
time_obs = Observable("Camp Fire — t = 0h 00m")

fig = Figure(size = (1100, 900), backgroundcolor = :gray10)
ax = Axis3(fig[1, 1];
    xlabel = "East-West (km)", ylabel = "North-South (km)",
    zlabel = "Elevation (km, $(z_exag)× exag.)",
    title = time_obs,
    azimuth = -0.4π, elevation = 0.35π, perspectiveness = 0.4,
)

surface!(ax, collect(x_km), collect(y_km), elev_scaled; color = color_obs, shading = NoShading)

# Markers at ignition and Paradise
ign_elev_viz = z_exag * elevation[
    clamp(round(Int, (x_ign - (x_center-x_half)) / (2*x_half) * (img_nx-1)) + 1, 1, img_nx),
    clamp(round(Int, (y_ign - (y_center-y_half)) / (2*y_half) * (img_ny-1)) + 1, 1, img_ny)] / 1000.0
par_elev_viz = z_exag * elevation[
    clamp(round(Int, (x_par - (x_center-x_half)) / (2*x_half) * (img_nx-1)) + 1, 1, img_nx),
    clamp(round(Int, (y_par - (y_center-y_half)) / (2*y_half) * (img_ny-1)) + 1, 1, img_ny)] / 1000.0
scatter!(ax, [x_ign / 1000], [y_ign / 1000], [ign_elev_viz + 0.1];
    color = :red, markersize = 15, marker = :star5)
scatter!(ax, [x_par / 1000], [y_par / 1000], [par_elev_viz + 0.1];
    color = :yellow, markersize = 12, marker = :diamond)

output_path = joinpath(@__DIR__, "camp_fire.gif")
record(fig, output_path, eachindex(times); framerate = 6) do i
    psi_img = interp_psi_to_img(psi[i, :, :], x_grid, y_grid, img_nx, img_ny, x_half, y_half, x_center, y_center)
    color_obs[] = make_surface_colors(satellite, hillshade, psi_img .< 0)
    local hh = Int(floor(times[i] / 3600))
    local mm = Int(floor((times[i] % 3600) / 60))
    time_obs[] = "Camp Fire — t = $(hh)h $(lpad(mm, 2, '0'))m"
end
println("Saved: $output_path")
println("Done!")
