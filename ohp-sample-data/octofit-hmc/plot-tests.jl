using GLMakie
using Octofitter
using Distributions
using CSV, DataFrames
# df = CSV.read(joinpath(@__DIR__, "target_1.csv"), DataFrame)
ENV["JULIA_SSL_NO_VERIFY_HOSTS"] = "*"
# fname = download("https://dace.unige.ch/downloads/open_data/dace-gaia-ohp/files/target_1.csv")
df = CSV.read("gaia_dr4_sim2.csv", DataFrame);
ref_epoch_mjd = 57936.375

gaia_dr4_obs = GaiaDR4AstromObs(
    df,
    # For plotting reasons, you need to supply a Gaia ID to know e.g. the absolute Ra and Dec
    # For these ~simulated examples, you can pick one at random
    gaia_id=4373465352415301632,
    primary_star_perturbation=false,
    variables=@variables begin
        astrometric_jitter ~ LogUniform(0.00001, 10) # mas
        ra_offset_mas  ~ Normal(0, 10000)
        dec_offset_mas ~ Normal(0, 10000)
        pmra ~ Uniform(-1000, 1000) # mas/yr
        pmdec ~  Uniform(-1000, 1000) # mas/yr
        # ra_offset_mas = -0.00154
        # dec_offset_mas = 0.0019354
        # pmra = 5.421
        # pmdec = -24.121
        plx = system.plx
        ref_epoch = $ref_epoch_mjd
    end
)
mjup2msol = Octofitter.mjup2msol
ref_epoch_mjd = Octofitter.meta_gaia_DR3.ref_epoch_mjd
orbit_ref_epoch = mean(gaia_dr4_obs.table.epoch)

b = Planet(
    name="b",
    basis=Visual{KepOrbit},
    observations=[],
    variables=@variables begin

        P ~ LogUniform(0.001, 10000)
        a = cbrt(system.M * P^2)

        e ~ Uniform(0, 0.99)
        ω ~ Uniform(0,2pi)
        i ~ Sine()
        Ω ~ Uniform(0,2pi)
        θ ~ Uniform(0,2pi)
        tp = θ_at_epoch_to_tperi(θ, $orbit_ref_epoch; M=system.M, e, a, i, ω, Ω)
        mass = system.q*system.M_pri/Octofitter.mjup2msol
    end
)
sys = System(
    name="target_1",
    companions=[b],
    observations=[gaia_dr4_obs,],
    variables=@variables begin

        M_pri = 1.0
        q ~ LogUniform(1e-5, 1)
        M = M_pri * (1 + q)

        # Note: keep these physically plausible to prevent numerical errors
        plx ~ Uniform(0.01,100) # mas
    end
)
model = Octofitter.LogDensityModel(sys, verbosity=4)
# chain = Octofitter.loadchain("/Users/thompsonw/Downloads/dr4.fits")
##
# chain = Octofitter.loadchain("/Users/thompsonw/Downloads/dr4_target6.fits")

##
initialize!(model)
chain = octofit(model, max_depth=10)
Octofitter.savechain("/Users/thompsonw/Downloads/dr4_target2.fits", chain)
##
using Printf
##################################################
# Gaia Star Plot
# Shows the star's orbit in RA/Dec space for a single posterior sample
# Similar to astromplot but shows the star's motion due to companions,
# and like rvpostplot, only plots a single draw at a time since
# the detrending is different per draw.
# Wrap layout updates in an update block to avoid triggering multiple updates
function update(f, fig)
    layout = get_layout(fig)
    block_updates = layout.block_updates
    layout.block_updates = true
    output = f(fig)
    layout.block_updates = block_updates
    block_updates || Makie.GridLayoutBase.update!(layout)
    return output
end
get_layout(gl::Makie.GridLayout) = gl
get_layout(f::Union{Makie.Figure, Makie.GridPosition}) = f.layout
get_layout(l::Union{Makie.Block, Makie.GridSubposition}) = get_layout(l.parent)

function gaiastarplot(
    model,
    results,
    args...;
    fname="$(model.system.name)-gaiastarplot.png",
    kwargs...,
)
    fig = Figure(
        size=(600, 600)
    )
    gaiastarplot!(fig.layout, model, results, args...; kwargs...)

    Makie.save(fname, fig, px_per_unit=3)

    return fig
end

function gaiastarplot!(
    gs,
    model::Octofitter.LogDensityModel,
    results::Chains,
    sample_idx=argmax(results["logpost"][:]);
    axis=(;),
    colormap_instruments=Makie.cgrad(:Egypt,categorical=true)
)
    # if gridspec_or_fig isa Figure
    #     gs = GridLayout(gridspec_or_fig[1, 1])
    # else
    #     gs = gridspec_or_fig
    # end

    mjup2msol = Octofitter.mjup2msol

    # Find Gaia DR4 likelihood objects
    gaia_likes = filter(model.system.observations) do like_obj
        like_obj isa Octofitter.GaiaDR4AstromObs
    end
    if isempty(gaia_likes)
        error("No GaiaDR4AstromObs found in model")
    end
    likeobj = first(gaia_likes)

    # Get the sample
    θ_system = Octofitter.mcmcchain2result(model, results, sample_idx)
    θ_obs = θ_system.observations[Octofitter.normalizename(Octofitter.likelihoodname(likeobj))]

    # Construct orbits for this sample
    orbits = Tuple(map(keys(model.system.planets)) do planet_key
        Octofitter.construct_elements(model, results, planet_key, sample_idx)
    end)

    # Compute simulation at data epochs
    solutions = Tuple(map(orbits) do orbit
        return orbitsolve.(orbit, likeobj.table.epoch)
    end)
    centroid_pos_al_model_buffer = zeros(size(likeobj.table, 1))
    sim = Octofitter.simulate(
        likeobj,
        θ_system,
        θ_obs,
        orbits,
        solutions,
        0,
        centroid_pos_al_model_buffer
    )

    # Create axis
    if gs isa Axis
        ax = gs
    else
        ax = Axis(gs[1, 1],
            xreversed=true,
            autolimitaspect=1,
            xgridvisible=false,
            ygridvisible=false,
            xlabel="Δα* [mas]",
            ylabel="Δδ [mas]";
            axis...
        )
    end
    # vlines!(ax, 0, color=:grey, linestyle=:dash)
    # hlines!(ax, 0, color=:grey, linestyle=:dash)

    # Plot the Keplerian orbit of the star
    EAs = range(0, 2pi, length=150)
    Δα_kep = zeros(150)
    Δδ_kep = zeros(150)


    # Calculate residuals and project them back on the orbit
    resids = sim.along_scan_residuals_buffer .- likeobj.table.centroid_pos_al
    s = sin.(likeobj.table.scan_pos_angle)
    c = cos.(likeobj.table.scan_pos_angle)
    alpha_res = @. resids * s
    delta_res = @. resids * c

    # Calculate model position at data epochs
    Δα_kep_track = zeros(length(likeobj.table.epoch))
    Δδ_kep_track = zeros(length(likeobj.table.epoch))

    for planet_i in eachindex(orbits)
        sol = orbitsolve.(orbits[planet_i], likeobj.table.epoch)
        # Add perturbation from planet
        Δα_kep_track .+= raoff.(sol, θ_system.planets[planet_i].mass * mjup2msol)
        Δδ_kep_track .+= decoff.(sol, θ_system.planets[planet_i].mass * mjup2msol)
    end

    # Plot error bars along scan direction
    σ = likeobj.table.centroid_pos_error_al
    x_centers = Δα_kep_track .+ alpha_res
    y_centers = Δδ_kep_track .+ delta_res

    n = length(σ)
    x_bars = Vector{Float64}(undef, 3n)
    y_bars = Vector{Float64}(undef, 3n)

    for i in 1:n
        x_bars[3i-2] = x_centers[i] - σ[i] * s[i]
        y_bars[3i-2] = y_centers[i] - σ[i] * c[i]
        x_bars[3i-1] = x_centers[i] + σ[i] * s[i]
        y_bars[3i-1] = y_centers[i] + σ[i] * c[i]
        x_bars[3i] = NaN
        y_bars[3i] = NaN
    end


    # Connect data points to their corresponding positions on the Keplerian orbit
    for i in 1:n
        lines!(ax, 
            [Δα_kep_track[i] + alpha_res[i], Δα_kep_track[i]],
            [Δδ_kep_track[i] + delta_res[i], Δδ_kep_track[i]],
            color=:grey,
            linestyle=:dot,
            linewidth=1
        )
    end

    lines!(ax, x_bars, y_bars, color=:black)

    # Plot data points
    scatter!(ax,
        Δα_kep_track .+ alpha_res,
        Δδ_kep_track .+ delta_res,
        # color=colormap_instruments[1], # gaia
        color=likeobj.table.epoch,
        # color=likeobj.table.scan_pos_angle,
        # colormap=:cyclic_mrybm_35_75_c68_n256,
        colormap=:twilight,
        strokewidth=1,
        strokecolor=:black,
        # markersize=4,
    )


    # Plot the Keplerian orbit of the star
    EAs = range(0, 2pi, length=150)
    Δα_kep = zeros(150)
    Δδ_kep = zeros(150)

    for planet_i in eachindex(orbits)
        orbit = orbits[planet_i]
        # Orbit is perfectly periodic, so take equal steps in eccentric anomaly
        sols = orbitsolve_eccanom.(orbit, EAs)
        # Add perturbation from planet
        Δα_kep .+= raoff.(sols, θ_system.planets[planet_i].mass * mjup2msol)
        Δδ_kep .+= decoff.(sols, θ_system.planets[planet_i].mass * mjup2msol)
        txt = @sprintf("P=%.2g d\ne=%.2f",period(orbit),eccentricity(orbit))
        text!(ax, 0, 1, text=txt, space=:relative, align=(:left,:top))
    end

    lines!(ax, Δα_kep, Δδ_kep, color=Makie.wong_colors()[1], linewidth=2)


    xl = extrema(Δα_kep_track .+ alpha_res)
    yl = extrema(Δδ_kep_track .+ delta_res)

    # Plot star at origin
    scatter!(ax, [0], [0], marker='★', markersize=20, color=:white, strokecolor=:black, strokewidth=1.5)

    return ax, xl, yl
end

ff = Figure(size=(1024,600))
out = update(ff) do f
    Octofitter.astromplot!(GridLayout(f[1,1]),model, chain)#, mark_epochs_mjd=[mjd("2016")])

    fig = GridLayout(f[1,2])



    indices = rand(1:size(chain,1), 9)

    axis = (;
        topspinecolor="#AAA",
        bottomspinecolor="#AAA",
        leftspinecolor="#AAA",
        rightspinecolor="#AAA",
        xtickcolor="#AAA",
        ytickcolor="#AAA",
        xticklabelcolor="#666",
        yticklabelcolor="#666",
        # xlabelcolor="#AAA",
        # ylabelcolor="#AAA",
        # rightspinevisible=false,
        # topspinevisible=false,
        # xtrimspine=true,
        # ytrimspine=true,
        # xticks=1:-1:-1,
        # yticks=-1:1,
        titlesize=11,
        # xautolimitmargin=(0.01,0.01),
        # yautolimitmargin=(0.01,0.01),
    )

    ax1, xl1, yl1 = gaiastarplot!(fig[1,1], model, chain, indices[1]; axis)
    ax2, xl2, yl2 = gaiastarplot!(fig[1,2], model, chain, indices[2]; axis)
    ax3, xl3, yl3 = gaiastarplot!(fig[1,3], model, chain, indices[3]; axis)
    ax4, xl4, yl4 = gaiastarplot!(fig[2,1], model, chain, indices[4]; axis)
    ax5, xl5, yl5 = gaiastarplot!(fig[2,2], model, chain, indices[5]; axis)
    ax6, xl6, yl6 = gaiastarplot!(fig[2,3], model, chain, indices[6]; axis)
    ax7, xl7, yl7 = gaiastarplot!(fig[3,1], model, chain, indices[7]; axis)
    ax8, xl8, yl8 = gaiastarplot!(fig[3,2], model, chain, indices[8]; axis)
    ax9, xl9, yl9 = gaiastarplot!(fig[3,3], model, chain, indices[9]; axis)
    axes = [
        ax1
        ax2
        ax3
        ax4
        ax5
        ax6
        ax7
        ax8
        ax9
    ]
    hidexdecorations!.((ax1,ax2,ax3,ax4,ax5,ax6,))
    hideydecorations!.((ax2,ax3,ax5,ax6,ax8,ax9))
    for ax in (ax1,ax2,ax3,ax4,ax5,ax6,)
        ax.bottomspinevisible=false
    end
    for ax in (ax2,ax3,ax5,ax6,ax8,ax9)
        ax.leftspinevisible=false
    end
    linkaxes!(axes)

    xl = extrema([xl1...,xl2...,xl3...,xl4...,xl5...,xl6...,xl7...,xl8...,xl9...]) .* 1.4
    yl = extrema([yl1...,yl2...,yl3...,yl4...,yl5...,yl6...,yl7...,yl8...,yl9...]) .* 1.4
    xlims!.(axes,xl[2], xl[1])
    ylims!.(axes,xl[1], xl[2])

    Colorbar(fig[3,4],label="time",ticklabelsvisible=false,colormap=:cyclic_mrybm_35_75_c68_n256)
    colsize!.((fig,), 1:3, (Auto(1.0),))
    colgap!.((fig,), 1:2, 0)
    rowgap!.((fig,), 1:2, 0)
    # colsize!.((fig.layout,), 4, (Auto(0.1),))
    Label(f[0,1],text="Companion", tellwidth=false)
    Label(f[0,2],text="Primary", tellwidth=false)

    f
end
save("daniel_target_2.png",out)
out
##
using PairPlots
els = Octofitter.construct_elements(model,chain,:b,:);
fig = pairplot((;
    per=log10.(period.(els)),
    e=eccentricity.(els),
    i=rad2deg.(inclination.(els)),
    pmra=chain[:GaiaDR4_pmra][:],
    pmdec=chain[:GaiaDR4_pmdec][:],
    plx=chain[:GaiaDR4_plx][:],
))
save("cornerplot.png",fig)
##
Octofitter.dotplot(model,chain,mode=:period)


##
fig = Figure(size=(1024,920))
indices = rand(1:size(chain,1), 9)

axis = (;
    topspinecolor="#AAA",
    bottomspinecolor="#AAA",
    leftspinecolor="#AAA",
    rightspinecolor="#AAA",
    xtickcolor="#AAA",
    ytickcolor="#AAA",
    xticklabelcolor="#666",
    yticklabelcolor="#666",
    # xlabelcolor="#AAA",
    # ylabelcolor="#AAA",
    # rightspinevisible=false,
    # topspinevisible=false,
    # xtrimspine=true,
    # ytrimspine=true,
    # xticks=1:-1:-1,
    # yticks=-1:1,
    titlesize=11,
    # xautolimitmargin=(0.01,0.01),
    # yautolimitmargin=(0.01,0.01),
)

ax, xl1, yl1 = gaiastarplot!(fig[1,1], model, chain, indices[1]; axis)
gaiastarplot!(ax, model, chain, indices[2]; axis)
gaiastarplot!(ax, model, chain, indices[3]; axis)
gaiastarplot!(ax, model, chain, indices[4]; axis)
gaiastarplot!(ax, model, chain, indices[5]; axis)
gaiastarplot!(ax, model, chain, indices[6]; axis)
gaiastarplot!(ax, model, chain, indices[7]; axis)
gaiastarplot!(ax, model, chain, indices[8]; axis)


fig
##
##################################################
# Orbit axis vs scan angle
# For each posterior draw we take the orbit-plane normal vector (defined by
# inclination i and longitude of ascending node Ω) and project it onto the
# sky. We overlay Gaia's scan-angle directions so alignment between the
# orbit axis and the scan pattern — a warning sign for scan-harmonic false
# positives — shows up visually.

function orbit_axis_vs_scan_plot(
    model::Octofitter.LogDensityModel,
    results::Chains;
    planet_name=first(keys(model.system.planets)),
    fname="$(model.system.name)-orbit-axis-vs-scan.png",
)
    gaia_likes = filter(o -> o isa Octofitter.GaiaDR4AstromObs, model.system.observations)
    isempty(gaia_likes) && error("No GaiaDR4AstromObs found in model")
    likeobj = first(gaia_likes)
    scan_angles = collect(likeobj.table.scan_pos_angle) # radians, PA east of north

    i_vals = vec(results["$(planet_name)_i"][:])
    Ω_vals = vec(results["$(planet_name)_Ω"][:])

    # Orbit-plane normal projected onto the sky.
    # Ω is the position angle of the ascending node, so the line of nodes lies
    # along PA = Ω. The orbit axis projects onto the sky perpendicular to this,
    # with a length of sin(i) from a unit angular-momentum vector.
    ax_east  =  sin.(i_vals) .* cos.(Ω_vals)
    ax_north = -sin.(i_vals) .* sin.(Ω_vals)

    # PA of orbit axis, mod π (the ± sign of the normal is physically irrelevant
    # for checking alignment with scan directions).
    axis_pa = mod.(atan.(ax_east, ax_north), π)
    scan_pa = mod.(scan_angles, π)

    fig = Figure(size=(1100, 500))

    ax1 = Axis(fig[1,1],
        xlabel="Position angle east of north [deg]",
        ylabel="Posterior density",
        title="Orbit axis PA vs Gaia scan angles",
        xticks=0:30:180,
    )
    hist!(ax1, rad2deg.(axis_pa); bins=72, normalization=:pdf,
          color=(Makie.wong_colors()[1], 0.7), strokewidth=0,
          label="Orbit axis PA")
    vlines!(ax1, rad2deg.(scan_pa); color=(:red, 0.5), linewidth=1,
            label="Gaia scans")
    xlims!(ax1, 0, 180)
    axislegend(ax1, position=:rt)

    ax2 = Axis(fig[1,2],
        xlabel="Axis east component",
        ylabel="Axis north component",
        title="Orbit-axis direction on sky",
        aspect=DataAspect(),
        xreversed=true,
    )
    scatter!(ax2, ax_east, ax_north;
        color=(Makie.wong_colors()[1], 0.3),
        markersize=3, strokewidth=0,
    )
    # Unit circle: edge-on orbits (i = 90°) land here.
    θθ = range(0, 2π, length=200)
    lines!(ax2, sin.(θθ), cos.(θθ); color=:grey, linestyle=:dash)
    # Scan-angle lines through the origin for visual comparison.
    for s in scan_angles
        lines!(ax2, [sin(s), -sin(s)], [cos(s), -cos(s)];
               color=(:red, 0.35), linewidth=1)
    end
    limits!(ax2, -1.05, 1.05, -1.05, 1.05)

    Makie.save(fname, fig, px_per_unit=3)
    return fig
end

orbit_axis_vs_scan_plot(model, chain)
##
