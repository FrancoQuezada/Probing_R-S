module GridHeatmap

# Force GR to render off-screen before Plots initializes its backend.
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using CSV
using DataFrames
using Plots
using Printf

export parse_cli,
       detect_z_columns,
       build_grid_matrix,
       generate_heatmap,
       main,
       usage

"""Return command-line usage for the two-dimensional grid visualizer."""
function usage()
    return """
Usage:
  julia --project=assembly_line_model assembly_line_model/examples/plot_grid_heatmap.jl \\
    --input=PATH --metric=COLUMN --output=PATH

Required arguments:
  --input PATH       candidate_results.csv produced by the grid experiment
  --metric COLUMN    numeric CSV column to use as heatmap color (for example: objective)
  --output PATH      destination PNG path
"""
end

"""Parse the three required CLI options, accepting `--key=value` and `--key value`."""
function parse_cli(args::AbstractVector{<:AbstractString}=ARGS)
    values = Dict{Symbol,String}()
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            length(args) == 1 || throw(ArgumentError("--help cannot be combined with other arguments"))
            return (; help=true, values)
        end

        key, value = if startswith(argument, "--") && occursin('=', argument)
            option, supplied = split(argument[3:end], '='; limit=2)
            Symbol(option), supplied
        elseif startswith(argument, "--")
            index < length(args) || throw(ArgumentError("missing value for $argument"))
            index += 1
            Symbol(argument[3:end]), args[index]
        else
            throw(ArgumentError("unexpected argument: $argument"))
        end

        key in (:input, :metric, :output) || throw(ArgumentError("unknown option: --$key"))
        haskey(values, key) && throw(ArgumentError("duplicate option: --$key"))
        isempty(value) && throw(ArgumentError("--$key must be nonempty"))
        values[key] = value
        index += 1
    end

    missing = ["--$key" for key in (:input, :metric, :output) if !haskey(values, key)]
    isempty(missing) || throw(ArgumentError("missing required option(s): $(join(missing, ", "))"))
    return (; help=false, values)
end

"""Read a candidate-results CSV after checking that it exists and is nonempty."""
function read_candidates(input_path::AbstractString)::DataFrame
    isfile(input_path) || throw(ArgumentError("input CSV does not exist: $input_path"))
    frame = CSV.read(input_path, DataFrame)
    nrow(frame) > 0 || throw(ArgumentError("input CSV has no candidate rows: $input_path"))
    return frame
end

"""Find the `z_1` and `z_2` columns and reject non-two-dimensional result files."""
function detect_z_columns(frame::DataFrame)::NTuple{2,Symbol}
    indexed_columns = Tuple{Int,Symbol}[]
    for name in names(frame)
        matched = match(r"^z_(\d+)$", String(name))
        isnothing(matched) || push!(indexed_columns, (parse(Int, matched.captures[1]), Symbol(name)))
    end
    sort!(indexed_columns; by=first)
    length(indexed_columns) == 2 || throw(ArgumentError(
        "expected exactly two decision columns named z_1 and z_2; found $(length(indexed_columns))"))
    first.(indexed_columns) == [1, 2] || throw(ArgumentError(
        "decision columns must be named z_1 and z_2; found $(join(string.(first.(indexed_columns)), ", "))"))
    return (indexed_columns[1][2], indexed_columns[2][2])
end

function _finite_numeric_column(frame::DataFrame, column::Symbol, description::AbstractString)
    values = Float64[]
    for (row, value) in enumerate(frame[!, column])
        ismissing(value) && throw(ArgumentError("$description has a missing value in row $row"))
        value isa Real || throw(ArgumentError("$description must be numeric; row $row contains $(repr(value))"))
        number = Float64(value)
        isfinite(number) || throw(ArgumentError("$description must be finite; row $row contains $number"))
        push!(values, number)
    end
    return values
end

"""Build an ordered z₂-by-z₁ matrix, using NaN for cells absent from an incomplete grid."""
function build_grid_matrix(frame::DataFrame, z_columns::NTuple{2,Symbol}, metric::AbstractString)
    metric_column = Symbol(metric)
    metric in names(frame) || throw(ArgumentError("metric column does not exist in CSV: $metric"))

    z1 = _finite_numeric_column(frame, z_columns[1], String(z_columns[1]))
    z2 = _finite_numeric_column(frame, z_columns[2], String(z_columns[2]))
    metric_values = _finite_numeric_column(frame, metric_column, "metric $metric")
    z1_values, z2_values = sort!(unique(z1)), sort!(unique(z2))
    z1_index = Dict(value => index for (index, value) in enumerate(z1_values))
    z2_index = Dict(value => index for (index, value) in enumerate(z2_values))
    matrix = fill(NaN, length(z2_values), length(z1_values))
    seen = Set{Tuple{Float64,Float64}}()

    for row in eachindex(metric_values)
        coordinate = (z1[row], z2[row])
        coordinate in seen && throw(ArgumentError(
            "duplicate candidate combination: z_1=$(coordinate[1]), z_2=$(coordinate[2])"))
        push!(seen, coordinate)
        matrix[z2_index[z2[row]], z1_index[z1[row]]] = metric_values[row]
    end

    expected_cells = length(z1_values) * length(z2_values)
    if length(seen) != expected_cells
        @warn "candidate grid is incomplete; missing combinations are shown as NaN" candidates=length(seen) expected_cells missing_cells=expected_cells - length(seen)
    end
    best_value, best_row = findmax(metric_values)
    return (; z1_values, z2_values, matrix, candidate_count=length(seen), expected_cells,
            best_z=(z1[best_row], z2[best_row]), best_value)
end

"""Render and save a manuscript-resolution heatmap with the best candidate marked."""
function generate_heatmap(grid, metric::AbstractString, output_path::AbstractString)
    output_absolute = abspath(output_path)
    mkpath(dirname(output_absolute))
    metric_label = replace(metric, '_' => ' ')
    best_label = @sprintf("Maximum: z = (%.4g, %.4g), %s = %.8g",
                          grid.best_z[1], grid.best_z[2], metric, grid.best_value)
    figure = heatmap(grid.z1_values, grid.z2_values, grid.matrix;
        xlabel="Probing decision z₁",
        ylabel="Probing decision z₂",
        colorbar_title=metric_label,
        title="Exhaustive probing-grid search — $(metric_label)",
        c=:viridis,
        size=(1800, 1400),
        dpi=300,
        legend=:topright)
    scatter!(figure, [grid.best_z[1]], [grid.best_z[2]];
        marker=:star5,
        markersize=11,
        markercolor=:red,
        markerstrokecolor=:white,
        markerstrokewidth=1.5,
        label=best_label)
    savefig(figure, output_absolute)
    return output_absolute
end

function main(args::AbstractVector{<:AbstractString}=ARGS)
    options = parse_cli(args)
    if options.help
        print(usage())
        return nothing
    end
    frame = read_candidates(options.values[:input])
    z_columns = detect_z_columns(frame)
    grid = build_grid_matrix(frame, z_columns, options.values[:metric])
    output_absolute = generate_heatmap(grid, options.values[:metric], options.values[:output])

    println("Detected grid dimensions: z₁=$(length(grid.z1_values)), z₂=$(length(grid.z2_values))")
    println("Candidates: $(grid.candidate_count) of $(grid.expected_cells) grid cells")
    println("Metric: $(options.values[:metric])")
    @printf("Best z: (%.17g, %.17g)\n", grid.best_z[1], grid.best_z[2])
    @printf("Best value: %.17g\n", grid.best_value)
    println("Generated heatmap: $output_absolute")
    return (; frame, z_columns, grid, output_path=output_absolute)
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error
        error isa ArgumentError || rethrow()
        println(stderr, "Error: $(sprint(showerror, error))")
        print(stderr, usage())
        exit(2)
    end
end

end # module GridHeatmap
