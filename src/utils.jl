import Random

function one_hot_vector(n::Int, i::Int)
    vec = zeros(n)
    if i > 0
        vec[i] = 1
    end
    return vec
end

split_ids(factor::Vector{Int}; kwargs...) = split(1:length(factor), factor; kwargs...)

function split(
        array::AbstractVector{TV}, factor::AbstractVector{<:Union{<:Integer, Missing}};
        max_factor::Union{Int, Nothing}=nothing, drop_zero::Bool=false
    )::Array{Vector{TV}, 1} where TV
    @assert length(array) == length(factor)
    if max_factor === nothing
        max_factor = maximum(skipmissing(factor))
    end

    counts = count_array(factor; max_value=max_factor, drop_zero=drop_zero)
    splitted = [Vector{TV}(undef, c) for c in counts]
    last_id = zeros(Int, max_factor)

    for i in eachindex(array)
        fac = factor[i]
        (ismissing(fac) || (drop_zero && fac == 0)) && continue

        li = (last_id[fac] += 1)
        splitted[fac][li] = array[i]
    end

    return splitted
end

count_array(values::VT where VT<: AbstractVector{<:Integer}; max_value::Union{<:Integer, Nothing}=nothing) =
    count_array!(
        zeros(Int, max_value !== nothing ? max_value : (isempty(values) ? 0 : maximum(values))),
        values;
        erase_counts=false
    )

function count_array!(
        counts::VT1 where VT1 <: AbstractVector{<:Integer}, values::VT2 where VT2 <: AbstractVector{<:Integer};
        erase_counts::Bool=true
    )
    if erase_counts
        counts .= 0
    end

    for v in values
        counts[v] += 1
    end

    return counts
end

@inline function fsample(w::AbstractVector{Float64})::Int
    n = length(w)
    if n == 0
        error("Empty vector for sampling")
    end

    t = rand(Random.GLOBAL_RNG) * sum(w)
    i = 1
    cw = w[1]
    while cw < t && i < length(w)
        i += 1
        @inbounds cw += w[i]
    end
    11
    return i
end