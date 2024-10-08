using ProgressMeter
using Base.Threads
using Statistics

function expect_kmeans!(
        clust_ids::AbstractVector{Int}, clust_ids_prev::AbstractVector{Int}, data::AbstractMatrix{Float64}, centroids::Matrix{Float64},
        nn_counts::Vector{Vector{Int}}, weights::Matrix{Float64},
        adj_ids::Union{Vector{Vector{Int}}, Nothing}=nothing;
        nn_weight::Float64=0.5, is_random::Bool=false
    )

    tci = 1:size(centroids, 1)
    @threads for ni in 1:size(data,1)
        ti = threadid()
        for ci in 1:size(centroids, 1)
            # cd = 0
            # for di in 1:size(data,2)
            #     cd += (data[ni,di] - centroids[ci,di])^2
            # end
            # weights[ni, ci] = 1 / cd
            @views weights[ni, ci] = (cor(data[ni,:], centroids[ci,:]) + 1) / 2
        end

        # if nn_weight > 0 && adj_ids !== nothing
        #     @views count_array!(nn_counts[ti], clust_ids_prev[adj_ids[ni]])
        #     for di in axes(weights, 2)
        #         weights[ni, di] = ((nn_counts[ti][di] + 1) * nn_weight) + (weights[ni, di] * (1 - nn_weight))
        #         # weights[ni, di] ^= 10
        #     end
        # end

        @views clust_ids[ni] = is_random ? fsample(weights[ni,:]) : argmax(weights[ni,:])
        # @views clust_ids[ni] = fsample(weights[ni,:])
        # clust_ids[ni] = rand(tci)
    end
end

function maximize_kmeans!(
        centroids::Matrix{Float64}, data::Matrix{Float64}, clust_ids::Vector{Int},
        counts::Vector{Int}
    )
    centroids .= 0
    count_array!(counts, clust_ids)
    for ni in 1:size(data,1)
        @views centroids[clust_ids[ni],:] .+= data[ni,:]
    end
    for ci in 1:size(centroids, 1)
        if counts[ci] > 0
            centroids[ci,:] ./= counts[ci]
        else
            centroids[ci,:] .= data[rand(1:size(data, 1)),:]
        end
    end
end

function maximize_kmeans!(
        centroids::Matrix{Float64}, data::Matrix{Float64}, clust_ids::Vector{Int}, weights::Vector{Float64},
        cum_weights::Vector{Float64}
    )
    centroids .= 0
    # count_array!(cum_weights, clust_ids, weights)
    # for ni in 1:size(data,1)
    #     for di in 1:size(data,2)
    #         centroids[clust_ids[ni],di] += data[ni,di] * weights[ni]
    #     end
    # end
    # for ci in 1:size(centroids, 1)
    #     if cum_weights[ci] > 0
    #         centroids[ci,:] ./= cum_weights[ci]
    #     else
    #         centroids[ci,:] .= data[rand(1:size(data, 1)),:]
    #     end
    # end

    ids_per_clust = split_ids(clust_ids)
    for (ci,mids) in enumerate(ids_per_clust)
        centroids[clust_ids[mids[1]],:] .= median(data[mids,:], dims=1)[:]
    end
end

function train_kmeans!(
        centroids::Matrix{Float64}, data::Matrix{Float64},
        adj_ids::Union{Vector{Vector{Int}}, Nothing}=nothing;
        point_weights::Union{Vector{Float64}, Nothing}=nothing,
        nn_weight::Float64=0.5, n_iters::Int=50, frac_random_iters::Float64=0.75,
        min_change_frac::Float64=1e-5, verbose::Bool=true
    )
    if any(isnan.(data))
        error("NaNs in data")
    end

    n_random_iters = round(Int, n_iters * frac_random_iters)

    clust_ids = zeros(Int, size(data, 1))
    clust_ids_prev = zeros(Int, size(data, 1))
    weights = zeros(Float64, (size(data, 1), size(centroids, 1)))
    counts = isnothing(point_weights) ? zeros(Int, size(centroids, 1)) : zeros(Float64, size(centroids, 1))
    nn_counts = [zeros(Int, size(centroids, 1)) for _ in 1:nthreads()]
    change_fracs = Float64[]

    progress = Progress(n_iters, 0.3)
    for iter in 1:n_iters
        expect_kmeans!(
            clust_ids, clust_ids_prev, data, centroids, nn_counts, weights, adj_ids;
            nn_weight=(iter > 1 ? nn_weight : 0.0), is_random=(iter > 0) & (iter <= n_random_iters)
        )

        change_frac = mean(clust_ids .!= clust_ids_prev)
        push!(change_fracs, change_frac)

        prog_vals = [
            ("Iteration", iter),
            ("Fraction of probs changed", round(change_frac, sigdigits=3))
        ]
        if verbose
            next!(progress, showvalues=prog_vals)
        end

        if change_frac < min_change_frac
            if verbose
                finish!(progress, showvalues=prog_vals)
            end
            break
        end
        clust_ids_prev .= clust_ids

        if isnothing(point_weights)
            maximize_kmeans!(centroids, data, clust_ids, counts)
        else
            maximize_kmeans!(centroids, data, clust_ids, point_weights, counts)
        end
    end

    # weights ./= sum(weights, dims=2)
    return (;assignment=clust_ids, centroids, weights, change_fracs)
end