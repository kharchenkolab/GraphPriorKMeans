using ProgressMeter
using Base.Threads

function expect_kmeans!(
        clust_ids::AbstractVector{Int}, clust_ids_prev::AbstractVector{Int}, data::AbstractMatrix{Float64}, centroids::Matrix{Float64},
        nn_counts::Vector{Vector{Int}}, weights::Vector{Vector{Float64}},
        adj_ids::Union{Vector{Vector{Int}}, Nothing}=nothing;
        nn_weight::Float64=0.5, is_random::Bool=false
    )
    @threads for ni in 1:size(data,1)
        ti = threadid()
        for ci in 1:size(centroids, 1)
            cd = 0
            for di in 1:size(data,2)
                cd += (data[ni,di] - centroids[ci,di])^2
            end
            weights[ti][ci] = 1 / cd
        end

        if nn_weight > 0 && adj_ids !== nothing
            @views count_array!(nn_counts[ti], clust_ids_prev[adj_ids[ni]])
            for di in eachindex(weights[ti])
                weights[ti][di] = ((nn_counts[ti][di] + 1) * nn_weight) + (weights[ti][di] * (1 - nn_weight))
            end
        end

        clust_ids[ni] = is_random ? fsample(weights[ti]) : argmax(weights[ti])
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

function train_kmeans!(
        centroids::Matrix{Float64}, data::Matrix{Float64},
        adj_ids::Union{Vector{Vector{Int}}, Nothing}=nothing;
        nn_weight::Float64=0.5, n_iters::Int=50, frac_random_iters::Float64=0.0
    )
    if any(isnan.(data))
        error("NaNs in data")
    end

    n_random_iters = Int(n_iters * frac_random_iters)

    clust_ids = zeros(Int, size(data, 1))
    clust_ids_prev = zeros(Int, size(data, 1))
    weights = [zeros(Float64, size(centroids, 1)) for _ in 1:nthreads()]
    counts = zeros(Int, size(centroids, 1))
    nn_counts = [zeros(Int, size(centroids, 1)) for _ in 1:nthreads()]

    @showprogress for iter in 1:n_iters
        expect_kmeans!(
            clust_ids, clust_ids_prev, data, centroids, nn_counts, weights, adj_ids;
            nn_weight=(iter > 1 ? nn_weight : 0.0), is_random=(iter <= n_random_iters)
        )

        all(clust_ids .== clust_ids_prev) && break
        clust_ids_prev .= clust_ids

        maximize_kmeans!(centroids, data, clust_ids, counts)
    end
    return clust_ids
end