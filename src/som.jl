using ProgressMeter

function train_som!(
        som::Array{Float64, 3}, data::Matrix{Float64}, adj_ids::Vector{Vector{Int}};
        weight_reduction::Float64=0.5, learning_rate::Float64=1e-2, nn_weight::Float64=0.5, n_epochs::Int=5,
        weight_reduction_decay::Float64=0.9, learning_rate_decay::Float64=0.8
    )
    clust_ids = zeros(Int, size(data, 1))
    grid_ids = permutedims(cat([cat([[i,j] for i in 1:size(som,1)]..., dims=2)' for j in 1:size(som, 2)]..., dims=3), (1, 3, 2));

    nn_count_cache = zeros(Int, size(som,1)*size(som,2))

    @showprogress for epoch in 1:n_epochs
        weight_masks = [
            [learning_rate .* weight_reduction .^ sum(abs.(grid_ids .- grid_ids[i:i,j:j,:]), dims=3) for i in axes(grid_ids, 1)]
            for j in axes(grid_ids, 2)
        ];

        vec_diff = deepcopy(som)
        vec_diff_sq = deepcopy(som)
        dists = zeros(Float64, size(som,1), size(som,2), 1)
        for ni in 1:size(data,1)
            @views vec_diff .= reshape(data[ni,:], 1, 1, size(data,2))
            vec_diff .-= som
            vec_diff_sq .= vec_diff.^2
            for i in 1:size(som,1)
                for j in 1:size(som,2)
                    @views dists[i,j] = sum(vec_diff_sq[i,j,:])
                end
            end

            if epoch > 1 && nn_weight > 0
                count_array!(nn_count_cache, clust_ids[adj_ids[ni]])
                nn_count_cache .+= 1
                dists .= (1 ./ reshape(nn_count_cache, size(dists))' .* nn_weight) + (dists .* (1 - nn_weight))
            end
            ti,tj = Tuple(findmin(dists)[2]);
            clust_ids[ni] = (ti - 1) * size(som, 1) + tj
            @views vec_diff .*= weight_masks[ti][tj]
            som .+= vec_diff
        end

        learning_rate *= learning_rate_decay
        weight_reduction *= weight_reduction_decay
    end

    return clust_ids
end