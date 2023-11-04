function toeplitz(c, r)
    @assert c[1] == r[1]
    nc = length(c)
    nr = length(r)
    A = zeros(eltype(c), nc, nr)
    A[:, 1] = c
    A[1, :] = r
    for i = 2:nr
        A[2:end, i] = A[1:end-1, i-1]
    end
    A
end


@inline obslength(y::AbstractMatrix) = size(y, 1)
@inline obslength(y::AbstractVector) = length(y[1])

# @inline Base.oftype(x::T, y::T) where T = y
@inline Base.oftype(x::AbstractMatrix, y::Vector{<:Vector}) = reduce(hcat, y)
@inline Base.oftype(x::AbstractMatrix, y::Vector{<:Number}) = reshape(y, 1, :)
@inline Base.oftype(x::Vector{<:Vector}, y::Matrix) = [y[:, i] for i = 1:size(y, 2)]
@inline Base.oftype(x::Vector{<:Vector}, y::Vector{<:Number}) =
    [[y[i]] for i in eachindex(y)]
@inline function Base.oftype(x::Vector{<:Number}, y::Matrix)
    size(y, 1) == 1 || size(y, 2) == 1 && return vec(y)
    throw(ArgumentError("Cannot convert a matrix with both dimensions greater than 1 to a vector."))
end

@inline function Base.oftype(x::Vector{<:Number}, y::Vector{<:Vector})
    size(y, 1) == 1 ||
        size(y, 2) == 1 ||
        throw(ArgumentError("Cannot convert a matrix with both dimensions greater than 1 to a vector."))
    reduce(vcat, y)
end

@inline Base.oftype(::Type{Matrix}, y::Vector{<:Number}) = reshape(y, 1, :)
@inline Base.oftype(::Type{Matrix}, y::Vector{<:Vector}) = reduce(hcat, y)
@inline Base.oftype(::Type{Matrix}, y::Matrix) = y
@inline Base.oftype(::Type{Matrix}, y::AbstractMatrix) = Matrix(y)


@inline time1(y::Vector) = y
@inline time1(y::Vector{<:Vector}) = transpose(reduce(hcat, y))
@inline function time1(y::Matrix)
    Matrix(transpose(y))
end
@inline function time1(y::AbstractMatrix)
    Matrix(transpose(y))
end
@inline function time1(y::Union{Adjoint{T, Vector{T}}, Transpose{T, Vector{T}}}) where T
    transpose(y)
end

@inline time2(y::Vector) = oftype(Matrix, y)
@inline time2(y::Vector{<:Vector}) = oftype(Matrix, y)
@inline time2(y::Matrix) = oftype(Matrix, y)
@inline time2(y::AbstractMatrix) = oftype(Matrix, y)


function Base.oftype(x::Number, y::AbstractArray)
    length(y) == 1 || throw(ArgumentError("Can only autoconvert one-element arrays"))
    y[]
end

"""
    rms(x)
  
Root mean square of `x`.
"""
rms(x::AbstractVector) = sqrt(mean(abs2, x))

"""
    sse(x)

Sum of squares of `x`.
"""
sse(x::AbstractVector) = x ⋅ x

"""
    mse(x)

Mean square of `x`.
"""
mse(x::AbstractVector) = sse(x) / length(x)
mse(x::AbstractMatrix) = sum(abs2, x, dims=2) ./ size(x, 2)

rms(x::AbstractMatrix) = sqrt.(mean(abs2.(x), dims = 2))[:]
sse(x::AbstractMatrix) = sum(abs2, x, dims = 2)[:]

"""
    fpe(e, d::Int)

Akaike's Final Prediction Error (FPE) criterion for model order selection.

`e` is the prediction errors and `d` is the number of parameters estimated.
"""
function fpe(e::AbstractMatrix, d::Int)
    size(e, 2) > size(e, 1) || throw(ArgumentError("e must have more columns than rows"))
    N = size(e, 2)
    d < N || error("fpe ill-defined when d > N")
    det(sum(e[:, i]*e[:, i]' for i = axes(e, 2))) * (1 + d/N)/(1-d/N)
end
function fpe(e::AbstractVector, d::Int)
    N = length(e)
    d < N || error("fpe ill-defined when d > N")
    sum(abs2, e)/N * (1 + d/N)/(1-d/N)
end


"""
    modelfit(y, yh)

Compute the model fit of `yh` to `y` as a percentage, sometimes referred to as the normalized root mean square error (NRMSE).

```math
\\text{modelfit}(y, \\hat{y}) = 100 \\left(1 - \\frac{\\sqrt{\\text{SSE}(y - \\hat{y})}}{\\sqrt{\\text{SSE}(y - \\bar{y})}}\\right)
```

An output of 100 indicates a perfect fit, an output of 0 indicates that the fit is no better than the mean if the data. Negative values are possible if the prediction is worse than predicting the mean of the data.

See also [`rms`](@ref), [`sse`](@ref), [`mse`](@ref), [`fpe`](@ref), [`aic`](@ref).
"""
modelfit(y, yh) = 100 * (1 .- rms(y .- yh) ./ rms(y .- mean(y, dims = 2)))
modelfit(y::T, yh::T) where {T<:AbstractVector} =
    100 * (1 - rms(y .- yh) / rms(y .- mean(y)))

"""
    aic(e::AbstractVector, d)

Akaike's Information Criterion (AIC) for model order selection.

`e` is the prediction errors and `d` is the number of parameters estimated.

See also [`fpe`](@ref).
"""
aic(x::AbstractVector, d) = log(sse(x)) .+ 2d / size(x, 2)

"See [`modelfit`](@ref)"
const nrmse = modelfit
