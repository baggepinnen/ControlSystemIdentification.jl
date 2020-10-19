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
@inline Base.oftype(x::Matrix, y::Vector{<:Vector}) = reduce(hcat, y)
@inline Base.oftype(x::Matrix, y::Vector{<:Number}) = reshape(y, 1, :)
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
@inline time1(y::Vector{<:Vector}) = reduce(hcat, y)'
@inline function time1(y::Matrix)
    if size(y, 1) == 1
        return vec(y)
    end
    Matrix(y')
end
@inline function time1(y::AbstractMatrix)
    if size(y, 1) == 1
        return vec(y)
    end
    Matrix(y')
end

@inline time2(y::Vector) = oftype(Matrix, y)
@inline time2(y::Vector{<:Vector}) = oftype(Matrix, y)
@inline time2(y::Matrix) = oftype(Matrix, y)
@inline time2(y::AbstractMatrix) = oftype(Matrix, y)


function Base.oftype(x::Number, y::AbstractArray)
    length(y) == 1 || throw(ArgumentError("Can only autoconvert one-element arrays"))
    y[]
end

rms(x::AbstractVector) = sqrt(mean(abs2, x))
sse(x::AbstractVector) = x ⋅ x
mse(x::AbstractVector) = sse(x) / length(x)


rms(x::AbstractMatrix) = sqrt.(mean(abs2.(x), dims = 2))[:]
sse(x::AbstractMatrix) = sum(abs2, x, dims = 2)[:]
modelfit(y, yh) = 100 * (1 .- rms(y .- yh) ./ rms(y .- mean(y, dims = 2)))
modelfit(y::T, yh::T) where {T<:AbstractVector} =
    100 * (1 .- rms(y .- yh) ./ rms(y .- mean(y)))
aic(x::AbstractVector, d) = log(sse(x)) .+ 2d / size(x, 2)
const nrmse = modelfit
