import Distributions: Normal, Uniform, Truncated

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

rms(x::AbstractVector) = sqrt(mean(abs2, x))
sse(x::AbstractVector) = x ⋅ x
mse(x::AbstractVector) = sse(x) / length(x)
mse(x::AbstractMatrix) = sum(abs2, x, dims=2) ./ size(x, 2)

rms(x::AbstractMatrix) = sqrt.(mean(abs2.(x), dims = 2))[:]
sse(x::AbstractMatrix) = sum(abs2, x, dims = 2)[:]
modelfit(y, yh) = 100 * (1 .- rms(y .- yh) ./ rms(y .- mean(y, dims = 2)))
modelfit(y::T, yh::T) where {T<:AbstractVector} =
    100 * (1 .- rms(y .- yh) ./ rms(y .- mean(y)))
aic(x::AbstractVector, d) = log(sse(x)) .+ 2d / size(x, 2)
const nrmse = modelfit

"""
function which generates a sequence of inputs GBN
# Arguments:
- `N`: sequence length (total number of samples)
- `p_swd`: desired probability of switching (no switch: 0<x<1 :always switch)
- `Nmin`: minimum number of samples between two switches
- `Range`: input range
- `Tol`: tolerance on switching probability relative error
- `nit_max`: maximum number of iterations
"""
function gbnseq(N, p_swd; Nmin=1, Range=[-1.0, 1.0], Tol=0.01, nit_max=30)
    min_Range = minimum(Range)
    max_Range = maximum(Range)
    prob = rand(Uniform(0, 1))
    local gbn_b, Nswb
    if prob < 0.5
        gbn = -1.0 * ones(N)
    else
        gbn = 1.0 * ones(N)
    end
    # init. variables
    p_sw = p_sw_b = 2.0 # actual switch probability
    nit = 0
    while abs(p_sw - p_swd) / p_swd > Tol && nit <= nit_max
        i_fl = 0
        Nsw = 0
        for i in 1:1:N-1
            gbn[i+1] = gbn[i]
            if (i - i_fl >= Nmin)
                prob = rand(Uniform(0, 1))
                # track last test of p_sw
                i_fl = i
                if prob < p_swd
                    # switch and then count it
                    gbn[i+1] = -gbn[i+1]
                    Nsw = Nsw + 1
                end
            end
        end
        # check actual switch probability
        p_sw = Nmin * (Nsw + 1) / N #println("p_sw", p_sw);
        # set best iteration
        if abs(p_sw - p_swd) < abs(p_sw_b - p_swd)
            p_sw_b = p_sw
            Nswb = Nsw
            gbn_b = copy(gbn)
        end
        # increase iteration number
        nit = nit + 1 #print("nit", nit)
    end
    # rescale GBN
    for i in 1:1:N
        if gbn_b[i] > 0.0
            gbn_b[i] = max_Range
        else
            gbn_b[i] = min_Range
        end
    end
    return (gbn_b=gbn_b, p_sw_b=p_sw_b, Nswb=Nswb)
end

"""
function which generates a sequence of inputs as Random walk
# Arguments:
- `N`: sequence length (total number of samples);
- `rw0`: initial value 
- `sigma`: standard deviation (mobility) of randow walk
- `lb`: lower bound of the randow walk
- `ub`: upper bound of the randow walk
"""
function rwseq(N, rw0;sigma=0.1, lb=-1, ub=1)
    rw = rw0*ones(N)
    for i in 1:1:N-1
        # return random sample from a normal (Gaussian) distribution with:
        # mean = 0.0, standard deviation = sigma, and length = 1
        delta = rand(Truncated(Normal(0, sigma), lb, ub))
        # refresh input
        rw[i+1] = rw[i] + delta
    end
    return (rw=rw,)
end
