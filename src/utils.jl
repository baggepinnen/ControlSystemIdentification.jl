function toeplitz(c,r)
    @assert c[1] == r[1]
    nc = length(c)
    nr = length(r)
    A  = zeros(nc, nr)
    A[:,1] = c
    A[1,:] = r
    for i in 2:nr
        A[2:end,i] = A[1:end-1,i-1]
    end
    A
end

obslength(y::AbstractMatrix) = size(y,1)
obslength(y::AbstractVector) = length(y[1])

Base.oftype(x::Vector{<:Vector}, y::Vector{<:Vector}) = y
Base.oftype(x::Matrix, y::Vector{<:Vector}) = reduce(hcat,y)
Base.oftype(x::Matrix, y::Matrix) = y
Base.oftype(x::Vector{<:Vector}, y::Matrix) = [y[:,i] for i in 1:size(y,2)]

rms(x::AbstractVector) = sqrt(mean(abs2,x))
sse(x::AbstractVector) = x⋅x
mse(x::AbstractVector) = sse(x)/length(x)


rms(x::AbstractMatrix) = sqrt.(mean(abs2.(x),dims=2))[:]
sse(x::AbstractMatrix) = sum(abs2,x,dims=2)[:]
modelfit(y,yh) = 100 * (1 .-rms(y.-yh)./rms(y.-mean(y,dims=2)))
modelfit(y::T,yh::T) where T <: AbstractVector = 100 * (1 .-rms(y.-yh)./rms(y.-mean(y)))
aic(x::AbstractVector,d) = log(sse(x)) .+ 2d/size(x,2)
const nrmse = modelfit
