using ControlSystemsBase, ControlSystemIdentification, SymbolicRegression
using LossFunctions
using Setfield
using Symbolics

struct Simloss{S} <: Function
    simloss::S
end

struct Workspace2{Y,A}
    yh::Y
    a::A
end

varMap(na, nb) =  ["y(k-1)"; ["yh(k-$i)" for i in 1:na]; ["u(k-$i)" for i in 1:nb]]

function narx(d, na, nb;
    options,
    parallelism=:serial,
    niterations=2,
    kwargs...
)
   
    y, A = getARXregressor(d, na, nb)
    
    A = [A[:, 1] A] # All but the first column will be replaced by the simulation results

    T = eltype(y)
    yh = zero(y)::Vector{T}
    a = zeros(na+nb+1)

    workspaces = [Workspace2(copy(yh), copy(a)) for _ in 1:Threads.nthreads()]

    function simulate(tree, X::AbstractArray{T}, y, options) where T
        ws = workspaces[Threads.threadid()]
        ws.yh[1] = X[1]
        for i in 1:size(X, 2)-1
            ws.a[1] = X[1, i] # Use the first measurement for PEM-style feedback
            for j = (1:na) .+ 1
                ws.a[j] = ws.yh[max(i-j+2, 1)]
            end
            for j = (1:nb) .+ (na+1)
                ws.a[j] = X[j, i]
            end
    
            yhi, did_succeed = eval_tree_array(tree, reshape(ws.a, :, 1), options)
            if any(!isfinite, yhi)
                @views fill!(ws.yh[i:end], Inf)
                return ws.yh
            end
            ws.yh[i+1] = yhi[]
        end
        ws.yh
    end
    
    function simloss(tree, dataset, output)
        simulate(tree, dataset.X, dataset.y, options) # updates yh in place
        ws = workspaces[Threads.threadid()]
        ws.yh .= abs2.(y .- ws.yh) # reuse allocated memory
        sqrt(mean(ws.yh))
    end

    opts2 = @set options.loss_function = Simloss(simloss)
    hall_of_fame = EquationSearch(A', y; niterations, options=opts2, parallelism, varMap = varMap(na, nb), kwargs...)

    (; hall_of_fame, simulate, simloss, y, A)
end

#=
function expand_regressor(A0::AbstractMatrix, funs)
    A = copy(A0)
    for f in funs
        A = [A f.(A0)]
    end
end
function expand_regressor!(a, a0::AbstractVector{T}, funs) where T
    n = length(a0)
    # a = zeros(T, n*(length(funs) + 1))
    inds = 1:n
    a[inds] .= a0
    inds = inds .+ n
    for f in funs
        a[inds] .= f.(a0)
        inds = inds .+ n
    end
    a
end
=#

function SymbolicRegression.LossFunctionsModule.evaluator(
    sl::Simloss, tree::Node{T}, dataset::Dataset{T}, options::Options
)::T where {T<:Real}
    return sl.simloss(tree, dataset, options)
end

w = 2pi .* exp10.(LinRange(-3, log10(0.5), 500))
G0 = tf(1, [10, 1]) # The true system, 10ẋ = -x + u
G = c2d(G0, 1)      # discretize with a sample time of 1s

u0 = sign.(sin.((0:0.01:20) .^ 2))' .+ 0.9 # sample a control input for identification
u1 = sin.((0:0.01:20) .^ 2 .+ 1)' # sample a control input for identification
u = u0 + u1
unl = sqrt.(abs.(u))        # Apply nonlinear function

y, t, x = lsim(ss(G), unl) # Simulate the true system to get test data
yn = y .+ 0.1 .* randn.() # add measurement noise
d = iddata(yn, u, t[2] - t[1]) # create a data object

plot(d)

##

sqrtabs(x) = sqrt(abs(x))
options = SymbolicRegression.Options(
    binary_operators        = [+, *, -],
    unary_operators         = [sqrt, abs],
    complexity_of_operators = [sqrt => 2, (+) => 0.5],
    npopulations            = 11,
    maxdepth                = 12,
    complexity_of_constants = 0.1,
    nested_constraints      = [sqrt => [sqrt => 0], abs => [abs => 0]],
    enable_autodiff         = true,
)

na = 1
nb = 1
output = narx(d, na, nb; options, parallelism=:multithreading, niterations=10,)
hall_of_fame, simulate_fun, simloss_fun, y1, A = output

# with d[1:1000]
# 17.145589 seconds (268.45 M allocations: 13.675 GiB, 9.04% gc time, 8.87% compilation time: 100% of which was recompilation
# 17.339151 seconds (301.62 M allocations: 10.602 GiB, 7.22% gc time, 10.25% compilation time: 60% of which was recompilation) T type parameter
# 11.039410 seconds (206.37 M allocations: 7.552 GiB, 8.05% gc time, 14.18% compilation time: 100% of which was recompilation) type assert y
# 8.054081 seconds (134.20 M allocations: 4.934 GiB, 7.30% gc time, 19.54% compilation time: 100% of which was recompilation) reuse yh in simloss

dataset = (; y=y1, X=A')
dominating = calculate_pareto_frontier(A', y1, hall_of_fame, options)
trees = [member.tree for member in dominating]
ti = 3
tree = trees[ti]
default(show=false)
f1 = plot(d)
plot!(f1, d.t, y[:], lab="Noise-free y")
simlosses = Float64[]
for (ti, tree) in enumerate(trees[1:end])
    eqn = node_to_symbolic(tree, options, varMap=varMap(na, nb))
    push!(simlosses, simloss_fun(tree, dataset, options))
    println("Tree number $ti, simloss: $(round(simlosses[end], sigdigits=4))  ", Symbolics.simplify(eqn; expand=true))
    yh = simulate_fun(tree, A', y1, options)
    yh[1] = 0
    plot!(d.t[2:end], yh, sp=1, lab="Tree $ti")
end


plot(f1, plot(simlosses, m=:circle))
display(current())


## npem

# build a tree that computs (x⁺, y) = f(x, u)