abstract type AbstractSolution end

"""
    ContinuationSolution(u, prob::ContinuationProblem)

Constructs a solution object for a continuation problem.

The solution is represented by a vector of the parameter values `λ` and a matrix of solutions `x` with the different components.

# Fields
- `x`: The solution matrix.
- `λ`: The parameter values.
- `prob`: The continuation problem.
- `ncols`: The number of columns of the solution matrix `x` printed in the terminal.

# Example of usage
```julia
using NumericalContinuation

function f(x, λ, p)
    x₁, x₂ = x
    f₁ = x₁^2 + x₂^2 - λ
    f₂ = x₂^2 - 2x₁ + 1
    return [f₁, f₂]
end

cont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.5, Δs = 0.25,
                                   direction = :backward, predictor = PseudoArcLength(),
                                   corrector = Newton(), verbose = false)

prob = ContinuationProblem(f, cont_pars; autodiff = false)
x₀, λ₀ = [1.0, -1.0], 2.0

sol = continuation(prob, x₀, λ₀)
x, λ = sol.x, sol.λ
```
"""
struct ContinuationSolution{xType, λType, P} <: AbstractSolution
    x::xType
    λ::λType
    prob::P
    ncols::Int
end

function ContinuationSolution(u, prob::AbstractContinuationProblem)
    ncols = prob.ContPars.ncols
    u = reduce(hcat, u)[:, 1:end]
    x, λ = u[1:(end - 1), :], u[end, :]
    return ContinuationSolution(x, λ, prob, ncols)
end

function build_solution(u, prob::AbstractContinuationProblem)
    return ContinuationSolution(u, prob)
end

function Base.show(io::IO, ::MIME"text/plain", sol::ContinuationSolution)
    @unpack x, λ, ncols = sol

    u = [x; λ']'
    m = size(u, 1)

    labels = ["x_$i" for i in 1:ncols]
    push!(labels, "λ")

    println(io, "Continuation Problem Solution:")

    for i in eachindex(labels)
        @printf(io, "%12s ", labels[i])
    end
    println(io)

    for i in 1:m
        for j in 1:(ncols + 1)
            @printf(io, "%12.4e ", u[i, j])
        end
        println(io)
    end
end
