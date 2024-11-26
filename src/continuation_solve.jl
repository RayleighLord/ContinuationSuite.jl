"""
    continuation(prob::ContinuationProblem, x₀, λ₀)

Solve the continuation problem `prob` starting from the initial solution `x₀`
and the initial parameter value `λ₀`. Return a `ContinuationSolution` object
containing the solution.

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
```
"""
function continuation(prob::AbstractContinuationProblem, x₀, λ₀)
    @unpack ContPars = prob
    @unpack λmin, λmax, verbose, maxsteps, ncols = ContPars
    info = ContPars.corrector.info

    u₁, t = initialize_solution(x₀, λ₀, prob)
    u = [u₁]

    verbose && print_evolution(1, u₁, info; ncols = ncols)

    for n in 1:(maxsteps - 1)
        uₙ₊₁, t = continue_step(u[end], t, prob)
        push!(u, uₙ₊₁)

        if (λmin ≥ uₙ₊₁[end]) || (uₙ₊₁[end] ≥ λmax)
            final_step!(u[end], u[end - 1], λmax, λmin, prob)
            verbose && print_evolution(n + 1, u[end], info; ncols = ncols)
            break
        end

        verbose && print_evolution(n + 1, uₙ₊₁, info; ncols = ncols)
    end

    return build_solution(u, prob)
end

function continue_step(uₙ, tₚ, prob::AbstractContinuationProblem)
    @unpack jac, ContPars = prob
    @unpack Δs, predictor = ContPars

    uₛ, t = predict(uₙ, tₚ, jac, Δs, predictor)
    uₙ₊₁ = correct(uₛ, uₙ, t, Δs, prob)

    return uₙ₊₁, t
end

function initialize_solution(x₀, λ₀, prob::AbstractContinuationProblem)
    @unpack ContPars = prob
    @unpack direction = ContPars

    u_init = vcat(x₀, λ₀)

    t = zeros(length(u_init))
    t[end] = direction == :forward ? 1.0 : -1.0

    u₁ = correct(u_init, u_init, t, 0.0, prob)

    return u₁, t
end

function final_step!(uₙ₊₁, uₙ, λmax, λmin, prob::AbstractContinuationProblem)
    @unpack ContPars = prob
    @unpack direction = ContPars

    if uₙ₊₁[end] ≤ λmin
        u, t = initialize_solution(uₙ[1:(end - 1)], λmin, prob)
        uₙ₊₁ .= u
    elseif uₙ₊₁[end] ≥ λmax
        u, t = initialize_solution(uₙ[1:(end - 1)], λmax, prob)
        uₙ₊₁ .= u
    end
end

function compute_stability(u, jac)
    J = zeros(length(u) - 1, length(u) - 1)
    J .= jac(u)[:, 1:(end - 1)]

    eigenvalues = eigvals(J)
    is_stable = maximum(real, eigenvalues) < 0.0

    return is_stable
end

function print_evolution(n, uₙ, info; ncols = 2)
    n == 1 && create_header(ncols = ncols)

    res_f, res_u, iters = info
    x, λ = uₙ[1:ncols], uₙ[end]

    values = [x..., λ, res_u, res_f]

    @printf("%12d ", n)
    for value in values
        @printf("%12.4e ", value)
    end
    @printf("%12d ", iters)
    @printf("\n")
end

function create_header(; ncols = 2)
    labels = ["Step"]
    push!(labels, ["x_$i" for i in 1:ncols]...)
    push!(labels, "λ")
    push!(labels, ["Res_u", "Res_f", "Iters"]...)

    for label in labels
        @printf("%12s ", label)
    end
    @printf("\n")
end
