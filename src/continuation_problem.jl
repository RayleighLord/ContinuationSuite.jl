abstract type AbstractContinuationProblem end

"""
    ContinuationProblem(f, ContPars [, p]; jac = nothing, autodiff = true)

Defines a Continuation Problem

To define a continuation problem, you need to provide the function `f` and the continuation parameters `ContPars`. Optionally, you can provide the Jacobian `jac` and additional parameters in `p`. The `autodiff` keyword argument is used to determine whether to use automatic differentiation or not. The default is `true`. This setting is ignored if the jacobian is provided.

A continuation problem is of the form:

```math
f(x, λ, p) = 0,
```
where `p` is a parameter vector, `x` is the solution vector and `λ` is the continuation parameter.

# Fields
    - `f`: The function defining the continuation problem
    - `jac`: The Jacobian of the function defining the continuation problem
    - `p`: The parameter vector
    - `ContPars`: The continuation parameters
    - `autodiff`: Whether to use automatic differentiation or not
    - `f_org`: The original function provided by the user
    - `jac_org`: The original Jacobian provided by the user

# Example of usage
```julia
using NumericalContinuation

function f(x, λ, p)
    x₁, x₂ = x
    f₁ = x₁^2 + x₂^2 - λ
    f₂ = x₂^2 - 2x₁ + 1
    return [f₁, f₂]
end

function jac(x, λ, p)
    x₁, x₂ = x
    fₓ = [2x₁ 2x₂; -2 2x₂]
    fλ = [-1; 0]
    return [fₓ fλ]
end

cont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.5, Δs = 0.25,
                                   direction = :backward, predictor = PseudoArcLength(),
                                   corrector = Newton(), verbose = false)

prob = ContinuationProblem(f, cont_pars; autodiff = false)
x₀, λ₀ = [1.0, -1.0], 2.0

sol = continuation(prob, x₀, λ₀)
```
"""
struct ContinuationProblem{F, J, ptype, C, OF, OJ} <: AbstractContinuationProblem
    f::F
    jac::J
    p::ptype
    ContPars::C
    autodiff::Bool
    f_org::OF
    jac_org::OJ
end

function ContinuationProblem(func, ContPars, p = nothing; jac = nothing, autodiff = true)
    f = u -> func(u[1:(end - 1)], u[end], p)

    jac_red = (jac === nothing) ? nothing : (u, p) -> jac(u[1:(end - 1)], u[end], p)
    jacobian = jacobian_function(jac_red, autodiff, f, p)

    return ContinuationProblem(f, jacobian, p, ContPars, autodiff, func, jac)
end

function Base.show(io::IO, ::MIME"text/plain", prob::ContinuationProblem)
    @unpack jac, ContPars, autodiff = prob
    @unpack λmin, λmax, Δs, direction, predictor, corrector = ContPars
    println(io, "Continuation Problem with parameters:")

    if jac === nothing
        println(io, "- autodiff: $(autodiff)")
    else
        println(io, "- jac: analytical")
    end

    println(io, "- λmin: $(λmin)")
    println(io, "- λmax: $(λmax)")
    println(io, "- Δs: $(Δs)")
    println(io, "- predictor: $(nameof(typeof(predictor)))")
    println(io, "- corrector: $(nameof(typeof(corrector)))")
end
