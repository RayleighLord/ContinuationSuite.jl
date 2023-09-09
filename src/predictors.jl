abstract type Predictor end

abstract type TangentPredictor <: Predictor end

predict(uₛ₀, J, TangentPredictor) = error("The predictor method is not implemented.")

"""
    PseudoArcLength <: TangentPredictor

Uses pseudo-arclength continuation to make a prediction for the next solution.

From a known value of the solution `uₛ₀`, the prediction is updated by computing the tangent
vector `t` and the next solution is computed as:

```math
uₛ = uₛ₀ + Δs t,
```
where `Δs` is the continuation step size in the pseudo-arclength. The tangent vector `t` is
obtained from:

```math
t = \\frac{(xₛ, λₛ)}{\\|(xₛ, λₛ)\\|},

```
where `xₛ`, `λₛ` is the solution of the linear system:

```math
fₓ xₛ + fλ λₛ = 0,
```
which is equivalent to finding the nullspace of the extended jacobian [fₓ, fλ].

The tangent vector of the previous step is used to determine the direction of the tangent
vector `t`. If the dot product of the previous tangent vector and the current tangent vector
is negative, the tangent vector is flipped.
"""
struct PseudoArcLength <: TangentPredictor end

function predict(uₛ₀, tₚ, jac, Δs, ::PseudoArcLength)
    t = vec(nullspace(jac(uₛ₀)))
    (t ⋅ tₚ > 0.0) || (t .*= -1.0)

    uₛ = uₛ₀ + Δs * t
    return uₛ, t
end
