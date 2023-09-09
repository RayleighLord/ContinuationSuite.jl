# ContinuationSuite

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RayleighLord.github.io/ContinuationSuite.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RayleighLord.github.io/ContinuationSuite.jl/dev/)
[![Build Status](https://github.com/RayleighLord/ContinuationSuite.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/RayleighLord/ContinuationSuite.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/RayleighLord/ContinuationSuite.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/RayleighLord/ContinuationSuite.jl)

## Example usage

```julia
using ContinuationSuite

function f(x, λ, p)
    x₁, x₂ = x
    f₁ = x₂^3 - (2 - √(x₁^2 + λ)) * (x₁^2 + λ)^2
    f₂ = x₁^4 - 6x₁^2 * λ + λ^2 + x₂^3
    return [f₁, f₂]
end

cont_pars = ContinuationParameters(λmin = 0.0, λmax = 9.0, Δs = 0.05, maxsteps = 3000,
    direction = :forward, predictor = PseudoArcLength(),
    corrector = Newton(), verbose = true)

prob = ContinuationProblem(f, cont_pars; autodiff = true)
x₀, λ₀ = [3, -3 * 3^(1 / 3)], 0.0

sol = continuation(prob, x₀, λ₀)
```
