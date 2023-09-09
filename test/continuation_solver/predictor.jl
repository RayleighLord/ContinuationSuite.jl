using ContinuationSuite
using Test

@testset "Prediction with Pseudo-arclength and explicit jacobian" begin
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

    prob = ContinuationProblem(f, cont_pars; jac = jac)

    tₚ = [0.0, 0.0, 1.0]
    uₛ₀ = [1.0, 1.0, 2.0]
    jacobian = prob.jac
    Δs = cont_pars.Δs

    u, t = ContinuationSuite.predict(uₛ₀, tₚ, jacobian, Δs, PseudoArcLength())

    t = t ./ t[end]

    @test t ≈ [0.25, 0.25, 1.0]
end

@testset "Prediction with Pseudo-arclength and autodiff jacobian" begin
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

    p = 0.0

    cont_pars = ContinuationParameters(λmin = 0.0, λmax = 2.5, Δs = 0.25,
        direction = :backward, predictor = PseudoArcLength(),
        corrector = Newton(), verbose = false)

    prob = ContinuationProblem(f, p, cont_pars)

    tₚ = [0.0, 0.0, 1.0]
    uₛ₀ = [1.0, 1.0, 2.0]
    jacobian = prob.jac
    Δs = cont_pars.Δs

    u, t = ContinuationSuite.predict(uₛ₀, tₚ, jacobian, Δs, PseudoArcLength())

    t = t ./ t[end]

    @test t ≈ [0.25, 0.25, 1.0]
end

@testset "Prediction with Pseudo-arclength and finite diff jacobian" begin
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

    tₚ = [0.0, 0.0, 1.0]
    uₛ₀ = [1.0, 1.0, 2.0]
    jacobian = prob.jac
    Δs = cont_pars.Δs

    u, t = ContinuationSuite.predict(uₛ₀, tₚ, jacobian, Δs, PseudoArcLength())

    t = t ./ t[end]

    @test t ≈ [0.25, 0.25, 1.0]
end
