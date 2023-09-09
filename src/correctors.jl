function correct(uₛ, uₙ, t, Δs, prob::AbstractContinuationProblem)
    @unpack f, jac, ContPars = prob
    @unpack corrector = ContPars

    f_aug = u -> [f(u); ((u - uₙ) ⋅ t - Δs)]
    jac_aug = u -> [jac(u); t']

    uₙ₊₁ = nlsolve(f_aug, jac_aug, uₛ, corrector)

    return uₙ₊₁
end
