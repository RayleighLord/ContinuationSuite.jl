abstract type AbstractCache end

"""
    ContinuationParameters{T <: Real, P <: Predictor, NLS <: NonlinearEqSolver} <: AbstractCache

Continuation parameters for the continuation problem. The fields are

# Fields
    - `λmin::T`: minimum value of the continuation parameter `λ`. (default: -1.0)
    - `λmax::T`: maximum value of the continuation parameter `λ`. (default: 1.0)
    - `Δs::T`: continuation step size. (default: 0.1)
    - `direction::Symbol`: direction of the continuation. Can be `:forward` or `:backward`. (default: `:forward`)
    - `maxsteps::Int`: maximum number of continuation steps. (default: 100)
    - `predictor::P`: predictor method. (default: `PseudoArcLength()`)
    - `corrector::NLS`: corrector method. (default: `Newton()`)
    - `verbose::Bool`: whether to print information about the continuation process or not. (default: `false`)
    - `ncols::Int`: number of columns to use when printing information about the continuation process. (default: 2)
"""
@with_kw struct ContinuationParameters{T <: Real, P <: Predictor,
                                       NLS <: NonlinearEqSolver} <: AbstractCache
    λmin::T = -1.0
    λmax::T = 1.0

    @assert λmin<λmax "λmin must be less than λmax"

    Δs::T = 0.1

    @assert Δs≥0 "Δs must be positive"

    direction::Symbol = :forward

    @assert direction in (:forward, :backward) "direction must be :forward or :backward"

    maxsteps::Int = 100

    @assert maxsteps>0 "maxsteps must be positive"

    predictor::P = PseudoArcLength()
    corrector::NLS = Newton()
    verbose::Bool = false
    ncols::Int = 2
end

function Base.show(io::IO, ::MIME"text/plain", c_pars::ContinuationParameters)
    println(io, "Continuation Parameters configured")
end
