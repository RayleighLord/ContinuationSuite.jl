abstract type NonlinearEqSolver end

"""
    Newton{T <: Real, LS <: LinearSolver} <: NonlinearEqSolver

The Newton Raphson method for solving nonlinear equations. The nonlinear problem
is solved iteratively using

```math
J(x_n) Δx_n = -f(x_n)
```
where `J(x_n)` is the Jacobian matrix of `f(x_n)` and `Δx_n` is the update to the solution
at the `n`th iteration. The update is given by
```math
x_{n+1} = x_n + Δx_n.
```

# Arguments
- `ϵₓ::T = 1e-8`: The tolerance for the norm of the successive iterations ||xₙ₊₁ - xₙ||/||xₙ||
- `ϵᵣ::T = 1e-8`: The tolerance for the norm of the residual ||f(xₙ)||
- `maxiters::Int = 20`: The maximum number of iterations.
- `lsolver::LS = LUSolver()`: The linear solver to use to solve the linear system.
- `info::Vector{Float64} = [0.0, 0.0, 0.0]`: The information about the nonlinear solver.
    - `info[1]`: The norm of the residual ||f(xₙ)||
    - `info[2]`: The norm of the successive iterations ||xₙ₊₁ - xₙ||/||xₙ||
    - `info[3]`: The number of iterations.

# Examples
```julia
using NumericalContinuation

f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]
jac(x, p) = [2x[1] 2x[2]; 1 1]

p = 0.0

prob = NonlinearProblem(f, p; jac = jac)

x0 = [0.25, 0.5]
sol = solve(prob, Newton(), x0)
sol = solve(prob, Newton(ϵₓ = 1e-10, maxiters = 100), x0) # with different parameters
```
"""
@with_kw struct Newton{T <: Real, LS <: LinearSolver} <: NonlinearEqSolver
    ϵₓ::T = 1e-8
    ϵᵣ::T = 1e-8
    maxiters::Int = 20
    linsolver::LS = LUSolver()
    info::Vector{Float64} = [0.0, 0.0, 0.0]
end

function nlsolve(func, jac, x₀, nlsolver::Newton)
    @unpack ϵₓ, ϵᵣ, maxiters, linsolver, info = nlsolver

    x = copy(x₀)
    f = similar(x)
    Δx = similar(x)
    J = similar(x, length(x), length(x))

    res_x, res_f = 0.0, 0.0

    f .= func(x)
    J .= jac(x)

    for k in 1:maxiters
        Δx .= linsolve(J, -f, linsolver)

        x .+= Δx
        f .= func(x)

        res_x, res_f = norm(Δx) / (norm(x) + eps()), norm(f)

        if (res_x < ϵₓ) && (res_f < ϵᵣ)
            info .= [res_f, res_x, k]
            return x
        end

        J .= jac(x)
    end
    error("The nonlinear equation solver failed to converge. Residuals are ||f_n||",
        " = $(res_f) and ||xₙ₊₁ - xₙ||/||xₙ|| = $(res_x)")
end

"""
    Broyden{T <: Real, LS <: LinearSolver} <: NonlinearEqSolver

The Broyden's method for solving nonlinear equations. The nonlinear problem
is solved iteratively using Broyden's method, which is a quasi-Newton method.

Broyden's method approximates the Jacobian matrix and updates it at every iteration.
The update formula for the Jacobian is

```math
J_{n+1} = J_n + ((f_{n+1} - f_n) - J_n Δx_n) Δx_n' / (Δx_n' Δx_n)
```
where `J_n` is the Jacobian matrix at the nth iteration and `Δx_n = x_{n+1} - x_n` is the change in the solution.


# Arguments
- `ϵₓ::T = 1e-8`: The tolerance for the norm of the successive iterations ||xₙ₊₁ - xₙ||/||xₙ||
- `ϵᵣ::T = 1e-8`: The tolerance for the norm of the residual ||f(xₙ)||
- `maxiters::Int = 20`: The maximum number of iterations.
- `lsolver::LS = LUSolver()`: The linear solver to use to solve the linear system.
- `info::Vector{Float64} = [0.0, 0.0, 0.0]`: The information about the nonlinear solver.
    - `info[1]`: The norm of the residual ||f(xₙ)||
    - `info[2]`: The norm of the successive iterations ||xₙ₊₁ - xₙ||/||xₙ||
    - `info[3]`: The number of iterations.

# Examples
```julia
using NumericalContinuation

f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]
jac(x, p) = [2x[1] 2x[2]; 1 1]

p = 0.0

prob = NonlinearProblem(f, p; jac = jac)

x0 = [0.25, 0.5]
sol = solve(prob, Broyden(), x0)
sol = solve(prob, Broyden(ϵₓ = 1e-10, maxiters = 100), x0) # with different parameters
```
"""
@with_kw struct Broyden{T <: Real, LS <: LinearSolver} <: NonlinearEqSolver
    ϵₓ::T = 1e-8
    ϵᵣ::T = 1e-8
    maxiters::Int = 20
    linsolver::LS = LUSolver()
    info::Vector{Float64} = [0.0, 0.0, 0.0]
end

function nlsolve(func, jac, x₀, nlsolver::Broyden)
    @unpack ϵₓ, ϵᵣ, maxiters, linsolver, info = nlsolver

    x = copy(x₀)
    f = similar(x)
    Δx = similar(x)
    J = similar(x, length(x), length(x))
    f_new = similar(x)

    res_x, res_f = 0.0, 0.0

    f .= func(x)
    J .= jac(x)

    for k in 1:maxiters
        Δx .= linsolve(J, -f, linsolver)

        x .+= Δx
        f_new .= func(x)

        res_x, res_f = norm(Δx) / (norm(x) + eps()), norm(f_new)

        if (res_x < ϵₓ) && (res_f < ϵᵣ)
            info .= [res_f, res_x, k]
            return x
        end

        J .+= ((f_new - f) - J * Δx) * Δx' / (Δx' * Δx)
        f .= f_new
    end
    error("The nonlinear equation solver failed to converge. Residuals are ||f_n||",
        " = $(res_f) and ||xₙ₊₁ - xₙ||/||xₙ|| = $(res_x)")
end
