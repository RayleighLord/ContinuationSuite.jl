abstract type AbstractNonlinearProblem end

"""
    NonlinearProblem(f, [, p]; jac = nothing, autodiff = true)

Defines a Nonlinear System of Equations Problem

To define a nonlinear problem, you need to provide the function `f`. Optionally, you can
provide the Jacobian `jac` and additional parameters in `p`. The `autodiff` keyword argument
is used to determine whether to use automatic differentiation or not. The default is `true`.
This setting is ignored if the jacobian is provided.

A nonlinear problem is of the form:

```math
f(x, p) = 0,
```
where `p` is a parameter vector and `x` is the solution vector.

# Fields
    - `f`: The function defining the nonlinear problem
    - `jac`: The Jacobian of the function defining the nonlinear problem
    - `p`: The parameter vector
    - `autodiff`: Whether to use automatic differentiation or not
    - `f_org`: The original function provided by the user
    - `jac_org`: The original Jacobian provided by the user

# Example of usage
```julia
using NumericalContinuation

f(x, p) = [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 1]
jac(x, p) = [2x[1] 2x[2]; 1 1]

p = 0.0

prob = NonlinearProblem(f)
prob = NonlinearProblem(f, p)
prob = NonlinearProblem(f, p; jac = jac)
```
"""
struct NonlinearProblem{F, J, ptype, OF, OJ} <: AbstractNonlinearProblem
    f::F
    jac::J
    p::ptype
    autodiff::Bool
    f_org::OF
    jac_org::OJ
end

function NonlinearProblem(func, p = nothing; jac = nothing, autodiff = true)
    f = x -> func(x, p)
    jacobian = jacobian_function(jac, autodiff, f, p)

    return NonlinearProblem(f, jacobian, p, autodiff, func, jac)
end
