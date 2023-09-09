abstract type LinearSolver end

linsolve(A, b, ::LinearSolver) = error("The linear solver is not implemented.")

"""
    LUSolver <: LinearSolver

The default linear solver just calls the backslash `\\` operator to solve the linear system
```
A * x = b
x = A \\ b
```
"""
struct LUSolver <: LinearSolver end

linsolve(A, b, ::LUSolver) = A \ b
