# ------------------------------------------------------------------------------
# Schwarz preconditioner
# ------------------------------------------------------------------------------

"""
```julia
Schwarz(operator)
```

Schwarz diagonal preconditioner for finite element operators

# Arguments:
- `operator`: finite element operator to precondition

# Returns:
- Schwarz preconditioner object

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# preconditioner
schwarz = Schwarz(mass);

# verify
println(schwarz)
println(schwarz.operator)

# output
schwarz preconditioner
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
```
"""
mutable struct Schwarz <: AbstractPreconditioner
    # data never changed
    operator::Operator

    # data empty until assembled
    operatordiagonalinverse::AbstractArray{Float64}

    # inner constructor
    Schwarz(operator::Operator) = new(operator)
end

# printing
# COV_EXCL_START
Base.show(io::IO, preconditioner::Schwarz) = print(io, "schwarz preconditioner")
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
getoperatordiagonalinverse(preconditioner)
```

Compute or retrieve the inverse of the symbol matrix diagonal for a Schwarz
    preconditioner

# Returns:
- Symbol matrix diagonal inverse for the operator

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# preconditioner
schwarz = Schwarz(diffusion)

# note: either syntax works
diagonalinverse = LFAToolkit.getoperatordiagonalinverse(schwarz);
diagonalinverse = schwarz.operatordiagonalinverse;

# verify
@assert diagonalinverse ≈ [6/7 0; 0 3/4]
 
# output

```
"""
function getoperatordiagonalinverse(preconditioner::Schwarz)
    # assemble if needed
    if !isdefined(preconditioner, :operatordiagonalinverse)
        # retrieve diagonal and invert
        diagonalinverse = preconditioner.operator.diagonal^-1

        # store
        preconditioner.operatordiagonalinverse = diagonalinverse
    end

    # return
    return getfield(preconditioner, :operatordiagonalinverse)
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(preconditioner::Schwarz, f::Symbol)
    if f == :operatordiagonalinverse
        return getoperatordiagonalinverse(preconditioner)
    else
        return getfield(preconditioner, f)
    end
end

function Base.setproperty!(preconditioner::Schwarz, f::Symbol, value)
    if f == :operator
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(preconditioner, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(preconditioner, ω, θ)
```

Compute or retrieve the symbol matrix for a Schwarz preconditioned operator

# Arguments:
- `preconditioner`: Schwarz preconditioner to compute symbol matrix for
- `ω`:              Smoothing weighting factor array
- `θ`:              Fourier mode frequency array (one frequency per dimension)

# Returns:
- Symbol matrix for the Schwarz preconditioned operator

# Example:
```jldoctest
using LinearAlgebra

for dimension in 1:3
    # setup
    mesh = []
    if dimension == 1
        mesh = Mesh1D(1.0);
    elseif dimension == 2
        mesh = Mesh2D(1.0, 1.0);
    elseif dimension == 3
        mesh = Mesh3D(1.0, 1.0, 1.0);
    end
    diffusion = GalleryOperator("diffusion", 3, 3, mesh);

    # preconditioner
    schwarz = Schwarz(diffusion);

    # compute symbols
    A = computesymbols(schwarz, [1.0], π*ones(dimension));

    # verify
    using LinearAlgebra;
    eigenvalues = real(eigvals(A));
    if dimension == 1
        @assert max(eigenvalues...) ≈ 1/7
    elseif dimension == 2
        @assert min(eigenvalues...) ≈ -1/14
    elseif dimension == 3
        @assert min(eigenvalues...) ≈ -0.33928571428571486
    end
end

# output

```
"""
function computesymbols(preconditioner::Schwarz, ω::Array, θ::Array)
    # validate number of parameters
    if length(ω) != 1
        Throw(error("exactly one parameter required for Schwarz smoothing")) # COV_EXCL_LINE
    end

    # return
    return I -
           ω[1] *
           preconditioner.operatordiagonalinverse *
           computesymbols(preconditioner.operator, θ)
end

# ------------------------------------------------------------------------------
