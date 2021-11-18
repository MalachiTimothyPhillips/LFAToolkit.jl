# ------------------------------------------------------------------------------
# ASM preconditioner
# ------------------------------------------------------------------------------

"""
```julia
ASM(operator)
```

ASM diagonal preconditioner for finite element operators

# Arguments:
- `operator`: finite element operator to precondition

# Returns:
- ASM preconditioner object

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# preconditioner
asm = ASM(mass);

# verify
println(asm)
println(asm.operator)

# output
asm preconditioner
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
mutable struct ASM <: AbstractPreconditioner
    # data never changed
    operator::Operator

    # data empty until assembled
    subdomaininverse::AbstractArray{Float64}

    subdomaininverse::AbstractArray{Float64}

    # inner constructor
    ASM(operator::Operator) = new(operator)
end

# printing
# COV_EXCL_START
Base.show(io::IO, preconditioner::ASM) = print(io, "asm preconditioner")
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
getsubdomaininverse(preconditioner)
```

Compute or retrieve the inverse of the symbol matrix diagonal for a ASM
    preconditioner

# Returns:
- Symbol matrix diagonal inverse for the operator

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# preconditioner
asm = ASM(diffusion)

# note: either syntax works
diagonalinverse = LFAToolkit.getsubdomaininverse(asm);
diagonalinverse = asm.subdomaininverse;

# verify
@assert diagonalinverse ≈ [6/7 0; 0 3/4]
 
# output

```
"""
function getsubdomaininverse(preconditioner::ASM)
    # assemble if needed
    if !isdefined(preconditioner, :subdomaininverse)
        # WARNING: hard-coded to 1d
        elementmatrix = preconditioner.operator.elementmatrix
        Nq, _ = size(elementmatrix)
        p = Nq-1
        Nqe = Nq+2
        pe = Nqe-1
        overlappingop = zeros(Nqe,Nqe)
        overlappingop[2:Nqe-1,2:Nqe-1] = elementmatrix
        overlappingop[1:2,1:2] += elementmatrix[p:Nq,p:Nq]
        overlappingop[pe:Nqe,pe:Nqe] += elementmatrix[1:2,1:2]
        
        subdomaininverse = pinv(overlappingop)

        preconditioner.subdomaininverse = subdomaininverse
    end

    # return
    return getfield(preconditioner, :subdomaininverse)
end

function getQs(preconditioner::ASM)
    # assemble if needed
    if !isdefined(preconditioner, :subdomaininverse)
        # WARNING: hard-coded to 1d
        elementmatrix = preconditioner.operator.elementmatrix
        Nq, _ = size(elementmatrix)
        p = Nq-1
        Nqe = Nq+2
        pe = Nqe-1

        Qs = zeros(Nqe,Nq)
        Qs[1,Nqe-1] = 1.0
        Qs[Nqe,2] = 1.0
        Qs[2:Nqe-1,1:Nq] = eye(Nq)

        preconditioner.Qs = Qs
    end

    # return
    return getfield(preconditioner, :Qs)
end

function getextendednodecoordinatedifferences(preconditioner::ASM)
    # assemble if needed
    if !isdefined(preconditioner, :extendednodecoordinatedifferences)
        # WARNING: hard-coded to 1d
        elementmatrix = preconditioner.operator.elementmatrix
        Nq, _ = size(elementmatrix)
        p = Nq-1
        Nqe = Nq+2
        pe = Nqe-1

        Qs = zeros(Nqe,Nq)
        Qs[1,Nqe-1] = 1.0
        Qs[Nqe,2] = 1.0
        Qs[2:Nqe-1,1:Nq] = eye(Nq)

        preconditioner.Qs = Qs
    end

    # return
    return getfield(preconditioner, :Qs)
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(preconditioner::ASM, f::Symbol)
    if f == :subdomaininverse
        return getsubdomaininverse(preconditioner)
    elseif f == :Qs
        return getQs(preconditioner)
    else
        return getfield(preconditioner, f)
    end
end

function Base.setproperty!(preconditioner::ASM, f::Symbol, value)
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

Compute or retrieve the symbol matrix for a ASM preconditioned operator

# Arguments:
- `preconditioner`: ASM preconditioner to compute symbol matrix for
- `ω`:              Nothing
- `θ`:              Fourier mode frequency array (one frequency per dimension)

# Returns:
- Symbol matrix for the ASM preconditioned operator

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
    asm = ASM(diffusion);

    # compute symbols
    A = computesymbols(asm, [], π*ones(dimension));

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
function computesymbols(preconditioner::ASM, ω::Array, θ::Array)

    rowmodemap = preconditioner.operator.rowmodemap
    columnmodemap = preconditioner.operator.columnmodemap
    dimension = preconditioner.operator.dimension
    elementmatrix = preconditioner.operator.elementmatrix
    Minv = preconditioner.subdomaininverse
    nodecoordinatedifferences = preconditioner.operator.nodecoordinatedifferences
    numberrows, numbercolumns = size(elementmatrix)

    symbolmatrixnodes = zeros(ComplexF64, numberrows, numbercolumns)
    for i = 1:numberrows, j = 1:numbercolumns
        symbolmatrixnodes[i, j] =
            Minv[i, j] *
            ℯ^(im * sum([θ[k] * nodecoordinatedifferences[i, j, k] for k = 1:dimension]))
    end
    symbolmatrixmodes = rowmodemap * symbolmatrixnodes * columnmodemap

    A = computesymbols(preconditioner.operator, θ)
    return I - symbolmatrixmodes * A
end

# ------------------------------------------------------------------------------
