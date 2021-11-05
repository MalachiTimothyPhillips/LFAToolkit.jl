# ------------------------------------------------------------------------------
# Schwarz smoother example
# ------------------------------------------------------------------------------

using LFAToolkit
using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
p = 3

# diffusion operator
diffusion = GalleryOperator("diffusion", p + 1, p + 1, mesh)

# Schwarz smoother
schwarz = Schwarz(diffusion)

# compute operator symbols
A = computesymbols(schwarz, [1.0], [π, π])
eigenvalues = real(eigvals(A))
println(eigenvalues)

# ------------------------------------------------------------------------------
