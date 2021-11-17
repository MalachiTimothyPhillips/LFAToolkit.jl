# ------------------------------------------------------------------------------
# Schwarz smoother example
# ------------------------------------------------------------------------------

using LFAToolkit
using LinearAlgebra

# setup
mesh = Mesh1D(2)
p = 2

# diffusion operator
diffusion = GalleryOperator("diffusion", p + 1, p + 1, mesh)
# Schwarz smoother
schwarz = Schwarz(diffusion)

# compute operator symbols
A = computesymbols(schwarz, [], [π])
eigenvalues = real(eigvals(A))
#println(eigenvalues)

# ------------------------------------------------------------------------------
