# dependencies
using LFAToolkit
using LinearAlgebra
using Plots

# setup
finep = 4
coarsep = 1
numbercomponents = 1
dimension = 1
mesh = Mesh1D(1.0)

ctofbasis = TensorH1LagrangePProlongationBasis(coarsep+1, finep+1, numbercomponents, dimension)

# diffusion operators
finediffusion = GalleryOperator("diffusion", finep+1, finep+1, mesh)
coarsediffusion = GalleryOperator("diffusion", coarsep+1, finep+1, mesh)

# Chebyshev smoother
chebyshev = Chebyshev(finediffusion)

# p-multigrid preconditioner
multigrid = PMultigrid(finediffusion, coarsediffusion, chebyshev, [ctofbasis])

# full operator symbols
numbersteps = 250
maxeigenvalue = 0
θ_min = -π/2
θ_max = 3π/2
θ_step = 2π/(numbersteps-1)
θ_range = θ_min:θ_step:θ_max

# compute and plot smoothing factor
# setup
eigenvalues = zeros(4)

# compute
xrange = [1, 2, 3, 4]
for i in 1:numbersteps, j in xrange
    θ = [θ_range[i]]
    ω = [j]
    if abs(θ[1]) >  π/128
        A = computesymbols(multigrid, ω, [1, 1], θ)
        currenteigenvalues = [abs(val) for val in eigvals(A)]
        eigenvalues[j] = max(eigenvalues[j], currenteigenvalues...)
    end
end

# plot
bar(
    xrange,
    xlabel="k",
    xtickfont=font(12, "Courier"),
    eigenvalues,
    ytickfont=font(12, "Courier"),
    ylabel="spectral radius",
    linewidth=3,
    legend=:none,
    title="Two-Grid Convergence Factor",
    palette=palette(:tab10)
)
ylims!(min(0.0, eigenvalues...) * 1.1, max(eigenvalues...) * 1.1)

savefig("two_grid_converge_4_to_1_chebyshev")