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
nos = Schwarz(finediffusion)

# p-multigrid preconditioner
multigrid = PMultigrid(finediffusion, coarsediffusion, nos, [ctofbasis])


# full operator symbols
numbersteps = 250
maxeigenvalue = 0
θ_min = -π/2
θ_max = 3π/2
θ_step = 2π/(numbersteps-1)
θ_range = θ_min:θ_step:θ_max

# compute and plot smoothing factor
# setup
ω = [2.0/3.0]
eigenvalues = zeros(numbersteps, finep)

# compute
for i in 1:numbersteps
    θ = [θ_range[i]]
    if abs(θ[1]) >  π/512
        A = computesymbols(multigrid, ω, [1,1], θ)
        #currenteigenvalues = [real(val) for val in eigvals(I-A)]
        currenteigenvalues = [real(val) for val in eigvals(A)]
        eigenvalues[i, :] = currenteigenvalues
    end
end

# plot
xrange = θ_range/π
plot(
    xrange,
    xlabel="θ/π",
    xtickfont=font(12, "Courier"),
    eigenvalues,
    ytickfont=font(12, "Courier"),
    ylabel="λ",
    linewidth=3,
    legend=:none,
    title="Spectrum of NOS Symbol",
    palette=palette(:tab10)
)
ylims!(min(0.0, eigenvalues...) * 1.1, max(eigenvalues...) * 1.1)
savefig("pmg_nos_spectrum_4_1")