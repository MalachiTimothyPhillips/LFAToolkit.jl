# dependencies
using LFAToolkit
using LinearAlgebra
using Plots
# setup
p = 2
dimension = 1
mesh = Mesh1D(1.0)

# operator
diffusion = GalleryOperator("diffusion", p+1, p+1, mesh)

# ChebSchwarz smoother
chebyshev = ChebSchwarz(diffusion)
# full operator symbols
numbersteps = 250
maxeigenvalue = 0
θ_min = -π/2
θ_max = 3π/2
θ_step = 2π/(numbersteps-1)
θ_range = θ_min:θ_step:θ_max

# compute and plot smoothing factor
# setup
ω = [3]
eigenvalues = zeros(numbersteps, p)

# compute
for i in 1:numbersteps
    θ = [θ_range[i]]
    if abs(θ[1]) > π/512
        A = computesymbols(chebyshev, ω, θ)
        currenteigenvalues = [real(val) for val in eigvals(I-A)]
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
    title="Spectrum of Third Order ChebSchwarz Symbol",
    palette=palette(:tab10)
)
ylims!(min(0.0, eigenvalues...) * 1.1, max(eigenvalues...) * 1.1)
savefig("chebyshev_schwarz_spectrum_2")

# setup
p = 4
dimension = 1
mesh = Mesh1D(1.0)

# operator
diffusion = GalleryOperator("diffusion", p+1, p+1, mesh)

# ChebSchwarz smoother
chebyshev = ChebSchwarz(diffusion)
# full operator symbols
numbersteps = 250
maxeigenvalue = 0
θ_min = -π/2
θ_max = 3π/2
θ_step = 2π/(numbersteps-1)
θ_range = θ_min:θ_step:θ_max

# compute and plot smoothing factor
# setup
ω = [3]
eigenvalues = zeros(numbersteps, p)

# compute
for i in 1:numbersteps
    θ = [θ_range[i]]
    if abs(θ[1]) > π/512
        A = computesymbols(chebyshev, ω, θ)
        currenteigenvalues = [real(val) for val in eigvals(I-A)]
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
    title="Spectrum of Third Order ChebSchwarz Symbol",
    palette=palette(:tab10)
)
ylims!(min(0.0, eigenvalues...) * 1.1, max(eigenvalues...) * 1.1)
savefig("chebyshev_schwarz_spectrum_4")


p = 7
dimension = 1
mesh = Mesh1D(1.0)

# operator
diffusion = GalleryOperator("diffusion", p+1, p+1, mesh)

# ChebSchwarz smoother
chebyshev = ChebSchwarz(diffusion)
# full operator symbols
numbersteps = 250
maxeigenvalue = 0
θ_min = -π/2
θ_max = 3π/2
θ_step = 2π/(numbersteps-1)
θ_range = θ_min:θ_step:θ_max

# compute and plot smoothing factor
# setup
ω = [3]
eigenvalues = zeros(numbersteps, p)

# compute
for i in 1:numbersteps
    θ = [θ_range[i]]
    if abs(θ[1]) > π/512
        A = computesymbols(chebyshev, ω, θ)
        currenteigenvalues = [real(val) for val in eigvals(I-A)]
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
    title="Spectrum of Third Order ChebSchwarz Symbol",
    palette=palette(:tab10)
)
ylims!(min(0.0, eigenvalues...) * 1.1, max(eigenvalues...) * 1.1)
savefig("chebyshev_schwarz_spectrum_7")