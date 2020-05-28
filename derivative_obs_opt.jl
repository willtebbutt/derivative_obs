using Stheno, Optim, Plots

using Stheno: derivative

# Helper function: Parameter handling.
function unpack(θ)
    σ² = exp(θ[1]) + 1e-6
    l = exp(θ[2]) + 1e-6
    return σ², l
end

# Helper function: construct the model and it's derivative.
function build_model(θ)
    σ², l = unpack(θ)
    k = σ² * stretch(EQ(), 1 / l)
    f = GP(k, GPC())
    df = derivative(f)
    return f, df
end

"""
    nlml(θ, x_f, y_f, x_df, y_df)

Compute the negative log marginal likelihood of the parameters θ given observations y_f of
f at x_f and observations y_df of df at x_df.
"""
function nlml(
    θ::Tuple{Real, Real},
    x_f::AbstractVector{<:Real},
    y_f::AbstractVector{<:Real},
    x_df::AbstractVector{<:Real},
    y_df::AbstractVector{<:Real},
)
    f, df = build_model(θ)

    # Specify points in processes at which we'll make observations.
    fx = f(x_f, 1e-3)
    dfx = f(x_df, 1e-3)
    return -logpdf([fx, dfx], [y_f, y_df])
end

# Toy problem with synthetic data.
g = sin
dg = cos

x_g = range(-10.0, 10.0; length=15);
x_dg = range(-10.0, 10.0; length=20);

y_g = g.(x_g)
y_dg = dg.(x_dg)

# Optimise the hyper-parameters using a gradient-free method to start with.
θ0 = randn(2);
results = Optim.optimize(θ -> nlml(θ, x_g, y_g, x_dg, y_dg), θ0, NelderMead())
θ_opt  = unpack(results.minimizer);

# Get process at optimal parameters, and produce the posteriors.
f, df = build_model(θ_opt);
f_post, df_post = (f, df) | (f(x_g, 1e-3) ← y_g, df(x_dg, 1e-3) ← y_dg);

# Make posterior predictions and visualise.
x_pr = range(-15.0, 15.0; length=250);

plt = plot();
scatter!(plt, x_g, y_g; color=:red, label="y_g");
scatter!(plt, x_dg, y_dg; color=:blue, label="y_dg");
plot!(plt, f_post(x_pr, 1e-3); label="f_post");
plot!(plt, df_post(x_pr, 1e-3); label="df_post");
display(plt);

# For some reason, this doesn't work. I'm trying to get to the bottom of it at the minute.
# See example in "Getting Started" bit of Stheno docs for how this would be used if it
# worked.
Zygote.gradient(θ -> nlml(θ, x_g, y_g, x_dg, y_dg), θ_opt)
