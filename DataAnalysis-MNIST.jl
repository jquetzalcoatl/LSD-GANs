include("./DCGANs.jl")

parsed_args = parseCommandLine(;dir=1)
PATH_out = "/nobackup/jtoledom/GANs/"
hparams = HyperParams()

# dict = load_dict("/nobackup/jtoledom/GANs/Dicts/0/Dict_2020-09-05T23:12:41.355.bson")
dict = load_dict(PATH_out * "/Dicts/1/" * readdir(PATH_out * "/Dicts/1")[1])

# load(dict[:Plots][end])

# mean_loss_d, mean_loss_g, disc, gen, d_opt, g_opt = train(epochs=500, snap=25)

# dataA = load_data(hparams)

gen = load_model(path_from_dict = dict[:ModelsG][end], model_type = "Gen")
clas = load_model(path_from_dict = dict[:ModelsC][end], model_type = "Clas")

encode = load_model(path_from_dict = dict[:ModelsE][end], model_type = "Enc")

# f1 = myPlots(dataA, 1, 1, gen, 1)
# f2 = myPlots(dataA, 1, gen, encode, 1,1)
#
# savefig(f1, "/nobackup/jtoledom/GANs/Graphs/MNIST/Sample_f1")
# savefig(f2, "/nobackup/jtoledom/GANs/Graphs/MNIST/Sample_f2")
dict[:Path]
# dict

# THIS SECTIONS BUILDS THE EIGENVECTORS IN LATENT SPACE
dataA, dataA_test = load_data(hparams,1)
test_loss(dataA_test, clas)
test_loss(dataA_test, clas, encode, gen)

function sort_vecs(l, supZ, supZ_filled, z, nsamples)
    #=
        Takes the z vectors and stacks them in supZ (10 x nsamples x latent_dim)
        l is a onecold vector one-on-one with z
        supZ_filled coordinates the stacking
    =#
    for j in 1:hparams.batch_size
        i = l[j]
        if supZ_filled[i]<=nsamples
            supZ[i,supZ_filled[i],:] = z[:,j]'
            supZ_filled[i] +=1
        end
    end
end

function build_zup(gen, clas, encode; nsamples=20)
    supZ = zeros(hparams.nclasses, nsamples, hparams.latent_dim)
    supZ_filled = Array{Int,1}(ones(hparams.nclasses))

    for i in 1:size(dataA,1)
        z_gpu, _, _, _ = encode(dataA[i][1]|>gpu)
        z = z_gpu |>cpu
        x_fake_gpu = gen(z_gpu)
        # x_fake_cpu = x_fake_gpu |>cpu
        # heatmap(x_fake_cpu[:,:,1,1])
        l = Flux.onecold(clas(x_fake_gpu)|>cpu)

        sort_vecs(l, supZ, supZ_filled, z, nsamples)
        # if sum(supZ_filled) - 10 < nsamples*hparams.nclasses
        #     @info "yes"
        #     break
        # end
    end
    supZ, supZ_filled
end

function build_zup(gen, clas; nsamples=20)
    supZ = zeros(hparams.nclasses, nsamples, hparams.latent_dim)
    supZ_filled = Array{Int,1}(ones(hparams.nclasses))

    sfty_idx=0
    while sum(supZ_filled) - 10 < nsamples*hparams.nclasses && sfty_idx < 5000
        z = randn(hparams.latent_dim,hparams.batch_size)
        z_gpu = z |> gpu
        x_fake_gpu = gen(z_gpu)
        # x_fake_cpu = x_fake_gpu |>cpu
        # heatmap(x_fake_cpu[:,:,1,1])
        l = Flux.onecold(clas(x_fake_gpu)|>cpu)

        sort_vecs(l, supZ, supZ_filled, z, nsamples)
        sfty_idx +=1
    end
    supZ, supZ_filled
end

function test_backward(arr, gen::Generator; idxIn=0, snap=[5,5], svfig=false,
    filename="plot.png", timestamp=false, modelname="Enc.bson", printplots=true)
    idxIn == 0 ? idx = rand(1:size(supZ,1)) : idx = idxIn

    # zz = CuArray(supZ[idx,:,:]')
    zz = CuArray(arr)
    ŷ = gen(zz) |> cpu
    im_ŷ = reshape(ŷ[:,:,:,1:snap[1]],28,28*snap[1])
    for i in 1:snap[2]-1
        r_ŷ = ŷ[:,:,:,snap[1]*i + 1:snap[1]*i + snap[1]]
        im_ŷ = vcat(im_ŷ, reshape(r_ŷ, 28,28*snap[1]))
    end

    # fig1 = heatmap(im_x)
    fig2 = heatmap(im_ŷ, xaxis=false, yaxis=false, legend=:none,
            imagesize=(700,700), dpi=300, xticks=false,
            c=cgrad(:cividis)) #reds or cividis
    # display(fig2)
    return fig2
end
supZ, sup_fill = build_zup(gen,clas, encode; nsamples=5000)
# plot_hist(supZ, svfig = false)
supZ2, sup_fill = build_zup(gen,clas; nsamples=5000)

fig = test_backward(supZ[3,:,:]', gen; snap=[5,4])

function save_number_samples(supZ, gen; with_enc=true)
    if with_enc
        for i in 1:10
            fig = test_backward(supZ[i,:,:]', gen; snap=[30,40])
            savefig(fig, "/nobackup/jtoledom/GANs/Graphs/Sample_Clas-Enc_N=$(i-1).png")
            display(fig)
        end
    else
        for i in 1:10
            fig = test_backward(supZ[i,:,:]', gen; snap=[30,40])
            savefig(fig, "/nobackup/jtoledom/GANs/Graphs/Sample_Clas_N=$(i-1).png")
            display(fig)
        end
    end
end
# save_number_samples(supZ, gen)
# save_number_samples(supZ2, gen; with_enc=false)

function plot_hist(supZ; filename="Hist_Clas-Enc.png", svfig=false)
    fig = plot(reshape(supZ[1,:,:],:), seriestype=:barhist,
            width=0, normalize=true, fillalpha=0.5, label="0",
        frame=:box, xlabel="latent variable", ylabel="PDF", imagesize=(700,700), dpi=300,
        xlim=(-5,5))
    for i in 2:size(supZ,1)
        fig = plot!(reshape(supZ[i,:,:],:), seriestype=:barhist,
                width=0, normalize=true, fillalpha=0.5, label="$(i-1)")
    end
    fig = plot!(-4:0.01:4, x->1/sqrt(2π)*exp(-x^2/2), lw=5, fillalpha=0.2, label="N(0,1)")
    svfig ? savefig(fig, "/nobackup/jtoledom/GANs/Graphs/" * filename) : nothing
    display(fig)
end

plot_hist(supZ, svfig = false)
plot_hist(supZ2; filename="Hist_Clas.png", svfig = false)

function plot_super_numbers(supZ, gen; filename="SuperNumbers_Clas-Enc.png", svfig=false)
    v1 = gen(mean(supZ[1,:,:] , dims=1)' |>gpu)[:,:,1,1]|>cpu
    for i in 2:Int(size(supZ,1)/2)
        v1 = hcat(v1, gen(mean(supZ[i,:,:] , dims=1)' |>gpu)[:,:,1,1]|>cpu)
    end
    v2 = gen(mean(supZ[Int(size(supZ,1)/2)+1,:,:] , dims=1)' |>gpu)[:,:,1,1]|>cpu
    for i in Int(size(supZ,1)/2)+2:size(supZ,1)
        v2 = hcat(v2, gen(mean(supZ[i,:,:] , dims=1)' |>gpu)[:,:,1,1]|>cpu)
    end
    v = vcat(v1,v2)
    fig = heatmap(v, legend=:none, xaxis=false, yaxis=false, xticks=false,
    c=cgrad(:cividis))
    svfig ? savefig(fig, "/nobackup/jtoledom/GANs/Graphs/" * filename) : nothing
    display(fig)
end
plot_super_numbers(supZ, gen, svfig = false)
plot_super_numbers(supZ2, gen; filename="SuperNumbers_Clas.png", svfig = false)

# heatmap(gen(mean(supZ[10,:,:] , dims=1)' |>gpu)[:,:,1,1]|>cpu)

function plot_vectors(supZ; filename="Vectors_Clas-Enc.png", svfig=false)
    fig = plot([mean(supZ[1,:,i]) for i in 1:size(supZ,3)], ribbon=[std(supZ[1,:,i])
            for i in 1:size(supZ,3)], fillalpha=0.1, size=(500,500), frame=:box, label="0")

    for j in 2:10
        fig = plot!([mean(supZ[j,:,i]) for i in 1:size(supZ,3)], ribbon=[std(supZ[j,:,i])
                for i in 1:size(supZ,3)],fillalpha=0.1, label="$(j-1)")
    end
    svfig ? savefig(fig, "/nobackup/jtoledom/GANs/Graphs/" * filename) : nothing
    display(fig)
end
plot_vectors(supZ)
plot_vectors(supZ2; filename="Vectors_Clas.png")

function plot_3D(supZ; filename="3D_Clas-Enc.png", svfig=false)
    fig = scatter([mean(supZ[1,i,:]) for i in 1:size(supZ,2)], [std(supZ[1,i,:])
            for i in 1:size(supZ,2)], 10 .* ones(size(supZ,2)),
            label="0", xlabel="mean", ylabel="std", size=(700,700), dpi=200)

    for j in 2:10
        fig = scatter!([mean(supZ[j,i,:]) for i in 1:size(supZ,2)], [std(supZ[j,i,:])
                for i in 1:size(supZ,2)], (11 - j) .* ones(size(supZ,2)), label="$(j-1)")
    end
    svfig ? savefig(fig, "/nobackup/jtoledom/GANs/Graphs/" * filename) : nothing
    display(fig)
end
plot_3D(supZ)
plot_3D(supZ2; filename="3D_Clas.png")

###G-S
proj(u,v) = u' * v/(u' * u) * u

function genUs(v) #GramSchmidt
    u = zeros(size(v,1),100)
    # u[1,:] = v[1,:]
    for i in 1:size(v,1)
        try
            u[i,:] = v[i,:] - sum(proj(u[j,:],v[i,:]) for j in 1:i-1)
        catch ex
            @info i
            u[i,:] = v[i,:]
        end
    end
    u = u/mean(std(u, dims=2))
    Array(u')
end

function genUs(v,modified)
    u = zeros(size(v,1),100)
    u[1,:] = v[1,:]/sqrt(v[1,:]' * v[1,:])
    for i in 2:size(v,1)
        u[i,:] = v[i,:]
        for j in 1:i-1
            u[i,:] = u[i,:] - proj(u[j,:],u[i,:])
        end
        u[i,:] = u[i,:]/sqrt(u[i,:]' * u[i,:])
    end
    u = u/mean(std(u, dims=2))
    Array(u')
end

function genVs(s1, s2)
    supZ, sup_fill = build_zup(gen,clas, encode; nsamples=s1)
    v = mean(supZ , dims=2)[:,1,:]
    for i in 1:9
        supZ, sup_fill = build_zup(gen,clas; nsamples=s2)
        v = vcat(v, mean(supZ , dims=2)[:,1,:])
    end
    v
end
v = genVs(5000, 5000)
u = genUs(v,1)
###
begin
    # using Plots
    default(titlefont = (20, "Helvetica"), legendfontsize = 18, guidefont = (15, "Helvetica"),
    tickfont = (15, Helvetica))

    heatmap(v * v', c=cgrad(:magma), xlabel="| η ⟩", ylabel="| η ⟩",
        tickfont = font(15, "Helvetica"), size=(500,500))
end
default(titlefont = (20, "Helvetica"), legendfontsize = 18, guidefont = (15, "Helvetica"),
tickfont = (15, Helvetica))
####

h1 = heatmap(v * v', c=cgrad(:magma), xlabel="| η ⟩", ylabel="| η ⟩",
    tickfont = font(15, "Helvetica"), guidefont = (15), size=(500,500),
    left_margin = 5Plots.mm, bottom_margin = 2Plots.mm,
    right_margin = 10Plots.mm, top_margin = 5Plots.mm)
# savefig(h1, "/nobackup/jtoledom/GANs/Graphs/Vs-heatmap.png")
h2 = heatmap(u' * u, c=cgrad(:magma), xlabel="| ξ ⟩", ylabel="| ξ ⟩",
    tickfont = font(15, "Helvetica"), guidefont = (15), size=(500,500),
    left_margin = 5Plots.mm, bottom_margin = 2Plots.mm,
    right_margin = 10Plots.mm, top_margin = 5Plots.mm)
# savefig(h2, "/nobackup/jtoledom/GANs/Graphs/Us-heatmap.png")

# plot([u[:,i]' * u[:,i] for i in 1:100])

# heatmap([(mean(supZ[i,:,:] , dims=1) * mean(supZ[j,:,:] , dims=1)')[1]/150 for i in 1:10, j in 1:10])
# std(supZ[1,:,:])
test_backward(v', gen; snap=[10,10])
# savefig(test_backward(v', gen; snap=[10,10]), "/nobackup/jtoledom/GANs/Graphs/Vs-numbers.png")
test_backward(u, gen; snap=[10,10])
# savefig(test_backward(u, gen; snap=[10,10]), "/nobackup/jtoledom/GANs/Graphs/Us-numbers.png")

# uproj = gen(u|>gpu)
# heatmap(uproj[:,:,1,20]|>cpu)
# clas(uproj)
# uclas = Flux.onecold(cpu(clas(uproj)))
# fig = plot(uclas .- 1, frame=:box, ylabel="Label", xlabel="| ξ ⟩", label="G Truth",
#         lw=2, size=(700,500), markershapes = [:circle], markerstrokewidth=0, ms=10,
#         legend=:none, legendfontsize = 15, ylim=(-1, 11),
#         tickfont = font(15, "Helvetica"), guidefont = (15), yticks=([i for i in 0:9], [i for i in 0:9]))
# savefig(fig, "/nobackup/jtoledom/GANs/Graphs/Us-numbers-class.png")

#HERE WE TEST THE BASIS AS A CLASSIFIER
length( dataA_test)
idx = 1

z, μ, logσ, y = encode(dataA_test[idx][1]|>gpu) |> cpu
plot(((z[:,14]' * u)')[1:10])
# u
# dataA_test[1][2]
heatmap(dataA_test[idx][1][:,:,1,25])
append!([:red for i in 1:25],[:blue])

res_exp = Flux.onecold(dataA_test[idx][2])
l=Array{Int64,2}(collect(1:100)')
res = ones(Int, 25)
for i in 1:25
    A = abs.((z[:,i]' * u))
    res[i] = l[findall(x->x==maximum(A),A)][1]
end
res
sum(sign.(abs.(res-res_exp)))
fig = plot(res_exp .- 1, frame=:box, ylabel="Label", xlabel="Batch Element", label="G Truth",
        lw=2, size=(700,500), markershapes = [:circle], markerstrokewidth=0, ms=20,
        legend=:topleft, legendfontsize = 15, ylim=(-1, 11),
        tickfont = font(15, "Helvetica"), guidefont = (15), yticks=([i for i in 0:9], [i for i in 0:9]))
fig = scatter!(res .- 1, label="maxᵢ |cᵢ|", markerstrokewidth=0, ms=10,
        mc=[res[i] == res_exp[i] ? :lightgreen : :red for i in 1:25])
savefig(fig, "/nobackup/jtoledom/GANs/Graphs/true_labels_amplitudes_comparisons.png")
# sum(sign.(abs.(res-res_exp)))
# 2/25
# size(dataA_test[400][1])
# 400*25
function compute_error(dataA; u=u)
    l=Array{Int64,2}(collect(1:100)')
    wrong_answers = 0
    gibberish = 0
    gibberish_list = [0,0]
    positions_of_correct_answers = []
    δ = []
    labels_of_wrong = []
    ii = 0
    for (x,y) in dataA
        z, μ, logσ, _ = encode(x|>gpu) |> cpu

        res_exp = Flux.onecold(y|>cpu)
        res = ones(Int, size(z,2))
        for i in 1:size(z,2)
            A = abs.((z[:,i]' * u))
            res[i] = l[findall(x->x==maximum(A),A)][1]
            A_sorted = sort(A, dims=2, rev=true)[1,:]
            append!(positions_of_correct_answers, findall(x->x==A[res_exp[i]], A_sorted))
            append!(δ, maximum(A) - A[res_exp[i]])
        end
        gibberish = gibberish + sum(res .>10)

        # append!(gibberish_list, [(res[i], res_exp[i]) .* (res[i]>10) for i in 1:25])
        for i in 1:size(y,2)
            if res[i] > 10
                gibberish_list = hcat(gibberish_list, [res[i], res_exp[i]])
            end
        end

        wrong_answers = wrong_answers + sum(sign.(abs.(res-res_exp)))
        append!(labels_of_wrong, sign.(abs.(res-res_exp)) .* res_exp)
    end
    wrong_answers, gibberish, positions_of_correct_answers, δ, labels_of_wrong, Array(gibberish_list[:,2:end]')
end
wrong_answers, gibberish, positions_of_correct_answers, δ, labels_of_wrong, gib_list = compute_error(dataA_test)
1 - wrong_answers/10000
plot(positions_of_correct_answers, yrange=(0,100))

accM = [test_loss(dataA_test, clas) for i in 1:20]
acc1 = [test_loss(dataA_test, clas, encode, gen) for i in 1:20]
acc2 = [1 - compute_error(dataA_test)[1]/10000 for i in 1:20]
mean(acc1)
mean(acc2)
f1 = plot([acc1 acc2], lw=3, label=["C ⋅ G ⋅ E|x⟩" "maxᵢ |⟨ξᵢ|E|x⟩|" "C |x⟩"], frame=:box,
    xlabel="trials", ylabel="Accuracy", legend=(.7,.75), legendfontsize = 15,
    tickfont = font(15, "Helvetica"), guidefont = (15), size=(700,500),
    markershapes = [:circle], markerstrokewidth=0, ms=10, ylim=(0.91,1.0))
f1 = plot!(accM, label="C |x⟩", lw=6)
savefig(f1,
    "/nobackup/jtoledom/GANs/Graphs/MNIST/Accuracy_comparison.png")

f2 = plot(labels_of_wrong[findall(x->x!= 0, labels_of_wrong)] .- 1, seriestype=:barhist,
        width=0, bins=10, legend=:topleft, normalize=true, xlabel="Label", ylabel="PDF", label="Wrong Predictions", frame=:box,
        size=(700,500), tickfont = font(15, "Helvetica"), legendfontsize = 15,
        xticks=([i + 0.5 for i in 0:9], [i for i in 0:9]), guidefont = (15))
savefig(f2,
        "/nobackup/jtoledom/GANs/Graphs/MNIST/Hist_error_vs_labels.png")

histogram(vcat([Flux.onecold(dataA_test[i][2]) for i in 1:400]...) .- 1, bins=10,
        normalize=true, xlabel="number", ylabel="PDF", label="Encoded samples", frame=:box,
        size=(500,400))
savefig(histogram(vcat([Flux.onecold(dataA_test[i][2]) for i in 1:400]...) .- 1, bins=10,
        normalize=true, xlabel="number", ylabel="PDF", label="Encoded samples", frame=:box,
        size=(500,400)),
        "/nobackup/jtoledom/GANs/Graphs/MNIST/Hist_vs_labels.png")

scatter(sort(positions_of_correct_answers, rev=true), xscale=:log10,
    frame=:box, size=(700,500), ms=7, xlabel="Rank", ylabel="abs error", legend=:none)
savefig(scatter(sort(positions_of_correct_answers, rev=true), xscale=:log10,
    frame=:box, size=(700,500), ms=7, xlabel="Rank", ylabel="abs error", legend=:none),
    "/nobackup/jtoledom/GANs/Graphs/error_rank.png")

histogram(positions_of_correct_answers, normalize=true)
mean(positions_of_correct_answers)
std(positions_of_correct_answers)
accM[1]
sum(positions_of_correct_answers .== 1)
sum(positions_of_correct_answers .== 2)
sum(positions_of_correct_answers .== 3)
f1 = plot!(1:100, x->accM[1], label="C |x⟩", lw=3, ls=:dash)
f1 = plot([sum([sum(positions_of_correct_answers .== i) for i in 1:j]) for j in 1:100]/10000,
        legend=:none, xscale=:log10, c=:purple, frame=:box, lw=2, xlabel="True label projection position",
        ylabel="Cumulative", size=(700,500), tickfont = font(15, "Helvetica"), legendfontsize = 15,
        xticks=([i for i in 0:9], [i for i in 0:9]), guidefont = (15), markershapes = [:circle],
        markerstrokewidth=0, ms=10)
savefig(f1,
        "/nobackup/jtoledom/GANs/Graphs/error_cumulative.png")

801*25-18036
18036+1965
(18036+1379+345)/20001
plot(δ[positions_of_correct_answers .== 2])
plot(δ)
scatter(sort(δ, rev=true))
scatter!(sort(positions_of_correct_answers, rev=true),
    frame=:box, size=(700,500), ms=7, xlabel="Rank", ylabel="abs error", legend=:none)
s
# PCA
# Operators
idx
gib_list
gib_list[gib_list[:,1] .< 20,:]
plot(gib_list)
function coeffAmpAnalysis(; lim = 0)
    l=Array{Int64,2}(collect(1:100)')
    positions_of_correct_answers = []
    ii = 0
    coeff_sorted = zeros(25,100)
    A, A_sorted = zeros(1,100), zeros(1,100)
    res, res_exp = zeros(25), zeros(25)
    for (x,y) in dataA_test
        z, μ, logσ, _ = encode(x|>gpu) |> cpu

        res_exp = Flux.onecold(y|>cpu)
        res = ones(Int, size(z,2))
        coeff = z' * u


        maximum(abs.(coeff), dims=2)
        if ii == 0
            coeff_sorted = sort(abs.(coeff),dims=2, rev=true) ./ maximum(abs.(coeff), dims=2)
        else
            coeff_sorted = vcat(coeff_sorted,
                sort(abs.(coeff),dims=2, rev=true) ./ maximum(abs.(coeff), dims=2))
        end
        for i in 1:size(z,2)
            A = abs.((z[:,i]' * u))
            res[i] = l[findall(x->x==maximum(A),A)][1]
            A_sorted = sort(A, dims=2, rev=true)[1,:]
            append!(positions_of_correct_answers, findall(x->x==A[res_exp[i]], A_sorted))
        end
        ii = ii + 1
        if ii > lim
            break
        end
    end
    res, res_exp, positions_of_correct_answers, coeff_sorted, A, A_sorted
end
res, res_exp, positions_of_correct_answers, coeff_sorted, A, A_sorted = coeffAmpAnalysis(lim=100)
positions_of_correct_answers
heatmap(dataA_test[1][1][:,:,1,1])
res_exp .- res
res[1]
res_exp[1]

g1 = plot(coeff_sorted[positions_of_correct_answers .== 1,1:10]', frame=:box,
        legend=:none, xlabel=:Rank, ylabel="Normalized Amplitude", c=:lightgray,
        markershapes = [:circle], lw=1, ms=10, markerstrokewidth=0, mc=:purple,
        size=(800,600), tickfont = font(15, "Helvetica"), legendfont = font(15, "Helvetica"),
        guidefont = (20))
savefig(g1, "/nobackup/jtoledom/GANs/Graphs/Amp_rank_1.png")
g1 = plot(coeff_sorted[positions_of_correct_answers .== 2,1:10]', frame=:box,
        legend=:none, xlabel=:Rank, ylabel="Normalized Amplitude", c=:lightgray,
        markershapes = [:circle], lw=1, ms=10, markerstrokewidth=0, mc=:purple,
        size=(800,600), tickfont = font(15, "Helvetica"), legendfont = font(15, "Helvetica"),
        guidefont = (20))
savefig(g1, "/nobackup/jtoledom/GANs/Graphs/Amp_rank_2.png")
g1 = plot(coeff_sorted[positions_of_correct_answers .== 3,1:10]', frame=:box,
        legend=:none, xlabel=:Rank, ylabel="Normalized Amplitude", c=:lightgray,
        markershapes = [:circle], lw=1, ms=10, markerstrokewidth=0, mc=:purple,
        size=(800,600), tickfont = font(15, "Helvetica"), legendfont = font(15, "Helvetica"),
        guidefont = (20))
savefig(g1, "/nobackup/jtoledom/GANs/Graphs/Amp_rank_3.png")
# scatter(coeff_sorted[positions_of_correct_answers .== 2,1:10]', frame=:box, legend=:none, xlabel=:Rank, ylabel=:Coefficients)
# scatter(coeff_sorted[positions_of_correct_answers .== 3,1:10]', frame=:box, legend=:none, xlabel=:Rank, ylabel=:Coefficients)

coeff_sorted[positions_of_correct_answers .== 1,2]
coeff_sorted[positions_of_correct_answers .== 2,2]
coeff_sorted[positions_of_correct_answers .== 3,2]
g1 = plot(coeff_sorted[positions_of_correct_answers .== 1,2], seriestype=:barhist,
        width=0, normalize=true, opacity=0.7, frame=:box, legend=:topright,
        xlabel="Normalized Amplitude", ylabel="PDF", label="2nd largest amplitude",
        size=(800,600), tickfont = font(15, "Helvetica"), legendfont = font(20, "Helvetica"),
        guidefont = (22))
g1 = plot!(coeff_sorted[positions_of_correct_answers .== 1,3], seriestype=:barhist,
        width=0, normalize=true, opacity=0.7, label="3rd largest amplitude")
g1 = plot!(coeff_sorted[positions_of_correct_answers .== 1,4], seriestype=:barhist,
        width=0, normalize=true, opacity=0.7, label="4th largest amplitude")
savefig(g1, "/nobackup/jtoledom/GANs/Graphs/Amp_hist_1.png")

g1 = plot(coeff_sorted[positions_of_correct_answers .== 2,2], seriestype=:barhist,
        width=0, normalize=true, opacity=0.7, frame=:box, legend=:topleft,
        xlabel="Normalized Amplitude", ylabel="PDF", label="2nd largest amplitude",
        size=(700,500), tickfont = font(15, "Helvetica"), legendfont = font(20, "Helvetica",:lightgreen),
        guidefont = (22))
g1 = plot!(coeff_sorted[positions_of_correct_answers .== 2,3], seriestype=:barhist,
        width=0, normalize=true, opacity=0.7, label="3rd largest amplitude")
g1 = plot!(coeff_sorted[positions_of_correct_answers .== 2,4], seriestype=:barhist,
        width=0, normalize=true, opacity=0.7, label="4th largest amplitude")
savefig(g1, "/nobackup/jtoledom/GANs/Graphs/Amp_hist_2.png")

g1 = plot(coeff_sorted[positions_of_correct_answers .== 3,2], seriestype=:barhist,
        width=0, normalize=true, opacity=0.7, frame=:box, legend=:topleft,
        xlabel="Normalized Amplitude", ylabel="PDF", label="2nd largest amplitude",
        size=(700,500), tickfont = font(15, "Helvetica"), legendfont = font(20, "Helvetica"),
        guidefont = (22))
g1 = plot!(coeff_sorted[positions_of_correct_answers .== 3,3], seriestype=:barhist,
        width=0, normalize=true, opacity=0.7, label="3rd largest amplitude")
g1 = plot!(coeff_sorted[positions_of_correct_answers .== 3,4], seriestype=:barhist,
        width=0, normalize=true, opacity=0.7, label="4rd largest amplitude")
savefig(g1, "/nobackup/jtoledom/GANs/Graphs/Amp_hist_3.png")

#HERE WE TRY THE BASIS AS A DENOISER
idx = 2
z, μ, logσ, y = encode(dataA_test[idx][1]|>gpu) |> cpu
coeff = z' * u

# coeff[1,:]
# maximum(abs.(coeff), dims=2)
# coeff_sorted = sort(abs.(coeff),dims=2, rev=true) ./ maximum(abs.(coeff), dims=2)
# scatter(coeff_sorted', frame=:box, legend=:none, xlabel=:Rank, ylabel=:Coefficients)
# savefig(scatter(coeff_sorted', frame=:box, legend=:none, xlabel=:Rank, ylabel=:Coefficients),
#         "/nobackup/jtoledom/GANs/Graphs/coeff.png")

find_projection(vec; idx=1) = findall(x->x==sort(vec,rev=true)[idx], vec)[1]
# sum(coeff,dims=2)
# find_projection(abs.(coeff[1,:]))

function denoised_vec(n, coeff; num_proj=1, u=u)
    sum([coeff[n,find_projection(abs.(coeff[n,:]);idx=i)] *
        u[:,find_projection(abs.(coeff[n,:]);idx=i)]/(u[:,find_projection(abs.(coeff[n,:]);idx=i)]'
        *u[:,find_projection(abs.(coeff[n,:]);idx=i)]) for i in 1:num_proj])
end

# z[:,1]

sum(denoised_vec(1, coeff; num_proj=100) - z[:,1])
begin
    idx = 300
    z, μ, logσ, y = encode(dataA_test[idx][1]|>gpu) |> cpu
    coeff = z' * u
    znew = zeros(100,25)
    znew2 = zeros(100,25)
    znew3 = zeros(100,25)
    znew4 = zeros(100,25)
    znew10 = zeros(100,25)
    for i in 1:25
        global znew[:,i] = denoised_vec(i, coeff; num_proj=1)
        global znew2[:,i] = denoised_vec(i, coeff; num_proj=2)
        global znew3[:,i] = denoised_vec(i, coeff; num_proj=3)
        global znew4[:,i] = denoised_vec(i, coeff; num_proj=4)
        global znew10[:,i] = denoised_vec(i, coeff; num_proj=10)
    end
    y_proj = gen(znew|>gpu) |>cpu
    y_proj2 = gen(znew2|>gpu) |>cpu
    y_proj3 = gen(znew3|>gpu) |>cpu
    y_proj4 = gen(znew4|>gpu) |>cpu
    y_proj10 = gen(znew10|>gpu) |>cpu
    y = gen(z|>gpu) |>cpu
    id=1

    # heatmap(vcat(y[:,:,1,id], dataA_test[idx][1][:,:,1,id], y_proj[:,:,1,id]), size=(10*28,10*28*3))

    ydenoised_arr = vcat(dataA_test[idx][1][:,:,1,1], y[:,:,1,1], y_proj[:,:,1,1],
                    y_proj2[:,:,1,1], y_proj3[:,:,1,1], y_proj4[:,:,1,1],
                    y_proj10[:,:,1,1])
    for id in 2:25
        global ydenoised_arr =hcat(ydenoised_arr, vcat(dataA_test[idx][1][:,:,1,id], y[:,:,1,id],
                            y_proj[:,:,1,id], y_proj2[:,:,1,id], y_proj3[:,:,1,id],
                            y_proj4[:,:,1,id], y_proj10[:,:,1,id]))
    end
end
# ydenoised_arr
heatmap(ydenoised_arr, size=(10*25*28,10*28*7), xaxis=false,
    yaxis=false, legend=:none, c=cgrad(:cividis))
savefig(heatmap(ydenoised_arr, size=(10*25*28,10*28*7), xaxis=false,
    yaxis=false, legend=:none, c=cgrad(:cividis)), "/nobackup/jtoledom/GANs/Graphs/MNIST/denoiser5.png")


#EVOLUTION (In construction.............)
Flux.onecold(dataA_test[1][2])
x = dataA_test[1][1][:,:,:,findall(x->x==1, Flux.onecold(dataA_test[1][2]) )]
for i in 1:100
    Flux.onecold(dataA_test[i][2])
    global x = Flux.cat(x, dataA_test[i][1][:,:,:,findall(x->x==1, Flux.onecold(dataA_test[i][2]) )], dims=4)
end

z, μ, logσ, _  = encode(x|>gpu) |> cpu

T10 = u[:,2] * u[:,1]'


Pz = T10 * z #|> gpu
Pzz = Pz ./ (mean(Pz, dims=1) .- 120.5*randn(1,size(Pz,2)))
Pzz = (Pz .- mean(Pz,dims=1))./std(Pz, dims=1)
y = gen(Pzz |>gpu) |> cpu



heatmap(y[:,:,1,3])
heatmap(x[:,:,1,1])


np = 10
Δn = 0.01
a = [1.0 - Δn*i for i in 1:Int(1/Δn)]
b = [Δn*i for i in 1:Int(1/Δn)]

myMat = zeros((np-1)*size(a,1)+1, np) .+ 0.0 .* randn((np-1)*size(a,1)+1, np)
myMat[1,1] = 1.0
for i in 1:size(myMat,2)-1
    global myMat[2 + (i-1)*size(a,1):i*size(a,1)+1,i] = a
    global myMat[2 + (i-1)*size(a,1):i*size(a,1)+1,i+1] = b
end
myMat
heatmap(myMat, legend=:none, xaxis=false, yaxis=false)
savefig(heatmap(myMat, legend=:none, xaxis=false, yaxis=false), "/nobackup/jtoledom/GANs/Graphs/MNIST/rot_mat.png")
# rot = myMat * [u[:,i] for i in 10:-1:1]
rot = myMat * [u[:,i] for i in 1:10]

rot[1] * rot[1]'
size(rot,1)
m=zeros(size(rot,1))
s=zeros(size(rot,1))
m2=zeros(size(rot,1))
s2=zeros(size(rot,1))
begin
    Pz = z
    rots = [x[:,:,1,i] for i in 1:10]

    for i in 1:size(rot,1)-1
        T = rot[i+1] .* rot[i]'
        Pz = T * Pz |> gpu
        y = gen(Pz) |> cpu
        Pz = Pz |>cpu
        # Pz = Pz ./ (std(Pz, dims=1) .+ 50.5*randn(1,size(Pz,2)))
        Pz = (Pz .- mean(Pz,dims=1))./(std(Pz, dims=1) )
        # Pz = Pz ./ mean(std(Pz, dims=1) )
        # Pz = Pz ./ std(Pz, dims=1)
        # m[i] = mean(y)
        # s[i] = std(y)
        # m2[i] = mean(abs.(reshape(mean(y, dims=(1,2)),:)))
        # s2[i] = mean(abs.(reshape(std(y, dims=(1,2)),:)))
        if i % 25 == 0
            rots = hcat(rots, [y[:,:,1,i] for i in 1:10])
        end
    end
end
# Pz
# std(Pz, dims=1)
# heatmap(hcat([y[:,:,1,i] for i in 1:10]...), size=(1300,250), xaxis=false, legend=:none)


# heatmap(hcat(rots[:,20]...), size=(1300,250), xaxis=false, legend=:none)

# heatmap(vcat(hcat(rots[:,20]...),hcat(rots[:,20]...)))
heatmap(vcat([hcat(rots[:,i]...) for i in 1:size(rots,2)]...), size=(1400,4150),
    xaxis=false, legend=:none, c=cgrad(:cividis))
savefig(heatmap(vcat([hcat(rots[:,i]...) for i in 1:size(rots,2)]...), size=(1400,4150),
    xaxis=false, legend=:none, c=cgrad(:cividis)), "/nobackup/jtoledom/GANs/Graphs/MNIST/rotations.png")

size(rots,2)


np = 10
Δn = 0.01
# a = [1.0 - Δn*i for i in 1:Int(1/Δn)]
a = [cospi(1/2 * Δn*i) for i in 1:Int(1/Δn)]
b = [sinpi(1/2 * Δn*i) for i in 1:Int(1/Δn)]

myMat = zeros((np-1)*size(a,1)+1, np) .+ 0.0 .* randn((np-1)*size(a,1)+1, np)
myMat[1,1] = 1.0
for i in 1:size(myMat,2)-1
    global myMat[2 + (i-1)*size(a,1):i*size(a,1)+1,i] = a
    global myMat[2 + (i-1)*size(a,1):i*size(a,1)+1,i+1] = b
end
# myMat
heatmap(myMat, legend=:none, xaxis=false, yaxis=false)
# savefig(heatmap(myMat, legend=:none, xaxis=false, yaxis=false), "/nobackup/jtoledom/GANs/Graphs/MNIST/rot_mat.png")
# rot = myMat * [u[:,i] for i in 10:-1:1]
rot = myMat * [u[:,i] for i in 1:10]

rot[1] * rot[1]'
size(rot,1)
# m=zeros(size(rot,1))
# s=zeros(size(rot,1))
# m2=zeros(size(rot,1))
# s2=zeros(size(rot,1))
rot[1+1] .* rot[1]' * u[:,1] / (u[:,1]' * u[:,1])
rot[1+1]

begin
    Pz = z
    rots = [x[:,:,1,i] for i in 1:10]

    for i in 1:size(rot,1)-1
        T = rot[i+1] .* rot[i]'
        # Pz = T * Pz |> gpu
        Pz = T * Pz |> gpu
        y = gen(Pz) |> cpu
        Pz = Pz |>cpu
        # Pz = Pz ./ (std(Pz, dims=1) .+ 50.5*randn(1,size(Pz,2)))
        # Pz = (Pz .- mean(Pz,dims=1))./(std(Pz, dims=1) )
        # Pz = Pz ./ mean(std(Pz, dims=1) )
        Pz = Pz ./ std(Pz, dims=1)
        # Pz = Pz .* 10 / sqrt((Pz' * Pz))
        # m[i] = mean(y)
        # s[i] = std(y)
        # m2[i] = mean(abs.(reshape(mean(y, dims=(1,2)),:)))
        # s2[i] = mean(abs.(reshape(std(y, dims=(1,2)),:)))
        if i % 25 == 0
            rots = hcat(rots, [y[:,:,1,i] for i in 1:10])
        end
    end
end

heatmap(vcat([hcat(rots[:,i]...) for i in 1:size(rots,2)]...), size=(1400,4150),
    xaxis=false, legend=:none, c=cgrad(:cividis))

mean(Pz,dims=1)

ss = randn(500)

mean(ss)

std(ss)

sqrt((ss' * ss)/size(ss,1))

####
#Convergence
####

fig = test_backward(supZ2[3,:,:]', gen; snap=[5,4])

begin
    numIdx = 2
    idxMax=100
    samples=500
    plt1 = plot([mean.([supZ2[numIdx,1:l,idx] for l in 2:samples]) for idx in 1:idxMax], lw=2, ribbon=[std.([supZ2[numIdx,1:l,idx] for l in 2:samples])
            for idx in 1:idxMax], fillalpha=0.05, legend=false)
    plt2 = plot(sort([mean(supZ2[numIdx,:,idx]) for idx in 1:idxMax]))
    plt3 = plot([mean(supZ2[numIdx,:,idx]) for idx in 1:idxMax], [std(supZ2[numIdx,:,idx]) for idx in 1:idxMax],
        markershapes = [:circle], markerstrokewidth=0, lw=0, xlabel="mean", ylabel="std")
    plot(plt1,plt2,plt3)
end

begin
    numIdx = 2
    idxMax=100
    samples=500
    plt1 = plot([mean.([supZ[numIdx,1:l,idx] for l in 2:samples]) for idx in 1:idxMax], lw=2, ribbon=[std.([supZ[numIdx,1:l,idx] for l in 2:samples])
            for idx in 1:idxMax], fillalpha=0.05, legend=false)
    plt2 = plot(sort([mean(supZ[numIdx,:,idx]) for idx in 1:idxMax]), c=:red, ribbon=[std(supZ[numIdx,:,idx]) for idx in 1:idxMax])
    plt3 = plot([mean(supZ[numIdx,:,idx]) for idx in 1:idxMax], [std(supZ[numIdx,:,idx]) for idx in 1:idxMax],
        markershapes = [:circle], markerstrokewidth=0, lw=0, xlabel="mean", ylabel="std")
    plot(plt1,plt2, plt3)
end

plot([mean(supZ[1,:,idx]) for idx in 1:idxMax], [std(supZ[1,:,idx]) for idx in 1:idxMax],
    markershapes = [:circle], markerstrokewidth=0, lw=0, xlabel="mean", ylabel="std")
plot!([mean(supZ[2,:,idx]) for idx in 1:idxMax], [std(supZ[2,:,idx]) for idx in 1:idxMax],
    markershapes = [:circle], markerstrokewidth=0, lw=0, xlabel="mean", ylabel="std")
# plot([mean(supZ2[3,:,idx]) for idx in 1:idxMax])
# plot([std(supZ2[3,:,idx]) for idx in 1:idxMax])
