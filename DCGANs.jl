using Flux, CUDA, Dates
using Plots, ArgParse
using Parameters: @with_kw
using DelimitedFiles
using MLDatasets, Images, Statistics, Random
using BSON: @load, @save

CUDA.allowscalar(false)

@with_kw struct HyperParams
    nclasses::Int=10
    batch_size::Int=25
    latent_dim::Int=100
    lr_dscr::Float64 = 0.0002
    lr_gen::Float64 = 0.0002
    output_x::Int = 6        # No. of sample images to concatenate along x-axis
    output_y::Int = 6        # No. of sample images to concatenate along y-axis
    enc_l::Int = 6272       # 8192 for CIFAR
    enc_c::Int = 128
    im_size::Int=28   #im_size=32 for CIFAR (gen)
    im_chn::Int=1       # im_chn=3 for CIFAR
    clas_num::Int=288   #512 for CIFAR
end

function parseCommandLine(; dir=2)

        # initialize the settings (the description is for the help screen)
        s = ArgParseSettings(description = "Example 1 for argparse.jl: minimal usage.")

        @add_arg_table! s begin
           "--opt"               # an option (will take an argument)
                help = "Optmizer 1 = Momentum, 0 = ADAM"
                arg_type = Int
                default = 1
           "--weight", "-w"         # another option, with short form
                help = "weight"
                arg_type = Float32
                default = 50f0
           "--n"
                    help = "dir name"
                    default = dir
           "--e"
                help = "epochs"
                default = 500
           "--snap"
                help = "snapshots"
                default = 25
            "arg1"                 # a positional argument
                arg_type=Float32
                default = 0.2f0
            "arg2"
                arg_type=Float32
                default = 0.2f0
            "arg3"
                arg_type=Float32
                default = 0.2f0
            "arg4"
                arg_type=Float32
                default = 0.4f0
            "arg5"
                arg_type=Float32
                default = 20f0
            # "arg6"
                # arg_type=Int=
        end

        return parse_args(s) # the result is a Dict{String,Any}
end


function init_dict(args)
    dict_path = PATH_out * "Dicts"
    isdir(dict_path) || mkdir(dict_path)
    dict_path = PATH_out * "Dicts/$(parsed_args["n"])"
    isdir(dict_path) || mkdir(dict_path)
    dict = Dict()
    dict[:Path] = dict_path * "/Dict_$(now()).bson"
    # dict[:Opt] = (parsed_args["opt"] == 1 ? "Momentum" : "ADAM")
    # dict[:Dropout_Params] = [args.Drop1, args.Drop2, args.Drop3, args.Drop4]
    dict[:Plots] = Array{String,1}([])
    dict[:ModelsG] = Array{String,1}([])
    dict[:ModelsE] = Array{String,1}([])
    dict[:ModelsC] = Array{String,1}([])

    dict[:Params] = Dict(:nclasses=>args.nclasses, :batch_size=>args.batch_size,
            :latent_dim=>args.latent_dim,
            :lr_dscr=>args.lr_dscr, :lr_gen=>args.lr_gen, :output_x=>args.output_x,
            :output_y=>args.output_y, :enc_l=>args.enc_l, :enc_c=>args.enc_c)

    @save dict[:Path] dict
    return dict
end

function load_dict(path)
    @load path dict
    dict
end

function save_model(model, opt ; filename = "Gen.bson", model_type = "Gen", dataset="MNIST")
    if model_type == "Gen"
        isdir(PATH_out * "Models") || mkdir(PATH_out * "Models")
        path = PATH_out * "Models/" * filename
        isdir(PATH_out * "Models/$(parsed_args["n"])") || mkdir(PATH_out * "Models/$(parsed_args["n"])")
        path = PATH_out * "Models/$(parsed_args["n"])/" * filename
        g_cpu = Generator(cpu(model.g_common))
        push!(dict[:ModelsG], path)
        @save string(path) g_cpu #opt
    elseif model_type == "Enc"
        isdir(PATH_out * "Models") || mkdir(PATH_out * "Models")
        path = PATH_out * "Models/" * filename
        isdir(PATH_out * "Models/$(parsed_args["n"])") || mkdir(PATH_out * "Models/$(parsed_args["n"])")
        path = PATH_out * "Models/$(parsed_args["n"])/" * filename
        enc_cpu = Encoder(cpu(model.enc), cpu(model.toLatentμ), cpu(model.toLatentσ), cpu(model.toLabel))
        push!(dict[:ModelsE], path)
        @save string(path) enc_cpu #opt
    elseif model_type == "Clas"
        isdir(PATH_out * "Models") || mkdir(PATH_out * "Models")
        path = PATH_out * "Models/" * filename
        isdir(PATH_out * "Models/$(parsed_args["n"])") || mkdir(PATH_out * "Models/$(parsed_args["n"])")
        path = PATH_out * "Models/$(parsed_args["n"])/" * filename
        clas_cpu = Classifier(cpu(model.c_common))
        push!(dict[:ModelsC], path)
        @save string(path) clas_cpu #opt
    end
end

function load_model(; path_from_dict = "No path", model_type = "Gen", dataset="MNIST")
    if model_type == "Gen"
        path_from_dict == "No path" ? path = dict[:ModelsG][end] :
            (path = path_from_dict; @info "Loading Model $path")
        model = genGen(hparams)
        g_cpu = Generator(cpu(model.g_common))
        opt = ADAM(0.0001)
        @load string(path) g_cpu #opt
        model = Generator(gpu(g_cpu.g_common))
    elseif model_type == "Enc"
        path_from_dict == "No path" ? path = dict[:ModelsE][end] :
            (path = path_from_dict; @info "Loading Model $path")
        if dataset == "ISING"
            str = dataset
        else
            str = "MNIST"
        end
        model = encoder(hparams; model=str)
        enc_cpu = Encoder(cpu(model.enc), cpu(model.toLatentμ), cpu(model.toLatentσ), cpu(model.toLabel))
        opt = ADAM(0.0001)
        @load string(path) enc_cpu# opt
        model = Encoder(gpu(enc_cpu.enc), gpu(enc_cpu.toLatentμ), gpu(enc_cpu.toLatentσ), gpu(enc_cpu.toLabel))
    elseif model_type == "Clas"
        path_from_dict == "No path" ? path = dict[:ModelsC][end] :
            (path = path_from_dict; @info "Loading Model $path")
        if dataset == "MNIST"
            model = genClassifier(hparams)
        elseif dataset == "CIFAR"
            model = Baby_ResNet(hparams)
        elseif dataset == "ISING"
            model = genClassifier(hparams)
        end
        clas_cpu = Classifier(cpu(model.c_common))
        opt = ADAM(0.0001)
        @load string(path) clas_cpu# opt
        model = Classifier(gpu(clas_cpu.c_common))
    end
    return model#, opt
end

function add_ising_data_NO_Rdm()
    dict_path = "/nobackup/jtoledom/Ising/Dicts/"
    dict_file = readdir(dict_path)[1]
    dict = load_dict(dict_path * dict_file)
    sp = readdlm(dict[:spin_path])
    t = readdlm(dict[:temp_path])
    Ldim = parse(Int, dict[:Ldim])
    sp_r = reshape(sp, (Ldim,Ldim,:))
    t = reshape(round.((t .-1.8)*10),:)
    sp_r = (sp_r .+ 1)/2
    # per_idx = Random.randperm(size(t,1))
    return sp_r, t
end

function add_ising_data()
    dict_path = "/nobackup/jtoledom/Ising/Dicts/"
    dict_file = readdir(dict_path)[1]
    dict = load_dict(dict_path * dict_file)
    sp = readdlm(dict[:spin_path])
    t = readdlm(dict[:temp_path])
    Ldim = parse(Int, dict[:Ldim])
    sp_r = reshape(sp, (Ldim,Ldim,:))
    t = reshape(round.((t .-1.8)*10),:)
    sp_r = (sp_r .+ 1)/2
    per_idx = Random.randperm(size(t,1))
    return sp_r[:,:,per_idx], t[per_idx]
end

function add_ising_data(c::Int)
    dict_path = "/nobackup/jtoledom/Ising/Dicts/"
    dict_file = readdir(dict_path)[1]
    dict = load_dict(dict_path * dict_file)
    sp = readdlm(dict[:spin_path])
    t = readdlm(dict[:temp_path])
    Ldim = parse(Int, dict[:Ldim])
    sp_r = reshape(sp, (Ldim,Ldim,:))
    t = reshape(round.((t .-1.8)*10),:)
    sp_r = (sp_r .+ 1)/2
    per_idx = Random.randperm(size(t,1))
    idx_tr = per_idx[1:Int(0.8*size(t,1))]
    idx_te = per_idx[1+Int(0.8*size(t,1)):end]
    return sp_r[:,:,idx_tr], t[idx_tr], sp_r[:,:,idx_te], t[idx_te]
end

function load_data(hparams; dataset="MNIST")
    # Load MNIST or CIFAR dataset
    if dataset=="MNIST"
        images, labels = MLDatasets.MNIST.traindata(Float32)
        l=size(images,3)
    elseif dataset=="CIFAR"
        images, labels = MLDatasets.CIFAR10.traindata(Float32)
        l=size(images,4)
    elseif dataset=="ISING"
        images, labels = add_ising_data()
        l=size(images,3)
    end
    # Normalize to [-1, 1]
    image_tensor = reshape(@.(2f0 * images - 1f0), hparams.im_size, hparams.im_size, hparams.im_chn, :)
    y = float.(Flux.onehotbatch(labels, 0:hparams.nclasses-1))
    # Partition into batches
    data = [(image_tensor[:, :, :, r], y[:, r]) for r in Iterators.partition(1:l, hparams.batch_size)]
    return data
end

function load_data(hparams, c::Int; dataset="MNIST")
    # Load MNIST dataset
    if dataset=="MNIST"
        images, labels = MLDatasets.MNIST.traindata(Float32)
        images_t, labels_t = MLDatasets.MNIST.testdata(Float32)
        l=size(images,3)
        l_t=size(images_t,3)
    elseif dataset=="CIFAR"
        images, labels = MLDatasets.CIFAR10.traindata(Float32)
        images_t, labels_t = MLDatasets.CIFAR10.testdata(Float32)
        l=size(images,4)
        l_t=size(images_t,4)
    elseif dataset=="ISING"
        images, labels, images_t, labels_t = add_ising_data(1)
        l=size(images,3)
        l_t=size(images_t,3)
    end
    # images, labels = MLDatasets.MNIST.traindata(Float32)
    # images_t, labels_t = MLDatasets.MNIST.testdata(Float32)
    # Normalize to [-1, 1]
    image_tensor = reshape(@.(2f0 * images - 1f0),  hparams.im_size, hparams.im_size, hparams.im_chn, :)
    image_tensor_t = reshape(@.(2f0 * images_t - 1f0),  hparams.im_size, hparams.im_size, hparams.im_chn, :)
    y = float.(Flux.onehotbatch(labels, 0:hparams.nclasses-1))
    y_t = float.(Flux.onehotbatch(labels_t, 0:hparams.nclasses-1))
    # Partition into batches
    data = [(image_tensor[:, :, :, r], y[:, r]) for r in Iterators.partition(1:l, hparams.batch_size)]
    data_test = [(image_tensor_t[:, :, :, r], y_t[:, r]) for r in Iterators.partition(1:l_t, hparams.batch_size)]
    return data, data_test
end

struct Discriminator
        d_common
end

function genDisc(args)
    N1 =  Chain(
            Conv((4, 4), args.im_chn => 64; stride = 2, pad = 1),
            x->leakyrelu.(x, 0.2f0),
            Dropout(0.25),
            Conv((4, 4), 64 => 128; stride = 2, pad = 1),
            x->leakyrelu.(x, 0.2f0),
            Dropout(0.25),
            x->reshape(x, Int(args.im_size/4) * Int(args.im_size/4) * 128, :),
            Dense(Int(args.im_size/4) * Int(args.im_size/4) * 128, 1)) |> gpu
    return Discriminator(N1)
end

function (m::Discriminator)(x)
        z = m.d_common(x)
        return z
end

struct Generator
        g_common
end

function genGen(hparams)
    N1 = Chain(
            Dense(hparams.latent_dim, Int(hparams.im_size/4) * Int(hparams.im_size/4) * 256),
            BatchNorm(Int(hparams.im_size/4) * Int(hparams.im_size/4) * 256, relu),
            x->reshape(x, Int(hparams.im_size/4), Int(hparams.im_size/4), 256, :),
            ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2),
            BatchNorm(128, relu),
            ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1),
            BatchNorm(64, relu),
            ConvTranspose((4, 4), 64 => hparams.im_chn, hardtanh; stride = 2, pad = 1),
            ) |> gpu
    return Generator(N1)
end

function (m::Generator)(x)
        z = m.g_common(x)
        return z
end

struct Encoder
    enc
    toLatentμ
    toLatentσ
    toLabel
end

function encoder(args; model="MNIST")
    if model == "MNIST"
        enc = Chain(ConvTranspose((4, 4), args.im_chn=>64; stride=1, pad=1),
            BatchNorm(64, leakyrelu),
            Conv((4, 4), 64=>128; stride=2, pad=1),
            BatchNorm(128, leakyrelu),
            Conv((4, 4), 128=>args.enc_c; stride=2, pad=1),
            # Dropout(0.4),
            x->reshape(x,:,size(x,4))) |> gpu
        toLatentμ = Chain(Dense(args.enc_l,100), BatchNorm(100, leakyrelu)) |>gpu
        toLatentσ = Chain(Dense(args.enc_l,100), BatchNorm(100, leakyrelu)) |>gpu
        toLabel = Chain(Dense(args.enc_l,10), BatchNorm(10, leakyrelu), softmax)|>gpu
    elseif model == "ISING"
        enc = Chain(ConvTranspose((4, 4), hparams.im_chn=>64, leakyrelu; stride=1, pad=1),
            BatchNorm(64),
            Conv((4, 4), 64=>128, leakyrelu; stride=2, pad=1),
            BatchNorm(128),
            Conv((4, 4), 128=>256, leakyrelu; stride=2, pad=1),
            # Dropout(0.4),
            BatchNorm(256),
            Conv((4, 4), 256=>512, leakyrelu; stride=2, pad=1),
            BatchNorm(512),
            x->reshape(x,:,size(x,4))) |> gpu
        toLatentμ = Chain(Dense(args.enc_l,100), BatchNorm(100, leakyrelu)) |>gpu
        toLatentσ = Chain(Dense(args.enc_l,100), BatchNorm(100, leakyrelu)) |>gpu
        toLabel = Chain(Dense(args.enc_l,10, leakyrelu), BatchNorm(10), softmax)|>gpu
    end
    return Encoder(enc, toLatentμ, toLatentσ, toLabel)
end

function (m::Encoder)(x)
    h = m.enc(x)
    μ, logσ, y = m.toLatentμ(h), m.toLatentσ(h), m.toLabel(h)
    z = μ + exp.( logσ) .* (randn(size(logσ))|>gpu)
    return z, μ, logσ, y
end

struct Classifier
    c_common
end

function genClassifier(args)
    NN1 = Chain(
        # First convolution, operating upon a 28x28 image
        Conv((3, 3), args.im_chn=>16, pad=(1,1), relu),
        MaxPool((2,2)),

        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),

        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        MaxPool((2,2)),

        # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
        flatten,
        Dense(args.clas_num, 10)) |> gpu

    return Classifier(NN1)
end

# VGG16 and VGG19 models
function vgg16(args)
    NN1 = Chain(
            Conv((3, 3), args.im_chn => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            MaxPool((2,2)),
            Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            MaxPool((2,2)),
            Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            MaxPool((2,2)),
            Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            MaxPool((2,2)),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            MaxPool((2,2)),
            flatten,
            Dense(512, 4096, relu),
            Dropout(0.5),
            Dense(4096, 4096, relu),
            Dropout(0.5),
            Dense(4096, 10)) |> gpu
    return Classifier(NN1)
end

function vgg19(args)
    NN1 =  Chain(
            Conv((3, 3), args.im_chn => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            MaxPool((2,2)),
            Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            MaxPool((2,2)),
            Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2,2)),
            Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2,2)),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2,2)),
            flatten,
            Dense(512, 4096, relu),
            Dropout(0.5),
            Dense(4096, 4096, relu),
            Dropout(0.5),
            Dense(4096, 10)) |> gpu
    return Classifier(NN1)
end

function SEBlock(chn, hw; r=2)
    NN = Chain(MeanPool((hw,hw)),
                x->reshape(x, :, size(x,4)),
                Dense(chn, Int(chn/r),leakyrelu),
                Dense(Int(chn/r),chn),
                Dense(chn,hw*hw*chn,x->x),
                x->reshape(x,hw,hw,chn,size(x,2))
                )
    NN2 = SkipConnection(NN, (mx, x) -> mx .+ x)
    return NN2
end

function ResidualBlock(chn, hw)
    NN = Chain(BatchNorm(chn),
        Conv((3,3), chn=>chn, leakyrelu, pad=(1,1)),
        BatchNorm(chn),
        Conv((3,3), chn=>chn, leakyrelu, pad=(1,1)),
        SEBlock(chn, hw))
    NN2 = SkipConnection(NN, (mx, x) -> mx .+ x)
    return NN2
end

function ResidualBlock(chn)
    NN = Chain(BatchNorm(chn),
        Conv((3,3), chn=>chn, leakyrelu, pad=(1,1)),
        BatchNorm(chn),
        Conv((3,3), chn=>chn, leakyrelu, pad=(1,1)))
    NN2 = SkipConnection(NN, (mx, x) -> mx .+ x)
    return NN2
end

function Baby_ResNet(args; flag=true)
    # chn_list = [(64,3,16,32),(128,3,8,16),(256,5,4,8),(512,3,2,4),(1024,3,1,2)]
    chn_list = [(64,3,16,32),(128,4,8,16),(256,6,4,8),(512,3,2,4)]
    # chn_list = [(64,2,16,32),(128,2,8,16),(256,2,4,8),(512,2,2,4)]
    # chn_list = [(64,2,16,32),(128,2,8,16)]
    res_array=Vector{Any}([Conv((3,3), args.im_chn=>chn_list[1][1], pad=(1,1))])
    for j in 1:size(chn_list,1)
        for i in 1:chn_list[j][2]
            # if i == chn_list[j][2] && flag
            if flag
                push!(res_array, ResidualBlock(chn_list[j][1],chn_list[j][4]))
            else
                push!(res_array, ResidualBlock(chn_list[j][1]))
            end
        end
        push!(res_array, MeanPool((2,2)))
        if j < size(chn_list,1)
            push!(res_array, BatchNorm(chn_list[j][1]))
            push!(res_array, Conv((3,3), chn_list[j][1]=>chn_list[j+1][1], pad=(1,1)))
            push!(res_array, Dropout(0.5))
        else
            push!(res_array, flatten)
            push!(res_array, Dense(chn_list[end][3]^2 * chn_list[end][1],4096, leakyrelu))
            push!(res_array, Dropout(0.5))
            push!(res_array, Dense(4096,10))
            push!(res_array, BatchNorm(10))
        end
    end

    NN1 = Chain(res_array...) |> gpu
    return Classifier(NN1)
end

function (m::Classifier)(x)
    ŷ = m.c_common(x)
    return ŷ
end

augment(x) = x .+ gpu(0.1f0*randn(eltype(x), size(x)))

accuracy(x, y, model) = mean(Flux.onecold(cpu(model(x))) .== Flux.onecold(cpu(y)))

function genPCA(gen; idx=0)
    z = randn(hparams.latent_dim, hparams.batch_size) |> gpu
    ŷ = gen(z)    |> cpu

    xArr[1+idx*hparams.batch_size:(idx+1)*hparams.batch_size, :] = Array(reshape(ŷ, 28*28, hparams.batch_size)')
end

function myPlots(dataA, l_d, l_g, gen::Generator, opt; idxIn=0, svfig=false,
    filename="plot.png", timestamp=false, modelname="Gen.bson", printplots=true)
    idxIn == 0 ? idx = rand(1:size(dataA,1)-1) : idx = idxIn
    x = dataA[idx][1]
    z = randn(hparams.latent_dim, hparams.batch_size) |> gpu
    ŷ = gen(z)    |> cpu

    im_x = hcat([x[:,:,1,j] for j in 1:5]...)
    im_ŷ = hcat([ŷ[:,:,1,j] for j in 1:5]...)
    for i in 1:4
        # r_x = x[:,:,:,5*i + 1:5*i + 5]
        r_x = hcat([x[:,:,1,5*i+j] for j in 1:5]...)
        im_x = vcat(im_x, r_x)

        r_ŷ = hcat([ŷ[:,:,1,5*i+j] for j in 1:5]...)#ŷ[:,:,:,5*i + 1:5*i + 5]
        im_ŷ = vcat(im_ŷ, r_ŷ)
    end

    fig1 = heatmap(im_x, c=cgrad(:cividis))
    fig2 = heatmap(im_ŷ, c=cgrad(:cividis))
    # fig3 = heatmap(im_y)
    # fig4 = heatmap(im_y .- im_ŷ)
    fig3 = scatter(l_d, label="loss D", xlabel=:epochs)
    fig4 = scatter(l_g, label="loss G", xlabel=:epochs)
    f = plot(fig1, fig2, fig3, fig4, size=(2000,2000), dpi=300,
            layout = @layout grid(2,2))

    if svfig
        timestamp ? (dt = now(); filename = "plot_$dt.png";
        modelname = "Gen_$dt.bson";) : nothing

        isdir(PATH_out * "Plots") || mkdir(PATH_out * "Plots")
        isdir(PATH_out * "Plots/$(parsed_args["n"])") || mkdir(PATH_out * "Plots/$(parsed_args["n"])")
        push!(dict[:Plots], PATH_out * "Plots/$(parsed_args["n"])/" * filename)
        savefig(f, dict[:Plots][end])

        save_model(gen, opt; filename=modelname)

        @save dict[:Path] dict
    end

    printplots ? (display(f); return f;) : nothing
end

function myPlots(dataA, l_e, gen::Generator, enc::Encoder, opt, mean_acc; idxIn=0, svfig=false,
    filename="plotE.png", timestamp=false, modelname="Enc.bson", printplots=true)
    testmode!(gen.g_common, true)
    testmode!(clas.c_common, true)
    idxIn == 0 ? idx = rand(1:size(dataA,1)-1) : idx = idxIn
    x = dataA[idx][1] |> gpu
    z, μ, logσ, y = enc(x)
    ŷ = gen(z)    |> cpu
    x = x |> cpu
    y_random = gen(randn(hparams.latent_dim,hparams.batch_size)|>gpu) |>cpu

    im_x = hcat([x[:,:,1,j] for j in 1:5]...)
    im_ŷ = hcat([ŷ[:,:,1,j] for j in 1:5]...)#reshape(ŷ[:,:,:,1:5],28,28*5)
    im_y = hcat([y_random[:,:,1,j] for j in 1:5]...)#reshape(y_random[:,:,:,1:5],28,28*5)
    for i in 1:4
        r_x = hcat([x[:,:,1,5*i+j] for j in 1:5]...)
        im_x = vcat(im_x, r_x)

        r_ŷ = hcat([ŷ[:,:,1,5*i+j] for j in 1:5]...)#ŷ[:,:,:,5*i + 1:5*i + 5]
        im_ŷ = vcat(im_ŷ, r_ŷ)

        r_y = hcat([y_random[:,:,1,5*i+j] for j in 1:5]...)#y_random[:,:,:,5*i + 1:5*i + 5]
        im_y = vcat(im_y, r_y)
    end

    fig1 = heatmap(im_x, c=cgrad(:cividis))
    fig2 = heatmap(im_ŷ, c=cgrad(:cividis))
    fig3 = heatmap(im_y, c=cgrad(:cividis))
    # fig4 = heatmap(im_y .- im_ŷ)
    fig4 = scatter(l_e, label="loss E", xlabel=:epochs)
    # fig4 = scatter(l_g, label="loss G", xlabel=:epochs)
    fig5 = scatter(mean_acc, label="Accuracy", xlabel="epochs", legend=:topleft)
    f = plot(fig1, fig2, fig3, fig4, fig5, fig5, size=(1000,1000), dpi=300,
            layout = @layout grid(3,2))

    if svfig
        timestamp ? (dt = now(); filename = "plotE_$dt.png";
        modelname = "Enc_$dt.bson";) : nothing

        isdir(PATH_out * "Plots") || mkdir(PATH_out * "Plots")
        isdir(PATH_out * "Plots/$(parsed_args["n"])") || mkdir(PATH_out * "Plots/$(parsed_args["n"])")
        push!(dict[:Plots], PATH_out * "Plots/$(parsed_args["n"])/" * filename)
        savefig(f, dict[:Plots][end])

        save_model(enc, opt; filename=modelname, model_type = "Enc")

        @save dict[:Path] dict
    end

    printplots ? (display(f); return f;) : nothing
end

function myPlots(dataA, l, clas::Classifier, opt, mean_acc; idxIn=0, svfig=false,
    filename="plotC.png", timestamp=false, modelname="Class.bson", printplots=true, dataset="MNIST")


    fig3 = scatter(mean_acc, label="Accuracy", xlabel="epochs", legend=:topleft)
    fig4 = scatter(l, label="loss C", xlabel=:epochs, legend=:topleft)
    # fig4 = scatter(l_g, label="loss G", xlabel=:epochs)
    f = plot(fig3, fig4, layout = @layout grid(2,1))

    if svfig
        timestamp ? (dt = now(); filename = "plotC_$dt.png";
        modelname = "Clas_$dt.bson";) : nothing

        isdir(PATH_out * "Plots") || mkdir(PATH_out * "Plots")
        isdir(PATH_out * "Plots/$(parsed_args["n"])") || mkdir(PATH_out * "Plots/$(parsed_args["n"])")
        push!(dict[:Plots], PATH_out * "Plots/$(parsed_args["n"])/" * filename)
        savefig(f, dict[:Plots][end])

        save_model(clas, opt; filename=modelname, model_type = "Clas", dataset=dataset)

        @save dict[:Path] dict
    end

    printplots ? (display(f)) : nothing
end

function test_loss(dataA, clas)
    acc_test = []
    for (x,y) in dataA
        x = x|>gpu
        y = y|>gpu
        acc = accuracy(x,y, clas)
        append!(acc_test, acc)
    end
    mean(acc_test)
end

function test_loss(dataA, clas, enc, gen)
    acc_test = []
    for (x,y) in dataA
        x = x|>gpu
        y = y|>gpu
        z, _, _, _ = enc(x)
        x̂ = gen(z)
        acc = accuracy(x̂,y, clas)
        append!(acc_test, acc)
    end
    mean(acc_test)
end

function custom_train!(dataA, disc::Discriminator, gen::Generator, discr_loss,
                generator_loss, g_ps, d_ps, g_opt, d_opt; epoch=0)
    g_mean_loss=[]
    d_mean_loss=[]
    idx_epoch = 0
    for (x,y) in dataA
        if idx_epoch % 400 == 0 && epoch != 0
            idx_g = Int(idx_epoch / 400 + 6 * (epoch-1))
            @show (idx_epoch, idx_g)
            genPCA(gen; idx = idx_g)
        end

        x = x |> gpu
        z = randn(hparams.latent_dim, hparams.batch_size) |> gpu

        d_gs = gradient(()->discr_loss(z,x),d_ps)
        Flux.update!(d_opt,d_ps, d_gs)
        append!(d_mean_loss, discr_loss(z,x))

        g_gs = gradient(()->generator_loss(z),g_ps)
        Flux.update!(g_opt,g_ps, g_gs)
        append!(g_mean_loss, generator_loss(z))

        idx_epoch += 1
    end
    mean(d_mean_loss), mean(g_mean_loss)
end

function custom_train!(dataA, enc::Encoder, g::Generator, clas::Classifier, loss, ps, opt,  kl_q_p)
  testmode!(g.g_common, true)
  testmode!(clas.c_common, true)
  lost_list = []
  for (x,y) in dataA
    x = x |>gpu
    y = y |>gpu
    gs = gradient(() -> loss(x,y), ps)
    Flux.update!(opt, ps, gs)
    append!(lost_list, loss(x,y))
  end
  return mean(lost_list)
end

function custom_train!(dataA, clas::Classifier, loss, ps, opt)
  lost_list = []
  for (x,y) in dataA
    x = x |>gpu
    y = y |>gpu
    gs = gradient(() -> loss(x,y), ps)
    Flux.update!(opt, ps, gs)
    append!(lost_list, loss(x,y))
  end
  return mean(lost_list)
end

function train(; epochs=10, snap=3, dataset="MNIST", printplots=true)
    # hparams = HyperParams()

    dataA = load_data(hparams; dataset=dataset)

    disc = genDisc(hparams)
    gen = genGen(hparams)

    function discr_loss(z,x; gen=gen, disc=disc)
        x_fake = gen(z)
        real_output = disc(x)
        fake_output = disc(x_fake)
        real_loss = mean(Flux.Losses.logitbinarycrossentropy.(real_output, 1f0))
        fake_loss = mean(Flux.Losses.logitbinarycrossentropy.(fake_output, 0f0))
        return (real_loss + fake_loss)
    end

    generator_loss(z; gen=gen, disc=disc) = (x_fake = gen(z); fake_output = disc(x_fake);
            mean(Flux.Losses.logitbinarycrossentropy.(fake_output, 1f0)))

    d_opt = Flux.OADAM(hparams.lr_dscr, (0.5, 0.99))
    g_opt = Flux.OADAM(hparams.lr_gen, (0.5, 0.99))

    d_ps = params(disc.d_common)
    g_ps = params(gen.g_common)

    myPlots(dataA, 1, 1, gen, g_opt, printplots=printplots)
    mean_loss_d, mean_loss_g = [], []
    for epoch in 1:epochs
        @info epoch
        l_d, l_g = custom_train!(dataA, disc, gen, discr_loss, generator_loss,
                        g_ps, d_ps, g_opt, d_opt)
        @show l_d, l_g
        append!(mean_loss_d, l_d), append!(mean_loss_g, l_g)
        if epoch % snap == 0
            myPlots(dataA, mean_loss_d, mean_loss_g, gen, g_opt; svfig=true,
                timestamp=true, printplots=printplots)
        end
    end
    mean_loss_d, mean_loss_g, disc, gen, d_opt, g_opt
end

function train(g::Generator, clas::Classifier ; epochs=2, snap=1, dataset="MNIST",
            printplots=true)
    # hparams = HyperParams()
    # dataA = load_data(hparams; dataset=dataset);
    dataA, dataA_test = load_data(hparams, 1; dataset=dataset)
    encode = encoder(hparams; model=dataset)
    testmode!(g.g_common, true)
    testmode!(clas.c_common, true)

    λ = 100f0
    kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- (2 .* logσ))
    # crossent(y1,y2) = -sum(y2 .* log.(y1 .+ 0.00000000001f0))
    M = hparams.batch_size
    loss(x, y; encode=encode, g=g, clas=clas, kl_q_p=kl_q_p) = ((z, μ, logσ, ŷ) = encode(x);
            h3 = g(z); w = clas(h3); Flux.hinge_loss(h3,x) + λ * Flux.Losses.logitcrossentropy(w, y)
            + (kl_q_p(μ, logσ))* 1 // M ) #* 1 // M

    opt = ADAM(0.0001, (0.9, 0.8))
    ps = Flux.params(encode.enc, encode.toLatentμ, encode.toLatentσ, encode.toLabel)

    myPlots(dataA, 1, g, encode, opt, 1; printplots=printplots)
    mean_loss_enc = []
    mean_acc = []
    for ep = 1:epochs
        @info "Epoch $ep"
        l_e = custom_train!(dataA, encode, g, clas, loss, ps, opt, kl_q_p)
        append!(mean_loss_enc, l_e)
        append!(mean_acc, test_loss(dataA_test, clas, encode, g))
        @info l_e, mean_acc[end]
        if ep % snap == 0
            myPlots(dataA, mean_loss_enc, g, encode, opt, mean_acc; svfig=true, timestamp=true,
            printplots=printplots)
        end
    end
    mean_loss_enc, encode, opt
end

function train(a::Int; epochs=10, snap=3, dataset="MNIST", printplots=true)
    # hparams = HyperParams()
    @info "Training Classifier"

    dataA, dataA_test = load_data(hparams, 1; dataset=dataset)

    if dataset == "MNIST"
        clas = genClassifier(hparams)
    elseif dataset == "CIFAR"
        # clas = vgg16(hparams)
        # clas = vgg19(hparams)
        clas = Baby_ResNet(hparams)
    elseif dataset=="ISING"
        clas = genClassifier(hparams)
    end

    loss(x, y) = (x̂ = augment(x); ŷ = clas(x̂); Flux.Losses.logitcrossentropy(ŷ, y))

    opt = ADAM(3e-5)

    ps = params(clas.c_common)

    myPlots(dataA_test, 1, clas, opt, 1; printplots=printplots)
    mean_loss = []
    mean_acc = []
    for epoch in 1:epochs
        @info epoch
        l = custom_train!(dataA, clas, loss, ps, opt)
        append!(mean_loss, l)
        append!(mean_acc, test_loss(dataA_test, clas))
        @show l, mean_acc[end]
        if epoch % snap == 0
            myPlots(dataA_test, mean_loss, clas, opt, mean_acc; svfig=true,
                timestamp=true, printplots=printplots, dataset=dataset)
        end

        if mean_acc[end] >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end
    end
    mean_loss, mean_acc, clas, opt
end

function continue_train!(disc, gen, d_opt, g_opt, mean_loss_d, mean_loss_g;
            epochs=10, snap=3, dataset="MNIST", printplots=true)
    # hparams = HyperParams()

    dataA = load_data(hparams; dataset=dataset)

    # disc = genDisc(hparams)
    # gen = genGen(hparams)

    function discr_loss(z,x; gen=gen, disc=disc)
        x_fake = gen(z)
        real_output = disc(x)
        fake_output = disc(x_fake)
        real_loss = mean(Flux.Losses.logitbinarycrossentropy.(real_output, 1f0))
        fake_loss = mean(Flux.Losses.logitbinarycrossentropy.(fake_output, 0f0))
        return (real_loss + fake_loss)
    end

    generator_loss(z; gen=gen, disc=disc) = (x_fake = gen(z); fake_output = disc(x_fake);
            mean(Flux.Losses.logitbinarycrossentropy.(fake_output, 1f0)))

    # d_opt = Flux.OADAM(hparams.lr_dscr, (0.5, 0.99))
    # g_opt = Flux.OADAM(hparams.lr_gen, (0.5, 0.99))

    d_ps = params(disc.d_common)
    g_ps = params(gen.g_common)

    myPlots(dataA, mean_loss_d, mean_loss_g, gen, g_opt, printplots=printplots)
    # mean_loss_d, mean_loss_g = [], []
    for epoch in 1:epochs
        @info epoch
        l_d, l_g = custom_train!(dataA, disc, gen, discr_loss, generator_loss,
                        g_ps, d_ps, g_opt, d_opt)
        @show l_d, l_g
        append!(mean_loss_d, l_d), append!(mean_loss_g, l_g)
        if epoch % snap == 0
            myPlots(dataA, mean_loss_d, mean_loss_g, gen, g_opt; svfig=true,
                timestamp=true, printplots=printplots)
        end
    end
    #mean_loss_d, mean_loss_g, disc, gen, d_opt, g_opt
end

function continue_train!(g::Generator, clas::Classifier , mean_loss_enc, encode, opt, mean_acc;
    epochs=2, snap=1, dataset="MNIST", printplots=true)
    # hparams = HyperParams()
    # dataA = load_data(hparams; dataset=dataset);
    dataA, dataA_test = load_data(hparams, 1; dataset=dataset)
    # encode = encoder(hparams; model=dataset)
    testmode!(g.g_common, true)
    testmode!(clas.c_common, true)

    λ = 100f0
    kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- (2 .* logσ))
    # crossent(y1,y2) = -sum(y2 .* log.(y1 .+ 0.00000000001f0))
    M = hparams.batch_size
    loss(x, y; encode=encode, g=g, clas=clas, kl_q_p=kl_q_p) = ((z, μ, logσ, ŷ) = encode(x);
            h3 = g(z); w = clas(h3); Flux.hinge_loss(h3,x) + λ * Flux.Losses.logitcrossentropy(w, y)
            + (kl_q_p(μ, logσ))* 1 // M ) #* 1 // M

    opt = ADAM(0.0001, (0.9, 0.8))
    ps = Flux.params(encode.enc, encode.toLatentμ, encode.toLatentσ, encode.toLabel)

    myPlots(dataA, mean_loss_enc, g, encode, opt, mean_acc; printplots=printplots)
    #mean_loss_enc = []
    #mean_acc = []
    for ep = 1:epochs
        @info "Epoch $ep"
        l_e = custom_train!(dataA, encode, g, clas, loss, ps, opt, kl_q_p)
        append!(mean_loss_enc, l_e)
        append!(mean_acc, test_loss(dataA_test, clas, encode, g))
        @info l_e, mean_acc[end]
        if ep % snap == 0
            myPlots(dataA, mean_loss_enc, g, encode, opt, mean_acc; svfig=true, timestamp=true,
            printplots=printplots)
        end
    end
    # mean_loss_enc, encode, opt
end

function continue_train!(a::Int, mean_loss, mean_acc, clas, opt;
                epochs=10, snap=3, dataset="MNIST", printplots=true)
    # hparams = HyperParams()
    @info "Training Classifier"

    dataA, dataA_test = load_data(hparams, 1; dataset=dataset)

    loss(x, y) = (x̂ = augment(x); ŷ = clas(x̂); Flux.Losses.logitcrossentropy(ŷ, y))

    # opt = ADAM(3e-5)

    ps = params(clas.c_common)

    myPlots(dataA_test, mean_loss, clas, opt, mean_acc; printplots=printplots)
    # mean_loss = []
    # mean_acc = []
    for epoch in 1:epochs
        @info epoch
        l = custom_train!(dataA, clas, loss, ps, opt)
        append!(mean_loss, l)
        append!(mean_acc, test_loss(dataA_test, clas))
        @show l, mean_acc[end]
        if epoch % snap == 0
            myPlots(dataA_test, mean_loss, clas, opt, mean_acc; svfig=true,
                timestamp=true, printplots=printplots, dataset=dataset)
        end

        if mean_acc[end] >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end
    end
    #mean_loss, mean_acc, clas, opt
end
