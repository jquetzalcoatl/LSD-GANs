include("DCGANs.jl")

parsed_args = parseCommandLine(;dir=1)
PATH_out = "/nobackup/jtoledom/GANs/"
# hparams = HyperParams()
#=
FOR CIFAR USE im_chn=3, enc_l=8192, im_size=32, clas_num=512
=#
hparams = HyperParams()
dict = init_dict(hparams)
# dict = load_dict(PATH_out * "/Dicts/1/" * readdir(PATH_out * "/Dicts/1")[1])

mean_loss_d, mean_loss_g, disc, gen, d_opt, g_opt = train(epochs=500, snap=25, dataset="MNIST")
# continue_train(disc, gen, d_opt, g_opt, mean_loss_d, mean_loss_g; epochs=500, snap=25, dataset="CIFAR")
# myPlots(dataA, 1,1, gen, 1; idxIn=0)

num_epochs = parsed_args["e"]
num_snap = parsed_args["snap"]
mean_loss_clas, mean_acc, clas, opt = train(1; epochs=num_epochs, snap=num_snap,
            dataset="MNIST")
# continue_train(1, clas, opt, mean_loss_clas, mean_acc; epochs=10, snap=1, dataset="CIFAR")
# myPlots(dataA_test, 1, clas, 1, 1)

mean_loss_enc, encode, e_opt = train(gen, clas; epochs=500, snap=25, dataset="MNIST")
continue_train(gen, clas, mean_loss_enc, encode, e_opt, [0.82]; epochs=50, snap=25, dataset="MNIST")
# myPlots(dataA, 1, gen, encode, 1)


# gen = load_model(path_from_dict = dict[:ModelsG][end], model_type = "Gen")
#
# clas = load_model(path_from_dict = dict[:ModelsC][end], model_type = "Clas", dataset="CIFAR")
# encode = load_model(path_from_dict = dict[:ModelsE][end], model_type = "Enc")
#
#
# dataA, dataA_test = load_data(hparams, 1)
# dict[:ModelsC]
# test_loss(dataA_test, clas)
