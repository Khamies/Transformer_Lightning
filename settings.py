global_setting = {

"seed": 3111,

}


model_setting = {


"embed_size": 300,
"ffnn_size": 600,
"nhead": 2,
"nlayers": 2,
"dropout": 0.2

}


training_setting = {

"epochs": 20,
"bsz_train": 32,
"bsz_test": 10,
"bptt": 35,
"lr" : 0.001,
"clip": 0.25,
"train_losses": [],
"test_losses": []

}