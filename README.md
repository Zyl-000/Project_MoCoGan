## MoCoGAN: Decomposing Motion and Content for Video Generation
# Introduction:
A tensorlayer implementation of Motion Content Generative Adversarial Network

# Files:   
new.py ---- the model definition code, which the generator_I model is defined by static model (work in tensorlayer2.1.1)  
train_new.py ---- training code that match the model definitions in new.py  
models ---- the saved weights of four models in MoCoGAN after training

p.s. the model_tl.py and train_tl.py are codes whith the generator_I model was written in dynamic model, but it didn't work in tensorlayer2.1.1
# TODO:
looks like the generators were not trained properly
