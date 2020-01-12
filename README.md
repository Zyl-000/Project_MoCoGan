# MoCoGAN: Decomposing Motion and Content for Video Generation
## Introduction:
This repository contains an tensorlayer implementation of [MoCoGAN: Decomposing Motion and Content for Video Generation](http://arxiv.org/abs/1707.04993) .

MoCoGAN is a generative adversarial network which contains two discriminators( discriminator_I for images and discriminator_V for videos) and two generators( contains a GRU model to generator random sequences and generator_I to generate frames from the random sequences). The generated models features separated representation of motion(dim=10 in the random vector) and content(dim=50 in random vector), offering control over what is generated. Theoretically, MoCoGAN can generate the same object performing different actions, as well as the same action performed by different objects.  
## Files:   
new.py ---- the model definition code, which the generator_I model is defined by static model (work in tensorlayer2.1.1)  
train_new.py ---- training code that match the model definitions in new.py  
models ---- the saved weights of four models in MoCoGAN after training

p.s. the model_tl.py and train_tl.py are codes whith the generator_I model was written in dynamic model, but it didn't work in tensorlayer2.1.1

## TODO:
looks like the generators were not trained properly
