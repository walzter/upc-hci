This is the repository for the class Human Computer Interaction (HCI) for the Masters in Artificial Intelligence at the UPC. 


The project that will be presented in this class consists of the following idea: 

Having an interactive table, using a projector, on this we'd be able to overlay games, instruments or a variety of different applications. 

A specific application that we are focusing on is the synthesis of music. 

The music synthesis will be done using Generative Adversarial Networs (GANs) by Goodfellow et al. These models have shown good results when generating synthetic images from a large training set of faces. Additionally, different applications of these models have been seen to be efficient when using different inputs. For instance, Deep Convolutional GANs (DCGAN) combines the strenghts of a Convolutional Neural Network (CNN) with GANs. This is particurlalry interesting when appliying it to music as we need to have two different components: Melody and Harmony. In addition to this, the generation of sound also relies on the combination of different instruments: piano, guitar, bass, trombone, or other instruments. 

Since sound is a sequence, previous methods have shown good sucess as being able to construct or predict future sequences such as RNN's, or LSTMs. However, they lack the ability to combine the the previous instrument with the next one, thus generating a "harmonic" sound. 

The main idea behind this is to create a tool for artists in order to decrease the time that it takes to create a new piece of music. It would allow for fast "prototyping" of the artist's creative process, and being able to save it for later use. 

The milestones for this project relating to the ML part for music synthesis are as follows: 

1. Generate a new beat / base from a set of sample instruments (most likley using the MIDI dataset with different instruments) 
	- Develop an End-to-End model which is able to have as input a specific style (from a given artist) and then replicate it with different instruments. 
	- Given a sample of different audio files, it should be able to generate a harmonic beat in which an artist is able to rap / freestyle on top of it. 

2. Given a beat and a theme or a set of words, generate lyrics which match with the BPM of the output of Step 1 above. The idea is that this would aid the artist when she/he is in an "artistic block" or "writers block". 

3. The final piece of this would be to input an image which would be similar to the theme/topic / set of words that were mentioned in Step 2. Why add an image? Artists are usually very visual and auditive people, so if we're able to generate a sound with a set of lyrics and then represent this in an image that is being transformed over time in combination with the beat and lyrics, it could help to capture the creative process. 

## Methods 

The models that will be used will be the following: 

For Step 1: 
- DCGANs 
- Neural Networks 

For Step 2:
- DNN 
- DCGANs?

For Step 3: 
- VQGANs
- ViT's  
