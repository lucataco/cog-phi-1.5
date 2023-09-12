# microsoft/phi-1_5 Cog model

This is an implementation of the [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="Write a detailed analogy between mathematics and a lighthouse.\nAnswer:"
