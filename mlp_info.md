### **Multi-level Perceptron**

I trained two models for this project. This first model is a simple Multi-level Perceptron without dropout, no CNN, nothing fancy. My goal was to train a fully-connected model on images with the purpose of gaining an intuition for the capabilities and pitfalls of such a design. The most obvious drawback is the size of this learning machine! It requires about five times more storage space than the ViT model I trained, even after compression and optimization with ONNX!

**Fine-Tuning**
The primary challenge in training an MLP, particularly on an imbalanced dataset, is rapid overfitting over the course of even a few epochs. Adding a drop-out layer would likely offer a great improvement. I trained on various metrics, which each quickly improved, to the detriment of other metrics. The model had very mediocre results overall, and not nearly good enough results to use in a medical setting. This model is powerful, but like a souped-up car, it just has too much power to meet safety standards!

**Training**
I created a custom trainer object (in Pytorch) to suit my purposes and ran several experiments. The custom trainer was very helpful for comparing models. I found it even easier to use than an off-the-shelf trainer like Huggingface offers because I designed it efficiently and with just enough flexibility that it is useful without having an overwhelming number of options or requirements for adapting it to a new model.

For each run, I iterated over the training set with enough epochs that the loss and evaluation loss decreased as much as possible before evaluation loss began to increase (the model overfit). As the training loss approached zero, the evaluation loss worsened, as expected, assuring me that the model was able to get the most out of the training with the hyperparameters and methods that I used.

I tried several FC models of varying numbers of layers and nodes, but I was surprised to find that one of the simplest architectures is the best among the designs I tried. It has only three hidden layers! The sizes of each layer are: 150528 (input), 2352, 2352, 294, and 1 (output) node. The input images are all 224 x 224 pixels by 3 channels (8-bit color depth in RGB format).