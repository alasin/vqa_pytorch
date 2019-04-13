# Assignment 3: Visual Question Answering with PyTorch!

- Anuj Pahuja (apahuja)

## Task 1: Data Loader (30 points)

**Q1** Give a summary of your dataset implementation. What should be in the \_\_getitem\_\_ dictionary? 
How should that information be encoded as Tensors? If you consider multiple options, explain your thought process in picking. What preprocessing should be done? 

**Answer**:
1. Pre-computed the GoogleNet features for all the images and saved them as `.npy` files. Code is in `compute_googlenet_feat.py`.
2. Pre-computed a vocabulary for both questions and answers from the training set and saved them as a `.json` file. Questions were first tokenized into individual words and were kept in vocabulary only if their count is greater than a threshold (6). Answers were preprocessed to handle punctuation and special characters. The top k (2000) answers were kept in vocabulary. Code is in `compute_vocab.py`.
3. Both the directory containing image features and the vocabulary json filename are passed as parameters to the dataset initializer.
4. Questions are encoded as a binary vector (tensor) of dimension `q_vocab_size` where `q_vocab_size` is the number of words in question vocabulary. The vector's value at an index is 1 if the word corresponding to that index is present in vocabulary.
5. Answers are encoded as integer vectors of dimension `a_vocab_size` where `a_vocab_size` is the number of words in answer vocabulary. The vector's value at an index is `count` where `count` is the number of times it was chosen as an answer by a human. Answers could also be encoded as a one-hot vector where the correct answer is chosen as the majority answer among the candidates (which is already happening but during training). A reason for not doing this during data loading is that `count` is required for evaluating metrics during training/testing.
6. `__getitem__` returns a dictionary with 3 items - 1024-dim image feature, `q_vocab_size`-dim question encoding and `a_vocab_size`-dim answer encoding.


### Deliverables
1. Your response to Q1.
1. A vqa_dataset.py that passes unit tests.

## Task 2: Simple Baseline (30 points)

**Q2** Describe your implementation in brief, focusing on any design decisions you made: e.g what loss and optimizer you used, any training parameters you picked,
how you computed the ground truth answer, etc. If you make changes from the original paper, describe here what you changed and why. 

**Answer**\
Number of epochs: 10\
Loss: Softmax Cross Entropy for 2000 classes\
Optimizer: SGD

Unlike the paper, I didn't do any weight/gradient clipping. I tried it initially but my loss wasn't going down. Original source code of the paper uses a threshold of 3 for an answer to be kept in the vocabulary, I didn't enforce any threshold and just used the top 2000 most frequent answers. Ground truth answer was chosen using a majority vote (argmax of answer occurences). Accuracy was computed as described in the original VQA paper for open-ended task.

### Deliverables
1. Your response to Q2.
1. Implementations in experiment_runner_base.py, simple_baseline_experiment_runner.py, simple_baseline_net.py
1. Graphs of loss and accuracy during training.


## Task 3: Co-Attention Network (30 points)
In this task you'll implement [3]. This paper introduces three things not used in the Simple Baseline paper: hierarchical question processing, attention, and 
the use of recurrent layers. You may choose to do either parallel or alternating co-attention (or both, if you're feeling inspired).

The paper explains attention fairly thoroughly, so we encourage you to, in particular, closely read through section 3.3 of the paper.

To implement the Co-Attention Network you'll need to:

1. Implement CoattentionExperimentRunner's optimize method. 
1. Implement CoattentionNet
    1. Encode the image in a way that maintains some spatial awareness (see recommendation 1 below).
    1. Implement the hierarchical language embedding (words, phrases, question)
        1. Hint: All three layers of the hierarchy will still have a sequence length identical to the original sequence length. 
        This is necessary for attention, though it may be unintuitive for the question encoding.
    1. Implement your selected co-attention method
    1. Attend to each layer of the hierarchy, creating an attended image and question feature for each
    1. Combine these features to predict the final answer

You may find the implementation of the network to be nontrivial. Here are some recommendations:

1. Pay attention to the image encoding; you may want to skim through [5] to get a sense for why they upscale the images.
1. Consider the attention mechanism separately from the hierarchical question embedding. In particular, you may consider writing 
a separate nn.Module that handles only attention (e.g some "AttentionNet"), that the CoattentionNet can then use.
1. Review the ablation section of the paper (4.4). You can see that you can get good results using only simpler subsets of the 
larger network. You can use this fact to test small subnets (e.g images alone, without any question hierarchy at all), then 
slowly build up the network while making sure that training is still proceeding effectively.
1. The paper uses a batch_size of 300, which we recommend using if you can. One way you can make this work is to pre-compute 
the pretrained network's (e.g ResNet) encodings of your images and cache them, and then load those instead of the full images. This reduces the amount of 
data you need to pull into memory, and greatly increases the size of batches you can run.
    1. This is why we recommended you create a larger AWS Volume, so you have a convenient place to store this cache.

Once again feel free to refer to the official Torch implementation: https://github.com/jiasenlu/HieCoAttenVQA

**Q3** As in the above sections, describe your implementation in brief, e.g loss, optimizer, any decisions you made just to speed up training, etc.
 If you make changes from the original paper, describe here what you changed and why. 


### Deliverables
1. Your response to Q3.
1. Implementations in coattention_experiment_runner.py, coattention_net.py
1. Graphs of loss and accuracy during training.


## Task 4: Custom Network  (10 points + 10 bonus points)
Brainstorm some ideas for improvements to existing methods or novel ways to approach the problem. 

For 10 extra points, pick at least one method and try it out. It's okay if it doesn't beat the baselines, we're looking for 
creativity here; not all interesting ideas work out. 

### Deliverables
1. A list of a few ideas (at least 3, the more the better).

For 10 bonus points:

1. Code implementing at least one of the ideas.
    1. If you tweak one of your existing implementations, please copy the network to a new, clearly named file before changing it.
1. Training loss and test accuracy graphs for your idea. 


## Relevant papers:
[1] VQA: Visual Question Answering (Agrawal et al, 2016): https://arxiv.org/pdf/1505.00468v6.pdf

[2] Simple Baseline for Visual Question Answering (Zhou et al, 2015): https://arxiv.org/pdf/1512.02167.pdf

[3] Hierarchical Question-Image Co-Attention for Visual Question Answering (Lu et al, 2017):  https://arxiv.org/pdf/1606.00061.pdf

[4] Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering (Goyal, Khot et al, 2017):  https://arxiv.org/pdf/1612.00837.pdf

[5] Stacked Attention Networks for Image Question Answering (Yang et al, 2016): https://arxiv.org/pdf/1511.02274.pdf
