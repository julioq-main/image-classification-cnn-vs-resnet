# Project Log

## Introduction

I started this project during an internship in the Computer Science & AI
department at the University of Seville. My goal was to bridge the gap between
mathematical and applied ML — a step I see as necessary in order to build a
career as a ML researcher.

Becoming a researcher requires more than coding — it means understanding theory,
learning to design experiments and getting familiar with the tools the field
uses. While this project mainly addresses the coding side of that, it sits
within a broader effort to develop all of those skills in parallel, and here I
set out to adress two of them in particular.

First, I wanted to prove that I can build a pipeline from scratch — using
libraries already available like PyTorch or NumPy — that is clean, modular and
extensible. Not just a script or a Jupyter notebook, but something closer to how
real software is built in practice. The scope is necessarily modest: a 
single-person project built over a few months while studying my Mathematics
degree. But, while staying modest in size it still demonstrates the ability to
structure, manage and scale a project deliberately. 

Second, I wanted to start conducting experiments using the pipeline I was
building. As the project progressed, though, I realized the pipeline itself had
become the primary focus. For this reason and the time constraints I decided to
keep the experiments minimal.

---

## Initial Scope vs. Final Scope

At the beginning, when first scoping the project I thought of implementing a ML
algorithm or small pipeline using both Python and Rust to explore whether Rust
could offer meaningful performance gains over Python in a ML context. However,
given the landscape of libraries and frameworks that the Python community already
offers, I realized that I could not replicate anything meaningful in such a
short timeframe, and that learning at the same time a Python set of tools
and an equivalent set for Rust would take me more time and even be
counterproductive.

So I focused on Python, as it is the de-facto tool for ML, and decided
to create a small pipeline to build, train and test a custom CNN against another
well established architecture like ResNet. The goal was to see how it performed
in a classification task using the 
[Architecture Dataset](https://www.kaggle.com/datasets/wwymak/architecture-dataset)
from Kaggle, with 25 labels and ~4800 images, and if it could beat a net like
ResNet finetuned for this exact task.

While developing the pipeline I added new functionalities:

**Core Pipeline**
- Built-in logger for debugging
- Parsing commands through CLI
- Single file to handle all configuration
- Multiple model architecture support (VGG16, ResNet50, EfficientNet-B0,
ConvNeXt-Tiny)
- Reproducibility controls (seeds: PyTorch, Python, NumPy...)

**Training Controls**
- Model and history saving and checkpoints
- Resuming training from checkpoint 
- Early stopping for patience and loss goal 
- Multiple optimizer support (SGD, Adam, AdamW)
- LR Scheduling (StepLR, Cosine Annealing)
- Pretrained ImageNet weights / fine-tuning support

**Data**
- Support for multiple datasets
- Dataset augmentation
 
**Evaluation**
- Advanced metrics (precision, recall, F1, confusion matrix)
- Automatic plotting during training and testing

As the pipeline grew in scope and complexity, it became clear that building a
custom CNN from scratch would require more time and experimentation than the
project allowed. By that point, the project had already shifted in a different 
direction. So I decided to  implement architectures that had already proven 
their value and compare them against each other, focusing on how architecture
designs evolved through history and how it affected performance.

With that, the final scope is a modular training pipeline supporting multiple
established architectures, optimizers, schedulers and datasets alongside a
minimal set of experiments using the
[Architecture Dataset](https://www.kaggle.com/datasets/wwymak/architecture-dataset).

---

## Key Decisions

### Framework

Once the pipeline idea was established and the Rust implementation discarded,
the first goal was to search for the framework to be used. There were three main
options to consider: Keras with TensorFlow, JAX and PyTorch.

Keras is usually recommended for starters, but its adoption in the research
community has been declining in favour of lower-level frameworks that offer more
flexibility and control. So I discarded the idea as I wanted to learn the tools
commonly used in both research and ML industry.

That led to consider PyTorch and JAX, which have been gaining popularity
over recent years. JAX has a lot of potential and it is the go-to choice for
performance-critical research. However, building a pipeline
with JAX requires working at a very low level, which would have shifted the
project's focus from building a research pipeline towards reimplementing 
foundational components from scratch.

On the other hand, PyTorch is the natural choice for research-oriented ML work.
It offers a rich ecosystem — torchvision, built-in models, optimizers, and LR
schedulers — that covered most of what the pipeline needed without adding
unnecessary complexity. Its compatibility with NumPy and scikit-learn was also
particularly useful for computing and handling evaluation metrics.

---

### Pipeline Architecture

The pipeline is organized into modules with clearly separated responsibilities,
making it easy to extend or replace individual components without touching the
rest of the codebase.

The orchestration layer `main.py` calls all the functions from the pipeline to
handle all the experimentation so all you need to run a new experiment is to
create a `.yaml` configuration file with the parameters of the experiment
— model, optimizer, hyperparameters, dataset... — and run the `main.py` script.
It lives outside the `src/` folder as it is built specifically for the workflow
of this project, while the modules in `src/` can be used to build different
workflows.

**`engine.py`**

It contains the logic to train or evaluate the model through one epoch. The
functions `train_one_epoch()` and `eval_one_epoch()` are the base foundation upon
which the entire training pipeline builds.

**`training`**

Inside `training/` you can find two files: `train.py` and `test.py`. The first
one handles the training loop: it retrieves the optimizer, LR scheduler and 
model — which can be untrained or loaded from checkpoint — through their 
respective modules, then runs the training and validation loop for each epoch,
handling metrics, checkpoints and early stopping. The second file handles
inference on the test set and final evaluation. Both rely on `engine.py` to run
the training and evaluation for each epoch.

**`models`**

`models/architectures.py` contains the model builder, so new architectures can
be added without touching the training and testing logic.

**`utils`**

`utils/` includes a range of helpers for different tasks: setting the logger with
`logger.py`, seeding the experiment with `seeds.py`, computing different metrics
with `metrics.py`, plotting those metrics with `plotting.py` and instantiating
optimizers and schedulers with `optim.py` — keeping that logic out of the 
training loop entirely.

---

### Models

The selected architectures represent key milestones in the evolution of CNN
design, chosen to illustrate how advances in architecture translated to
improvements in performance and efficiency:

- VGG16 (2014): demonstrated that depth using small 3×3 convolutions consistently
outperforms shallower networks with larger filters.
- ResNet50 (2015): introduced residual connections, solving the vanishing gradient
problem and enabling much deeper networks.
- EfficientNet-B0 (2019): proposed compound scaling to balance depth, width, 
and resolution simultaneously.
- ConvNeXt-Tiny (2022): modernized the classic ConvNet design by incorporating 
ideas from Vision Transformers.

The fine-tuned variants are included to compare against the from-scratch 
versions, testing whether pretrained ImageNet weights provide a meaningful 
advantage on a small dataset like this one.

### Dataset
The dataset was suggested by my mentor, as it had been used in previous projects.
The task itself — classifying architectural styles from images — is secondary 
to the pipeline; it serves as a real, multi-class classification problem to 
validate the training pipeline and compare the models against each other.

The dataset used is the [Architecture Dataset](https://www.kaggle.com/datasets/wwymak/architecture-dataset)
from Kaggle: 25 architectural style classes with approximately 4,800 images in
total. It is small enough to train on modest hardware in reasonable time, while 
being sufficiently complex — 25 classes, real-world photographs with varying 
conditions — to produce meaningful comparisons between architectures.

### Ideas Not Implemented

**Out of scope by design**: Features like multi-GPU training or distributed 
computing were deliberately out of scope — this pipeline is intended to run on a
single local machine. The current selection of architectures, optimizers, and 
schedulers is sufficient for the planned experiments, and the modular design 
makes adding new ones straightforward.

**Known limitations**: Since the dataset was fixed from the start, no general 
dataset handling pipeline was implemented. There are no validation or sanity
checks for inconsistent or malformed datasets, which would need to be addressed
before using this pipeline with an arbitrary dataset.

---

## What I Would Do Differently

Looking back, there are a few things I would have done differently.

On the process side, I would have defined the scope more clearly from the start
— specifically, settling earlier on using pretrained architectures rather than
spending time considering a custom CNN. The pivot was the right decision, but it
came later than it should have.

On the technical side, I would have taken a more object-oriented approach. A
`Trainer` class encapsulating the training loop, or a base class for model
architectures, would have made the codebase cleaner and easier to extend. The 
current functional approach works, but OOP would have been a better fit for a 
project designed to scale.

Finally, the project has no unit or integration tests. For a pipeline intended 
to be modular and extensible, that is a real gap — one I would address early if 
starting again.

---

## Future Work

Regarding the pipeline itself, adding proper dataset handling, expanding the
selection of models and optimizers, or supporting more task types — such as 
multi-label classification or segmentation — are interesting directions to
explore. Integrating an experiment tracking tool like MLflow or Weights & Biases
would also be a natural next step.

That said, the project is enough as it is right now and I would like to move on
to more research-oriented work. Outside this project, I want to implement some 
fundamentals from scratch — linear regression, the simplex method, and the 
metrics used here — to deepen my understanding of the underlying mechanics. From
there, I plan to study these architectures in depth by reimplementing them from
their original papers. The goal is to build a habit of reading and implementing
papers across different areas of ML.