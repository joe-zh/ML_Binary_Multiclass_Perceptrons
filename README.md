# ML_Binary_Multiclass_Perceptrons

This project simulates 2 types of standard perceptrons to classify datasets that cannot be extrapolated using generative models.

Generative models ask the question of generating some sort of joint distribution model of given classes and features. Examples include Bayes, Markov, and Hidden Markov Models. Such models, while precise, may not be the most easily computable. Often times it is easier to distinguish between classes of data rather than coming up with a formulated distribution - these types of models are discriminative, and perceptrons is one such example.

The binary and multiclass perceptrons trains the weight vector over a number of input iterations, adjusting the vector based on whether or not an error has been made in classifying the data with the current vector values. Data values are classified into either binary labels (+, -) or multiclass labels with various naming conventions.

Several application examples are also provided to illustrate the power of discriminative modelling, by classifying data that would appear much more difficult if using a generative model such as Bayes or HMM.


