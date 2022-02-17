This folder contains three serialized networks, each containing 2 layers of size 2 and the LeakyReLU activation function:
- initial.json contains weigths that were randomly generated.
- and-trained.json contains weights that were obtained after training the network on the AND boolean function.
- xor-trained.json contains weights that were obtained after training the network on the OR boolean function.
Once the forward propagation method has been implemented, these networks can be deserialized and tested with the BooleanFunctionTester project.