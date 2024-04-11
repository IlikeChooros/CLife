# Skrypt na film o sieciach neuronowych

General rules:
	- programming should be done only if the solution is correct or final
	- i'm showing how to acctually implement the project (not the visualizations)

Films:

	1. First step to ML
	2. Digit recognition
	3. Optimizing the gradient descent with multithreading and GPU
	4. How to visualize the network and learning process


## First step to ML

1. Wstęp:
	- Pokazanie projektów:
		- uczenie się modelu na wykresie z punktami
		- rysowanie cyfr
	- Ogólny wstęp do filmu, omównienie co będzie w tej serii:
		* Zaprogramowanie sieci neuronowej w C++ (what is ML, linear model, gradient descent, backpropagation, implementation in C++)
		* Rozpoznawanie cyfr (ADAM, o normalizacji i transformacji danych)
		* Dodatkowe optymalizacje (GPU, code refactor)
	* Ogólny wstęp:
		- Czym jest ML?
		- Do jakich problemów się przydaje i omówienie szerokiego zastosowania ML w np. algortymie YT, translatorze	

2. Progamming the neural network in C++

	1. Przejście do środowiska i pokazanie struktury projektu (powidzieć, że jest to setup):
		1. VS Code
		2. C++ complier, (gdb)
		3. CMake Extension (no tutorial on the CMake itself - not the scope of the video)
		4. SFML
	2. Showcase on the simplest problem:
		- we have 2 characteristics, plot them on the graph and see colleration
		- showcase on the linear model (maybe: size of the house and it's price)
		- we want to achieve prediction
		- SHOWCASE:
			- LINEAR MODEL:
				- create a simple example with 1 weight and 1 bias (all bearbone)
				- show the working on the graph (no code included), and teach by hand
			- NEURAL NET:
				- add more weights and biases and see how it affects the output
				- about non-linear effect: activation function
				- teaching by hand the network
		- PROGAMMING:
			- implementing the Layer and basic Neural Network (basically feed_forward and basic methods)
			- without the gradient descent
	3. TEACHING THE NETWORK:
		- about gradient descent in math
		- the derivative (basic example)
		- CALCULUS:
			- the derivative in proper form
			- chain rule
			- explain the backpropagation
				- on linear model
				- on 5,4,3,2 neural network
				- whole math model
	4. PROGRAMMING:
		- implement gradient descent (with refactored code):
  		- the gradient descent should be implemented with mini-batch support
		- show the lack of normalization of data:
			- first try of teaching the model with raw data
			- normalize the data
		- teach the network with proper data
		- show more advanced figures on the graph
	5. SUMMARY:
		- brief summary of each point
		- tell what will be in the next episode
  
## Digit recognition

1. Optimizing the gradient descent
   - ADAM optimization:
     - what is it?
     - how it works? with math explanation
     - implementation in code
     - showcase on the graph, and comparison with the basic gradient descent
   - MNIST dataset:
     - what is it? and showcase the dataset
     - how to load it in C++ (into the vector)
     - (visualization of the dataset in SFML not implemented)
   - Normalization of the data:
      - implementation of the normalization in C++
  - Teaching the network:
    - showcase the lack of transformation of the data:
       - explain the problem: the network is learning specific features of the data and not the digit patterns
    - transformation of the data (simple movement of the digit):
      - how to transform the data to be more general
      - implementation in C++ of `transform_data` method
      - showcase the transformation of single digit
    - teaching the network with transformed data
      - showcase the learning process
      - showcase the testing results
      - test the network with writing the digit on the screen
    - Further optimization:
      - rotate the image
      - add noise to the image
      - showcase an example of the transformed data
    - Final results with the transformed data:
      - teaching the network with new data
      - showcase the testing results
      - test the network with writing the digit on the screen

## Optimizing the gradient descent with multithreading and GPU

1. About the optimization:
	 - why we need it?
	 - what is the bottleneck of the network?
	 - how to optimize the network?
2. Multithreading:
	 - what is it?
	 - how to implement it in C++? (std::thread)
	 - PROGRAMMING:
  	- neccessary changes in the code: (add mutex, structure _FeedData)
  	- add multithreading to the learning process
3. GPU