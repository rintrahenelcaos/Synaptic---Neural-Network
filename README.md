# Synaptic.

Neural Network built from scratch

## Description

Neural Network designed as MNIST digit recognizer using numpy as its main library. The objective of this app was to create a scalable neural network with enought costumization so as to be useful for other problems. In its current configuration it is able to solve other problems similar to MNIST. The training and testing data is obtained through keras library for simplicity as it is not the objective of this app to deal with data. However, this can be easily changed by minor modifications.

The app consist basically of two parts, the network creator (net.py) and network modular parts (neurons.py and activations.py). The network creator, net.py from now on, bounds with the modular parts by a configuration file, config.py. This file works as an intermediate importer, as net.py imports list containing objects lists, not only constants. This may make the app quite vulnerable to spell errors in config.py file, but eliminates innecessary code in net.py and opnes the possiblity to create a new interface file for configuration on the run.

neurons.py contains only one class, Dense, which is the empty neuron with its forward and backward pass functions. activations.py contains all the available activations and loss calculation functions with their gradients. 



## Usage

### Dependencies

Libraries:
* BeautifulSoup4
* PyQt5 (both UI and thread)
* SQLite3


### Executing program

To run: scrappe.py. 

The app includes an existing db functioning as test and example.

### Instructions

![Screenshot of the UI with reference numbers in red](scarper2.png)

Once launched the app will inmediately try to scrap the required information required in the db. Once load, it will be shown in the table. Symbol (1) represents the CEDEAR symbol as used in the stock exchange. Price (2) is the real U$S value corrected by the exchange rate (7). The amount of shares in the portfolio is presented in Holding (3) and the value of the holding in sayd share is represented in Holding$ (4). 

To edit the portfolio:
* To add a share: Select the share symbol in the dropdown box and its amount (6). Then press "add specie"
* To edit a share: Double click on the row in the table of the desired share. It allows to change the amount as the rest of the entries are scraped or calculated.
* To delete a share: Click "DEL." (5) button

Total Value of the portfolio is shown at the lower right corner (8).

State commentaries appeared at (9) while scraping and threading situation is presented at (10)



## Author

Leonardo Mario Mazzeo
leomazzeo@gmail.com

## Updates

* 6/10/2024 Version 0.1: due to changes in the original information source webpage the scrapping function has to be changed
* 31/7/2024 Version 1.0: Two more scraping options added to prevent future crashes. 



Neural Network built from scratch.
Designed as MNIST digit recognizer, but can be expanded to solve other problems. 
Uses keras so as to access the database. The UI is limited to console.

To run: net.py
To configurate the neural network use config.py

Includes Tanh, ReLU, Sigmoid and Softmax activation functions.

Loss function alternatives: mean square error and crossentropy. 



