library(ChemmineR)
library(abind)
library(mxnet)

# Number coding for atoms
AtomNumber = 1:10
names(AtomNumber) = c("H", "C", "N", "O", "S", "P", "F", "Cl", "I", "Br")

## This function mines the atom data from SDF files into a 3D matrix.
## Dimensions: drug, label + atoms, X Y Z under each atom.

sdfTo3dArray = function(sdfMolecules, width, molClass)
{
  nOfSdf = length(sdfMolecules)
  ## Since we have empty fields, create the smallest matrix that will
  ## accommodate the largest molecule, filling empty features with
  ## the maximum integer. This should make the neural network consider
  ## them as outliers.
  Matrix = array(dim=c(4, width, nOfSdf), data=.Machine$integer.max)

  j = 1
  # Iterate through every sdf molecule
  while (j <= nOfSdf)
  {
    # Access jth molecule
    sdfMolecule = sdfMolecules[[j]]
    # Retrieve atom block that contains the atoms and their coordinates
    moleculeAtoms = atomblock(sdfMolecule)
    # Retrieve all atoms
    atoms = rownames(moleculeAtoms)
    # For every atom, assign its name and coordinates in the matrix
    a = 1
    while(a <= length(atoms))
    {
      # first column for every drug is its class
      # subsequent columns are atoms
      Matrix[1,1,j] = molClass # 1 for stimulant
      Matrix[1,a+1,j] = AtomNumber[unlist(strsplit(atoms[a], "_"))[1]] # assign atom
      Matrix[2,a+1,j] = moleculeAtoms[a,1] #assign X
      Matrix[3,a+1,j] = moleculeAtoms[a,2] #assign Y
      Matrix[4,a+1,j] = moleculeAtoms[a,3] #assign Z
      a = a + 1
    }
    j = j + 1
  }
  return(Matrix)
}

# Load molecules
stimulantsSDFs = read.SDFset("stimulants.sdf")
sedativesSDFs = read.SDFset("sedatives.sdf")

# Width of matrix is the maximum number of atoms in a molecule plus 1 (class).
arrayWidth =  max(max(sapply(atomcount(stimulantsSDFs), sum)) + 1,
                  max(sapply(atomcount(sedativesSDFs), sum)) + 1)

# Load and set dummy variable according to class: Stimulant (1) and Sedative (1).
stimulants = sdfTo3dArray(stimulantsSDFs, arrayWidth, 1)
sedatives = sdfTo3dArray(sedativesSDFs, arrayWidth, 0)

# Bind the sedatives and stimulant datasets together.
drugMatrix = abind(stimulants, sedatives, along=3)

## Set seed for number generator. Ensures repeatable results.
set.seed(1)

dataSize = length(drugMatrix[1,1,])

## Separate data into training and testing subsets.
## We will use majority of data to train neural network.
sampleSize = 0.8 * dataSize

trainingSample = sample(dataSize, sampleSize)
testingSample = setdiff(1:dataSize, trainingSample)

trainingSet = drugMatrix[,, trainingSample]
testingSet = drugMatrix[,,testingSample]

## Separate labels of drugs from their SDF input for both testing and training.

trainingX = trainingSet[,-1,]
trainingY = trainingSet[1,1,]

testingX = testingSet[,-1,]
testingY = testingSet[1,1,]

## mxnet requires a 4D input array?
dim(trainingX) = c(4, 59, 1, 98)
dim(testingX) = c(4, 59, 1, 25)

data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(1, 1), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")

## NOTE: When pooing kernel size is small (1,1), accuracy is improved (96%).
## During training, accuracy peaks at 89%.
## Prevents overfitting and forces generalizations? When a kernel size of (2,2)
## is used, accuracy drops to 76%. However, during training, accuracy peaks at 100%.
## Is this evidence of overfitting?

pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(1, 1), stride = c(2, 2))
# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(1, 1), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(1, 1), stride = c(2, 2))
# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)


devices <- mx.cpu()

# Train the model.
model <- mx.model.FeedForward.create(NN_model,
                                     X = trainingX,
                                     y = trainingY,
                                     ctx = devices,
                                     num.round = 300,
                                     array.batch.size = length(trainingY),
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

## Convert output probability matrix for each feature into list of labeled data.

predictProbs <- predict(model, testingX)
predictedLabels <- max.col(t(predictProbs)) - 1

## Evaluate accuracy from a confusion table.
confusionTable = table(predictedLabels, testingY)
accuracy = sum(diag(confusionTable))/25
accuracy
