data <- acquireData()
data <- cleanData(data$training, data$testing)

describe(data$training)
