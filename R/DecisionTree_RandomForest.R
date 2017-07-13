library(stringr)
library(rpart)

## FUNCTION DECLARATION
undoOneHotEncoding <- function(data, column, newName){
  
  # Check if input column range is valid.
  if (column[1] < 1 || column[1] > ncol(data) || column[length(column)] < 1 || column[length(column)] > ncol(data)){
    stop("Invalid column range.")
  }
  
  # Check if input column range is increasing.
  testRange = seq(column[1], column[length(column)], 1)
  if (length(testRange) != length(column) || !all(testRange == column)){
    stop("Column range not increasing.")
  }
  
  # This approach is too slow!
  #newCol = rep("", nrow(data))
  #for(i in 1:nrow(data)){
  #  currRow = unlist(c(data[i, column]))
  #  index = which(currRow == 1)
  #  print(sprintf("Checking row: %d", i))
  #  newCol[i] = index
  #}
  
  # This approach returns INSTANTLY! Thanks to vectorization.
  newCol = rep("", nrow(data))
  for(i in column){
    newCol = paste0(newCol, data[, i])
  }
  newCol = as.numeric(str_locate(newCol, "1")[, 1])
  
  # Rearrange columns
  # Need a dummy column because one can't append columns into empty data.frame.
  data2 = data.frame(dummycol=rep(NA, nrow(data)))
  
  for(i in 1:ncol(data)){
    if (!(i %in% column)){
      # Just append a new column.
      data2 = cbind(data2, data[, i])
      # Rename the last column so that it matches the original column name.
      names(data2)[ncol(data2)] = names(data)[i]
    } else {
      # Only consider once.
      if (i == column[1]){
        data2 = cbind(data2, newCol)
        names(data2)[ncol(data2)] = newName
      }
    }
  }
  
  # Remove the dummy first column.
  data2 = data2[, -1]
  
  return(data2)
}

# Reference: https://stackoverflow.com/questions/36068963/r-how-to-split-a-data-frame-into-training-validation-and-test-sets
splitData <- function(df, fractionTraining, fractionValidation, fractionTest){
  
  if (!(fractionTraining < 1 && fractionTraining > 0 && fractionValidation < 1 && fractionValidation > 0 && fractionTest < 1 && fractionTest > 0)){
    stop("Invalid ratio(s).")
  }
  
  if (sum(fractionTraining, fractionValidation, fractionTest) != 1.0){
    stop("Ratios don't add up to 1.")
  }
  
  sampleSizeTraining = floor(fractionTraining * nrow(df))
  sampleSizeValidation = floor(fractionValidation * nrow(df))
  sampleSizeTest = floor(fractionTest * nrow(df))
  
  indicesTraining = sort(sample(seq_len(nrow(df)), size=sampleSizeTraining))
  indicesNotTraining = setdiff(seq_len(nrow(df)), indicesTraining)
  indicesValidation = sort(sample(indicesNotTraining, size=sampleSizeValidation))
  indicesTest = setdiff(indicesNotTraining, indicesValidation)
  
  dfTraining = df[indicesTraining, ]
  dfValidation = df[indicesValidation, ]
  dfTest = df[indicesTest, ]
  
  return(list(train=dfTraining, validation=dfValidation, test=dfTest))
}

## Main Flow
data = read.csv("covtype.data")
dataWithCategorical1 = undoOneHotEncoding(data, c(11:14), "wilderness")
dataWithCategorical2 = undoOneHotEncoding(dataWithCategorical1, c(12:51), "soil")

# Make wilderness and soil into string.
dataFinal = dataWithCategorical2
dataFinal$wilderness = as.factor(dataFinal$wilderness)
dataFinal$soil = as.factor(dataFinal$soil)
dataFinal[, ncol(dataFinal)] = as.factor(dataFinal[, ncol(dataFinal)])

# Assign names for better viewing.
names(dataFinal) = c("Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area", "Soil_Type", "Cover_Type")

# Split data into Training/Validation/Testing
resSplit = splitData(dataFinal, 0.8, 0.1, 0.1)
dataFinal_train = resSplit$train
dataFinal_validation = resSplit$validation
dataFinal_test = resSplit$test

# Now data is clean for analysis.
# http://stat.ethz.ch/R-manual/R-patched/library/rpart/html/rpart.control.html
# https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf
# There is no param for determining the maximum number of bins to discretize numerical variables. (like in Spark MLLib)
Sys.time()
model = rpart(Cover_Type ~ ., method = "class", data = dataFinal_train, parms = list(split = "gini"), maxdepth=4)
Sys.time()

