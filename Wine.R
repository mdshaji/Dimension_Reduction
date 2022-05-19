# Loading Wine data

input <- read.csv(file.choose())
View(input)


## Removing unncessary columns
data <- input[, -1]
attach(data)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(data)


# Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

# Box plot Representation

boxplot(Alcohol, col = "orange",main = "Alcohol")
boxplot(Ash, col = "purple",main = "Ash")
boxplot(Alcalinity, col = "red",main = "Alcalinity")
boxplot(Magnesium, col = "dodgerblue4",main = "Magnesium")
boxplot(Color, col = "pink", horizontal = T,main = "Color")

# Histogram Representation

hist(Alcohol,col = "orange", main = "Alcohol" )
hist(Ash,col = "purple", main = "Ash")
hist(Alcalinity,col = "red", main = "Alcalinity")
hist(Magnesium,col = "blue", main = "Magnesium")
hist(Color,col = "pink", main = "Color")


pcaObj <- princomp(data, cor = TRUE, scores = TRUE, covmat = NULL)

str(pcaObj)
summary(pcaObj)

loadings(pcaObj)

plot(pcaObj) # graph showing importance of principal components 

biplot(pcaObj)

plot(cumsum(pcaObj$sdev * pcaObj$sdev) * 100 / (sum(pcaObj$sdev * pcaObj$sdev)), type = "b")

pcaObj$scores
pcaObj$scores[, 1:3]

# Top 3 pca scores 
final <- cbind(input[, 1], pcaObj$scores[, 1:3])
View(final)


# Scatter diagram
plot(final)

##########################################  HIERARCHICAL CLUSTERING  ########################################

input <- read.csv(file.choose())
View(input)


## Removing unnecessary columns
Data <- input[, -1]
attach(Data)

summary(Data)

# Normalize the data
normalized_data <- scale(Data[, 1:13]) 

summary(normalized_data)

# Distance matrix
d <- dist(normalized_data, method = "euclidean") 

fit <- hclust(d, method = "ward.D2")

# Display dendrogram
plot(fit) 
plot(fit, hang = -1)

groups <- cutree(fit, k =14)# Cut tree into 14 clusters

rect.hclust(fit, k =14, border = "red")

cluster <- as.matrix(groups)

final <- data.frame(cluster, Data)

aggregate(Data[, 1:11], by = list(final$cluster), FUN = mean)

library(readr)
write_csv(final, "Wine_R.csv")

getwd()

##########################################  K-MEANS CLUSTERING ############################################

input <- read.csv(file.choose())
View(input)


## Removing unnecessary columns
DATA <- input[, -1]
attach(DATA)
str(DATA)

summary(DATA)

# Normalize the data
normalized_data <- scale(DATA[, 1:13]) # As we already removed "Type" column so all columns need to normalize

summary(normalized_data)

# Elbow curve to decide the k value
twss <- NULL
for (i in 2:13) {
  twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:13, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


# 3 Cluster Solution
fit <- kmeans(normalized_data, 3) 
str(fit)
fit$cluster
final <- data.frame(fit$cluster, DATA) # Append cluster membership

wine <- aggregate(DATA[, 1:13], by = list(fit$cluster), FUN = mean)


library(readr)
write_csv(wine, "Wine_kmeans_R.csv")

getwd()
