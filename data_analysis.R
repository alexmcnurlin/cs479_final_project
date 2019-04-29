gamedata <- read.csv('/Users/corey/github/cs479_final_project/ml_data.csv')
summary(gamedata)
str(gamedata)

############ DATA CLEANUP #################
gamedata$QueryName <- as.character(gamedata$QueryName) 
gamedata$ResponseName <- as.character(gamedata$ResponseName) 
gamedata$ReleaseDate <- as.Date(gamedata$ReleaseDate, '%b %d %Y')
