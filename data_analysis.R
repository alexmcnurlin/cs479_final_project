gamedata <- read.csv('https://github.com/alexmcnurlin/cs479_final_project/raw/haydens_working_branch/ml_data.csv')
summary(gamedata)
str(gamedata)
options(scipen=999)
############### Packages ###############
#install.packages('car')
library(magrittr)
library(anchors)
library(ggpubr)
library(car)
library(dplyr)

############ DATA CLEANUP #################
plot(gamedata$Metacritic)
#gamedata$Metacritic <- as.factor(gamedata$Metacritic)
gamedata$QueryName <- as.character(gamedata$QueryName) 
gamedata$ResponseName <- as.character(gamedata$ResponseName) 
gamedata$ReleaseDate <- as.Date(gamedata$ReleaseDate, '%b %d %Y')
gamedata <- replace.value(gamedata, c('Metacritic'), 0, as.integer(NA))
# gamedata$PriceInitial[gamedata$PriceInitial > 200]
# gamedata <- gamedata %>% mutate(year = format(ReleaseDate, "%Y"))
with(gamedata, hist(PriceInitial))
gamedata$PriceInitial[gamedata$PriceInitial > 100]
gamedata$PriceMax <- gamedata$PriceInitial
gamedata$PriceMax[gamedata$PriceMax > 100] <- as.integer('100')



############## PLOTS ###################
# par(mfrow=c(1,1))
plot(gamedata$PriceInitial ~ gamedata$Metacritic,
     ylim = c(0,160),
     main = "Initial Price by Metacritic Score",
     xlab = 'Metacritic Score',
     ylab = 'Initial Price',
     col = 'red'
)
plot(gamedata$PriceInitial ~ gamedata$Metacritic,
     ylim = c(0,65),
     main = "Initial Price by Metacritic Score (Closer Look)",
     xlab = 'Metacritic Score',
     ylab = 'Initial Price',
     col = 'red'
)
# Both plots look pretty much identical
gamedata$SteamSpyOwnersbyMill <- gamedata$SteamSpyOwners/1000000
plot(gamedata$PriceInitial ~ gamedata$SteamSpyOwnersbyMill,
     #ylim = c(0,65),
     main = "Initial Price by Number of Owners",
     xlab = 'Number of Owners (millions)',
     ylab = 'Initial Price',
     col = 'blue'
)
# free games have most owners?

plot(gamedata$PriceInitial ~ gamedata$SteamSpyOwnersbyMill,
     xlim = c(0,16.5),
     main = "Initial Price by Number of Owners (Closer Look)",
     xlab = 'Number of Owners (millions)',
     ylab = 'Initial Price',
     col = 'blue'
)



############### DATA ANALYSIS ###############
plot(gamedata$Metacritic)
corr <- cor.test(gamedata$PriceMax,
                 gamedata$Metacritic,
                 method = 'spearman',
                 use = 'pairwise.complete.obs',
                 exact = FALSE,
                 na.action(na.omit)
                )
corr

# # https://rpubs.com/aaronsc32/spearman-rank-correlation
# gamedata.ranked <- data.frame(cbind(rank(gamedata$Metacritic,
#                                          ties.method = 'random'),
#                                     rank(gamedata$PriceInitial,
#                                          ties.method = 'random')))
# 
# 
# colnames(gamedata.ranked) <- c('metacritic', 'price')
# rho <- cov(gamedata.ranked) / (sd(gamedata.ranked$metacritic) *
#   sd(gamedata.ranked$price))
# rho[[2]]
# n <- length(gamedata.ranked$price)
# r <- cor(gamedata.ranked$price, gamedata.ranked$metacritic, method = 'pearson')
# s <- (n^3 - n) * (1 - r) / 6
# s
# 
# t <- r * sqrt((n - 2) / (1 - r^2))
# p <- 2 * (1 - pt(-abs(t), df = (n - 2)))
# p
# 
# mean(gamedata$Metacritic, na.rm=T)


# no evidence of correlation
#metacritic_lm = lm(Metacritic ~ PriceInitial, data=gamedata)
metacritic_lm = lm(PriceMax ~ Metacritic, data=gamedata)
anova(metacritic_lm)
summary(metacritic_lm)
#cor(gamedata$PriceInitial, gamedata$Metacritic)

# split games into three categories
gamedata$Price <- gamedata$PriceInitial
#gamedata$Price <- cut(gamedata$Price, quantile(gamedata$Price, c(0, 1/4, 1/2, 3/4, 299/300, 1)))
gamedata$Price <- cut(gamedata$Price, quantile(gamedata$Price, c(0, 1/4, 1/2, 3/4, 1)))
levels(gamedata$Price)

metacritic_aov = aov(Metacritic ~ Price, data=gamedata)
anova(metacritic_aov)
TukeyHSD(metacritic_aov)
# There is a significant difference between the  top 1/4 and the middle
# and 1/2 but not between the top 1/4 and the bottom 1/4

# number of people who own the game and the price of the game
ownprice_lm = lm(PriceInitial ~ SteamSpyOwners, data=gamedata)
summary(ownprice_lm)
cor(gamedata$PriceInitial, gamedata$SteamSpyOwners, method='pearson')
str(ownprice_lm)
ownprice_lm$coefficients

par(mfrow=c(2,2))
plot(metacritic_lm)

par(mfrow=c(2,2))
plot(ownprice_lm)
