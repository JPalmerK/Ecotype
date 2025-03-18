

rm(list=setdiff(ls(), c('allanno',"kwSub")))

allAnno = read.csv('E:\\DCLDE\\Annotations.csv')

trainingData = read.csv(
          'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Bckrnd_mn_srkw_tkw_offshore_TKW_balanced_4k\\Data\\Better_BackgroundSpread.csv')

kwSub_noMal = kwSub[kwSub$Provider != 'JASCO_Malahat',]


# Figure out which files are not in the the training data
kwSub_noMal$Training=kwSub_noMal$Soundfile %in% trainingData$Soundfile

kwEval = kwSub_noMal[kwSub_noMal$Training== FALSE,]


nonCalltype = subset(trainingData, CalltypeCategory=='None')
calltypeData = subset(trainingData, CalltypeCategory !='None')


OKWadded = subset(nonCalltype, Ecotype == 'OKW')
TKWadded = subset(nonCalltype, Ecotype == 'TKW')




# Just get a list of files that are not in the training data
NonTrainingFileData = allAnno[!allAnno$Soundfile %in% trainingData$Soundfile,]

# Too many humpback whales
NonTrainingFileData$Label = NonTrainingFileData$ClassSpecies
NonTrainingFileData$Label[NonTrainingFileData$Ecotype == 'SRKW'] = 'SRKW'
NonTrainingFileData$Label[NonTrainingFileData$Ecotype == 'NRKW'] = 'NRKW'
NonTrainingFileData$Label[NonTrainingFileData$Ecotype == 'TKW'] = 'TKW'
NonTrainingFileData$Label[NonTrainingFileData$Ecotype == 'OKW'] = 'OKW'

NonTrainingFileData = NonTrainingFileData[NonTrainingFileData$Provider !='JASCO_Malahat', ]
NonTrainingFileData = NonTrainingFileData[NonTrainingFileData$Label !='KW', ]


write.csv(NonTrainingFileData, 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\fix_mn_srkw_offshore_tkw_balanced_4k\\Data\\nonTrainingData.csv')

