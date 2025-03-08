callsDf = read.csv(
          'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Bckrnd_mn_srkw_tkw_offshore_TKW_balanced_4k\\Data\\Better_BackgroundSpread.csv')

nonCalltype = subset(callsDf, CalltypeCategory=='None')
calltypeData = subset(callsDf, CalltypeCategory !='None')


OKWadded = subset(nonCalltype, Ecotype == 'OKW')
TKWadded = subset(nonCalltype, Ecotype == 'TKW')