Cose da chiedersi:

- (PROF) Fare il dataset solo con features categoriche? Perché le numeriche vengono quasi tutte droppate
- (PROF) Bisogna fare il training pure sulle features venute fuori dal feature importance?
- Abbiamo droppato 148 features con one hot encoding e 147 usando solo le features numeriche (con Pearson), è un problema? Abbiamo fatto un casino col one hot? Scoprilo
- (PROF) Stiamo creando il K-means supervised. Se po' fa'?

TODO:

- Finish only numeric dataset. Don't use dimensionality reduction
- Fix onehot
- Create categorical dataset
- Capisci perché puoi fare lo scaling sulle features, in particolare su quelle che hanno una scala $[0, \infty]$ pur sapendo che non esiste un isometria da $[0, 1]$ a $[0, \infty]$
- Abbiamo provato a concludere deep learning per la regressione, però c'è un problema con lo scaling - con glki stessi dati otteniamo risultati diversi (se non sbaglio per la division del training set e test set utilizziamo lo stesso seed 42).
