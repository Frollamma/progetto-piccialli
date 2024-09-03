## Cose da chiedersi

- (PROF) Fare il dataset solo con features categoriche? Perché le numeriche vengono quasi tutte droppate
- Abbiamo droppato 148 features con one hot encoding e 147 usando solo le features numeriche (con Pearson), è un problema? Abbiamo fatto un casino col one hot? Scoprilo
- (PROF) Stiamo creando il K-means supervised. Se po' fa'?
- (PROF) Come funziona il Naive-Bayes quando ho delle feature numeriche, non dovrebbe essere molto probabile che il risultato sia 0?
- (PROF) Abbiamo notato che senza scaling il deep learning regression fa schifo? Mentre con StandardScaler o MinMax scaler migliora notevolmente?

## TODO

- Create categorical dataset
- Capisci perché puoi fare lo scaling sulle features, in particolare su quelle che hanno una scala $[0, \infty)$ pur sapendo che non esiste un isometria da $[0, 1]$ a $[0, \infty)$. Ho capito: praticamente quando fai il min-max scaler vai da $[0, M]$, dove $M$ è il massimo della feature cosndierata e quindi un 'isometria esiste, ma chiaramente se cambia il dataset cambia anche l'isometria, quindi non puoi trovare un'isometria "unica" per tutti i dataset, alsia devi rifare il min max scaler ogni volta.

## Notes

### One-Hot encoding

One-hot encoding: dato un sample, dalla feature F1 con k categorie ottengo un vettore di k elementi (k feature nuove)

$C_i -> (0, 0, 0, ..., 1, ..., 0, 0, 0) = e_i$ ($i$-esimo posto)

Siano $A, B$ 2 features categoriche con $k$ categorie, con one.hot encoding ho un'applicazione del genere
$$(Ai, Bj) -> (ei, ej) = F$$

dove $F$ è un vettore di $2k$ features.

Features dopo one-hot encoding: $F1, ..., F2k$

Cosa significare fare la correlazione di Pearson?

$$cov(F1, F2) = E((F1 - m1)(F2 - m2))$$

Covarianza fra due features che rappresentano rispettivamente la presenza della categoria A1 e A2 (se $k \geq 2$) .Cioè calcoliamo quanto sono correlate le categorie A1 e A2, intuitivamente si capisce che se la presena di una categoria è molto correlata con quella di un'altra una delle due informazioni è superflua e si può droppare quella feature. Quindi applicare Pearson va bene.
