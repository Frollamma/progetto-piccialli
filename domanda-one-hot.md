### One-hot encoding

Dato un sample $x$ con feature categoriche $F_1, \dots, F_n$, rispettivamente con $k_1, \dots, k_n$ categorie, se applichiamo one-hot encoding sulle feature categoriche, si ottiene un nuovo sample $x'$ con $k_1 + \cdots + k_n$ features, poiché ogni feature $F_i$ è stata sostituita con $k_i$ feature. In particolare, il valore corrispondente alla feature $F_i$, è stato sostituito con un vettore di $k_i$ elementi del tipo $(0, \dots, 0, 1, 0, \dots, 0)$, poiché l'unico $1$ si trova all'indice corrispondente alla categoria a cui il sample appartiene. Ognuna delle nuove feature $F_{i,j}'$ così ottenute rappresenta la presenza della categoria $j$-esima della feature categorica $F_i$.

### Coefficiente di Pearson

Ci chiediamo se abbia senso applicare Pearson ad un dataset che contiene feature categoriche. Date due feature in un dataset $X, Y$, il calcolo del coefficiente di correlazione di Pearson è il seguente

$$\rho_{X,Y} = \frac{\operatorname{cov}(X,Y)}{\sigma_{X} \sigma_{Y}} \, ,$$
$$\operatorname{cov}(X, Y) = \mathbb{E}((X - \mu_X)(Y - \mu_Y))$$

dove $\sigma_{X}, \sigma_{Y}, \mu_X, \mu_Y$ sono rispettivamente le deviazioni standard e le medie di $X$ e di $Y$.

### Feature selection usando il coefficiente di Pearson su un dataset con feature categoriche

Sia $F_{i, j}', F_{r, s}'$ due feature dopo aver effettuato il one-hot encoding su feature categoriche. Allora il coefficiente di correlazione di Pearson rappresenta la correlazione fra la presenza della categoria $j$-esima della feature $F_i$ e la presenza della categoria $s$-esima della feature $F_r$, cioè calcoliamo quanto sono correlate queste due categorie. Intuitivamente si capisce che se la presenta di una categoria è molto correlata con quella di un'altra, una delle due informazioni è superflua e quindi ha senso droppare una delle due nuove feature. A seguito di questo ragionamento, a noi sembra possibile poter applicare la feature selection utilizzando il metodo di Pearson su un dataset con features categoriche.
