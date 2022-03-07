%-Attribution de taches-

include "globals.mzn";

array[1..N,1..S] of int: cost; % cout d'une tache dependamment du sous-traitant
array[1..N,1..S] of int: duration; % duree d'une tache dependamment du sous-traitant, en jours

int: S; % nombre de sous-traitants
int: N; % nombre de tâches
int: T; % horizon temporel, en jours

array[1..N] of var 1..S: x;

constraint forall(i in 2..N)(x[i] != x[1]);
constraint forall(i in 3..N)(x[i] != x[2]);

constraint forall(i in 1..S)(T >= sum(j in 1..N where x[j] == i)(duration[j, i]));

var int: C = sum(i in 1..N)(cost[i, x[i]]);


solve minimize C;

output [
  "x: ", show(x), "\n",
  "Cost: ", show(C)]
;